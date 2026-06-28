import os
import sys
import cv2
import time
import threading
import argparse
import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from insightface.app import FaceAnalysis
from dotenv import load_dotenv
from datetime import datetime
from visit_log import VisitLogger

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "visit_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT", "55432")
LIVENESS_THRESHOLD = float(os.getenv("LIVENESS_THRESHOLD", "0.5"))
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.35"))
LOW_CONF_THRESHOLD = float(os.getenv("LOW_CONF_THRESHOLD", "0.55"))
RECOGNITION_EVERY_N_FRAMES = 15
parser = argparse.ArgumentParser()
parser.add_argument("--camera", type = int, default = 0)
parser.add_argument("--no-liveness", action = "store_true")
parser.add_argument("--width", type = int, default = 1280)
parser.add_argument("--height", type = int, default = 720)
args = parser.parse_args()
USE_LIVENESS = not args.no_liveness
GREEN = (0, 210, 0)      
BLUE = (200, 100, 0)    
ORANGE = (0, 140, 255)    
YELLOW = (0, 200, 220)    
CYAN = (220, 200, 0)    
RED = (0, 40, 220)
WHITE = (220, 220, 220)
DARK = (30, 30, 30)
try:
    conn = psycopg2.connect(
        host = DB_HOST, 
        database = DB_NAME,
        user = DB_USER, 
        password = DB_PASSWORD,
        port = DB_PORT
    )
    register_vector(conn)
    cursor = conn.cursor()
except psycopg2.OperationalError as e:
    print(f"Cannot connect to database: {e}\nRun: docker compose up -d")
    sys.exit(1)
logger = VisitLogger()
app = FaceAnalysis(name = "buffalo_l")
app.prepare(ctx_id = -1)
liveness_model = None
if USE_LIVENESS:
    try:
        from insightface.model_zoo import get_model as _get_model
        liveness_model = _get_model("antispoof")
        liveness_model.prepare(ctx_id=-1)
    except Exception as e:
        print(f"Anti-spoof unavailable: {e}")
def _search_table(table: str, id_col: str, embedding) -> dict | None:
    """
    Search a single embeddings table.
    Returns None on error, or a dict with keys:
        found, person_id, full_name, distance, verified, low_conf
    """
    try:
        cursor.execute(f"SELECT COUNT(*) FROM {table};")
        if cursor.fetchone()[0] == 0:
            return {"found": False}
        cursor.execute(
            f"""
            SELECT {id_col}, full_name, embedding <=> %s AS distance
            FROM   {table}
            ORDER  BY embedding <=> %s
            LIMIT  1;
            """,
            (embedding, embedding),
        )
        row = cursor.fetchone()
        if row:
            dist = float(row[2])
            return {
                "found":     True,
                "person_id": row[0],
                "full_name": row[1],
                "distance":  dist,
                "verified":  dist < MATCH_THRESHOLD,
                "low_conf":  MATCH_THRESHOLD <= dist < LOW_CONF_THRESHOLD,
            }
    except Exception as e:
        print(f"[DB ERROR] {e}")
        try:
            conn.rollback()
        except Exception:
            pass
    return None
def search_both_tables(embedding):
    """
    Search employees first, then visitors.
    Returns (person_type, result_dict) where person_type is one of:
        'employee', 'frequent_visitor', 'unknown', 'db_error'
    """
    emp = _search_table("employee_embeddings", "emp_id", embedding)
    if emp is None:
        return "db_error", None
    if emp.get("found") and emp.get("verified"):
        return "employee", emp
    if emp.get("found") and emp.get("low_conf"):
        return "employee_low_conf", emp
    vis = _search_table("visitor_embeddings", "visitor_id", embedding)
    if vis is None:
        return "db_error", None
    if vis.get("found") and vis.get("verified"):
        return "frequent_visitor", vis
    if vis.get("found") and vis.get("low_conf"):
        return "visitor_low_conf", vis
    return "unknown", None
class FaceResult:
    PENDING = "pending"
    EMPLOYEE = "employee"
    EMPLOYEE_LOW = "employee_low"
    FREQ_VISITOR = "frequent_visitor"
    VISITOR_LOW = "visitor_low"
    NEW_VISITOR = "new_visitor"
    SPOOF = "spoof"
    DB_ERROR = "db_error"
    def __init__(self, bbox):
        self.bbox = bbox
        self.state = self.PENDING
        self.person_id = None
        self.full_name = None
        self.distance = None
        self.liveness = None
_new_visitor_lock = threading.Lock()
_new_visitor_pending = False   
def prompt_new_visitor():
    """
    Collect new visitor info from terminal and log it.
    Must run in a daemon thread so the camera loop continues.
    """
    global _new_visitor_pending
    print("\n" + "=" * 50)
    print("  NEW VISITOR DETECTED — please fill in details")
    print("=" * 50)
    name = input("Full Name: ").strip()
    phone = input("Phone Number: ").strip()
    purpose = input("Purpose / Company: ").strip()
    if name:
        logger.log_new_visitor(name, phone, purpose, camera_idx = args.camera)
        print(f"  Logged new visitor: {name}\n")
    else:
        print("  Skipped (no name entered).\n")
    with _new_visitor_lock:
        _new_visitor_pending = False
_work_frame = None
_work_faces = None
_work_ready = False
_results = []
_results_lock = threading.Lock()
_work_lock = threading.Lock()
_worker_busy = False
def recognition_worker():
    global _work_frame, _work_faces, _work_ready, _results, _worker_busy, _new_visitor_pending
    while True:
        time.sleep(0.01)
        with _work_lock:
            if not _work_ready:
                continue
            frame = _work_frame.copy()
            faces = _work_faces
            _work_ready = False
            _worker_busy = True
        new_results = []
        for face in faces:
            result = FaceResult(bbox = face.bbox.astype(int).tolist())
            if liveness_model is not None:
                try:
                    score = float(liveness_model.predict(frame, face))
                    result.liveness = score
                    if score < LIVENESS_THRESHOLD:
                        result.state = FaceResult.SPOOF
                        new_results.append(result)
                        continue
                except Exception as e:
                    print(f"[LIVENESS ERROR] {e}")
            embedding = face.embedding.astype(np.float32)
            ptype, res = search_both_tables(embedding)
            if ptype == "employee":
                result.state = FaceResult.EMPLOYEE
                result.person_id = res["person_id"]
                result.full_name = res["full_name"]
                result.distance = res["distance"]
                logger.log_known(
                    result.person_id, "employee",
                    result.full_name, result.distance, args.camera,
                )
            elif ptype == "employee_low_conf":
                result.state = FaceResult.EMPLOYEE_LOW
                result.person_id = res["person_id"]
                result.full_name = res["full_name"]
                result.distance = res["distance"]
            elif ptype == "frequent_visitor":
                result.state = FaceResult.FREQ_VISITOR
                result.person_id = res["person_id"]
                result.full_name = res["full_name"]
                result.distance = res["distance"]
                logger.log_known(
                    result.person_id, "frequent_visitor",
                    result.full_name, result.distance, args.camera,
                )
            elif ptype == "visitor_low_conf":
                result.state = FaceResult.VISITOR_LOW
                result.person_id = res["person_id"]
                result.full_name = res["full_name"]
                result.distance = res["distance"]
            elif ptype == "unknown":
                result.state = FaceResult.NEW_VISITOR
                with _new_visitor_lock:
                    if not _new_visitor_pending:
                        _new_visitor_pending = True
                        threading.Thread(target = prompt_new_visitor, daemon = True).start()
            else:  
                result.state = FaceResult.DB_ERROR
            new_results.append(result)
        with _results_lock:
            _results = new_results
        _worker_busy = False
worker_thread = threading.Thread(target=recognition_worker, daemon = True)
worker_thread.start()
def draw_rounded_rect(img, x1, y1, x2, y2, colour, thickness = 2, r = 12):
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), colour, thickness)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), colour, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), colour, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), colour, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, colour, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, colour, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, colour, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, colour, thickness)
def draw_label(img, text, x, y, colour):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
    overlay = img.copy()
    cv2.rectangle(overlay, (x - 4, y - th - 4), (x + tw + 4, y + 4), DARK, -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    cv2.putText(img, text, (x, y), font, scale, colour, 1, cv2.LINE_AA)
def draw_face_result(frame, result):
    x1, y1, x2, y2 = result.bbox
    line2 = ""
    if result.state == FaceResult.EMPLOYEE:
        colour = GREEN
        name = result.full_name or result.person_id
        line1 = f"EMPLOYEE  {name[:20]}"
        line2 = f"GATE OPEN  |  dist: {result.distance:.3f}"
    elif result.state == FaceResult.EMPLOYEE_LOW:
        colour = ORANGE
        name = result.full_name or result.person_id
        line1 = f"EMPLOYEE (low conf)  {name[:16]}"
        line2 = f"dist: {result.distance:.3f}  — re-enroll?"
    elif result.state == FaceResult.FREQ_VISITOR:
        colour = BLUE
        name = result.full_name or result.person_id
        line1 = f"FREQUENT VISITOR  {name[:16]}"
        line2 = f"dist: {result.distance:.3f}"
    elif result.state == FaceResult.VISITOR_LOW:
        colour = ORANGE
        name = result.full_name or result.person_id
        line1 = f"VISITOR (low conf)  {name[:14]}"
        line2 = f"dist: {result.distance:.3f}  — re-enroll?"
    elif result.state == FaceResult.NEW_VISITOR:
        colour = CYAN
        line1 = "NEW VISITOR"
        line2 = "Please fill details in terminal"
    elif result.state == FaceResult.SPOOF:
        colour = YELLOW
        score = f"{result.liveness:.2f}" if result.liveness is not None else "?"
        line1 = "SPOOF DETECTED  — ACCESS DENIED"
        line2 = f"liveness score: {score}"
    elif result.state == FaceResult.DB_ERROR:
        colour = RED
        line1 = "DB ERROR"
        line2 = "Check database connection"
    else:
        colour = WHITE
        line1 = "Scanning..."
    draw_rounded_rect(frame, x1, y1, x2, y2, colour, thickness = 2)
    draw_label(frame, line1, x1, y1 - 22, colour)
    if line2:
        draw_label(frame, line2, x1, y1 - 6, colour)
def draw_hud(frame, fps, face_count, paused):
    h, _ = frame.shape[:2]
    lines = [
        f"FPS: {fps:.1f}",
        f"Faces: {face_count}",
        datetime.now().strftime("%Y-%m-%d  %H:%M:%S"),
    ]
    if paused:
        lines.insert(0, "[ PAUSED ]")
    if not USE_LIVENESS:
        lines.append("Anti-spoof: OFF")
    legend = [
        ("GREEN = Employee (gate open)", GREEN),
        ("BLUE = Frequent Visitor", BLUE),
        ("CYAN = New Visitor (unknown)", CYAN),
        ("ORANGE = Low Confidence", ORANGE),
        ("YELLOW = Spoof Detected", YELLOW),
    ]
    y = 22
    for line in lines:
        draw_label(frame, line, 10, y, WHITE)
        y += 22
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    lx = frame.shape[1] - 290
    ly = 22
    for text, col in legend:
        (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
        overlay = frame.copy()
        cv2.rectangle(overlay, (lx - 4, ly - th - 4), (lx + tw + 4, ly + 4), DARK, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, text, (lx, ly), font, scale, col, 1, cv2.LINE_AA)
        ly += th + 8
    draw_label(frame, "Q:Quit  S:Screenshot  P:Pause  L:Today log  E:Export CSV",
               10, h - 10, WHITE)
cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    print(f"Cannot open camera {args.camera}")
    sys.exit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
print(f"Camera ready at {int(cap.get(3))}x{int(cap.get(4))}")
os.makedirs("screenshots", exist_ok=True)
frame_count = 0
fps = 0.0
fps_timer = time.time()
fps_counter = 0
paused = False
while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed"); break
    frame_count += 1
    fps_counter += 1
    if fps_counter >= 30:
        elapsed = time.time() - fps_timer
        fps = fps_counter / elapsed if elapsed > 0 else 0
        fps_timer = time.time()
        fps_counter = 0
    if paused:
        with _results_lock:
            for r in _results:
                draw_face_result(frame, r)
        draw_hud(frame, fps, len(_results), paused = True)
        cv2.imshow("Access Control — Face Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        if key in (ord('p'), ord('P')):
            paused = False
        continue
    detected = app.get(frame)
    if frame_count % RECOGNITION_EVERY_N_FRAMES == 0 and detected:
        with _work_lock:
            if not _worker_busy:
                _work_frame = frame.copy()
                _work_faces = detected
                _work_ready = True
    with _results_lock:
        live_results = list(_results)
    if live_results:
        for r in live_results:
            draw_face_result(frame, r)
    elif detected:
        for face in detected:
            x1, y1, x2, y2 = face.bbox.astype(int)
            draw_rounded_rect(frame, x1, y1, x2, y2, WHITE)
            draw_label(frame, "Scanning...", x1, y1 - 10, WHITE)
    draw_hud(frame, fps, len(detected), paused = False)
    cv2.imshow("Access Control — Face Recognition", frame)
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), ord('Q'), 27):
        break
    elif key in (ord('s'), ord('S')):
        path = f"screenshots/capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(path, frame)
        print(f"Screenshot: {path}")
    elif key in (ord('p'), ord('P')):
        paused = True
    elif key in (ord('l'), ord('L')):
        known = logger.today_known()
        new_v = logger.today_new_visitors()
        print(f"\n=== Today's Log ({datetime.now().strftime('%Y-%m-%d')}) ===")
        for pid, ptype, name, seen_at, dist in known:
            tag = "EMP" if ptype == "employee" else "VIS"
            print(f"  {seen_at.strftime('%H:%M:%S')}  [{tag}] {pid:<20} {name or '':<20}  dist={dist:.4f}")
        for _, name, phone, purpose, seen_at in new_v:
            print(f"  {seen_at.strftime('%H:%M:%S')}  [NEW] {name:<20} {phone:<15} {purpose}")
        print()
    elif key in (ord('e'), ord('E')):
        print(f"Exported: {logger.export_csv()}")
cap.release()
cv2.destroyAllWindows()
logger.close()
cursor.close()
conn.close()