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
from attendance_log import AttendanceLogger

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
RED = (0, 40, 220)
YELLOW = (0, 200, 220)
CYAN = (220, 200, 0)
ORANGE = (0, 140, 255)
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
logger = AttendanceLogger()
app = FaceAnalysis(name = "buffalo_l")
app.prepare(ctx_id = -1)
liveness_model = None
if USE_LIVENESS:
    try:
        from insightface.model_zoo import get_model as _get_model
        liveness_model = _get_model("antispoof")
        liveness_model.prepare(ctx_id = -1)
    except Exception as e:
        print(f"Anti-spoof unavailable: {e}")
def search_database(embedding):
    try:
        cursor.execute("SELECT COUNT(*) FROM employee_embeddings;")
        count = cursor.fetchone()[0]
        if count == 0:
            return {"found": False, "emp_id": None, "distance": None}
        cursor.execute(
            """
            SELECT emp_id, embedding <=> %s AS distance
            FROM employee_embeddings
            ORDER BY embedding <=> %s
            LIMIT 1;
            """,
            (embedding, embedding),
        )
        row = cursor.fetchone()
        if row:
            dist = float(row[1])
            return {
                "found": True,
                "emp_id": row[0],
                "distance": dist,
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
class FaceResult:
    PENDING = "pending"
    VERIFIED = "verified"
    LOW_CONF = "low_conf"
    NEW_EMPLOYEE = "new_employee"
    SPOOF = "spoof"
    def __init__(self, bbox):
        self.bbox = bbox
        self.state = self.PENDING
        self.emp_id = None
        self.distance = None
        self.liveness = None
_work_frame = None
_work_faces = None
_work_ready = False
_results = []
_results_lock = threading.Lock()
_work_lock = threading.Lock()
_worker_busy = False
def recognition_worker():
    global _work_frame, _work_faces, _work_ready, _results, _worker_busy
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
            db_result = search_database(embedding)
            if db_result is None:             
                result.state = FaceResult.NEW_EMPLOYEE
            elif not db_result["found"]:
                result.state = FaceResult.NEW_EMPLOYEE
            else:
                result.emp_id = db_result["emp_id"]
                result.distance = db_result["distance"]
                if db_result["verified"]:
                    result.state = FaceResult.VERIFIED
                    logger.log(emp_id = result.emp_id, distance = result.distance,
                               camera_idx = args.camera)
                elif db_result["low_conf"]:
                    result.state = FaceResult.LOW_CONF
                else:
                    result.state = FaceResult.NEW_EMPLOYEE
            new_results.append(result)
        with _results_lock:
            _results = new_results
        _worker_busy = False
worker_thread = threading.Thread(target = recognition_worker, daemon = True)
worker_thread.start()
def draw_rounded_rect(img, x1, y1, x2, y2, colour, thickness = 2, r = 12):
    cv2.line(img,  (x1+r, y1), (x2-r, y1), colour, thickness)
    cv2.line(img,  (x1+r, y2), (x2-r, y2), colour, thickness)
    cv2.line(img,  (x1, y1+r), (x1, y2-r), colour, thickness)
    cv2.line(img,  (x2, y1+r), (x2, y2-r), colour, thickness)
    cv2.ellipse(img, (x1+r, y1+r), (r, r), 180, 0, 90, colour, thickness)
    cv2.ellipse(img, (x2-r, y1+r), (r, r), 270, 0, 90, colour, thickness)
    cv2.ellipse(img, (x1+r, y2-r), (r, r),  90, 0, 90, colour, thickness)
    cv2.ellipse(img, (x2-r, y2-r), (r, r),   0, 0, 90, colour, thickness)
def draw_label(img, text, x, y, colour):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
    overlay = img.copy()
    cv2.rectangle(overlay, (x-4, y-th-4), (x+tw+4, y+4), DARK, -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    cv2.putText(img, text, (x, y), font, scale, colour, 1, cv2.LINE_AA)
def draw_face_result(frame, result):
    x1, y1, x2, y2 = result.bbox
    if result.state == FaceResult.VERIFIED:
        colour = GREEN
        emp = result.emp_id[:14] if result.emp_id else "?"
        line1 = f"EXISTING EMPLOYEE  {emp}"
        line2 = f"dist: {result.distance:.3f}"
    elif result.state == FaceResult.LOW_CONF:
        colour = ORANGE
        emp = result.emp_id[:14] if result.emp_id else "?"
        line1 = f"LOW CONFIDENCE  {emp}"
        line2 = f"dist: {result.distance:.3f}  (enroll again?)"
    elif result.state == FaceResult.NEW_EMPLOYEE:
        colour = CYAN
        line1 = "NEW EMPLOYEE"
        line2 = "Not enrolled — use enroll_camera.py"

    elif result.state == FaceResult.SPOOF:
        colour = YELLOW
        score = f"{result.liveness:.2f}" if result.liveness is not None else "?"
        line1 = "SPOOF DETECTED"
        line2 = f"liveness: {score}"
    else:
        colour = WHITE
        line1 = "Scanning..."
        line2 = ""
    draw_rounded_rect(frame, x1, y1, x2, y2, colour, thickness = 2)
    draw_label(frame, line1, x1, y1 - 22, colour)
    if line2:
        draw_label(frame, line2, x1, y1 - 6, colour)
def draw_hud(frame, fps, face_count, log_count, paused):
    h, _ = frame.shape[:2]
    lines = [
        f"FPS: {fps:.1f}",
        f"Faces: {face_count}",
        datetime.now().strftime("%Y-%m-%d  %H:%M:%S"),
        f"Logged today: {log_count}"
    ]
    if paused:
        lines.insert(0, "[ PAUSED ]")
    if not USE_LIVENESS:
        lines.append("Anti-spoof: OFF")
    legend = [
        ("GREEN = Existing Employee", GREEN),
        ("CYAN = New Employee", CYAN),
        ("ORANGE = Low Confidence", ORANGE),
        ("YELLOW = Spoof Detected", YELLOW),
    ]
    y = 22
    for line in lines:
        draw_label(frame, line, 10, y, WHITE)
        y += 22
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    lx = frame.shape[1] - 260
    ly = 22
    for text, col in legend:
        (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
        overlay = frame.copy()
        cv2.rectangle(overlay, (lx-4, ly-th-4), (lx+tw+4, ly+4), DARK, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, text, (lx, ly), font, scale, col, 1, cv2.LINE_AA)
        ly += th + 8
    draw_label(frame, "Q: Quit  S: Screenshot  P: Pause  L: Today log  E: Export CSV",
               10, h - 10, WHITE)
cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    print(f"Cannot open camera {args.camera}")
    sys.exit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
print(f"Camera ready at {int(cap.get(3))}x{int(cap.get(4))}")
os.makedirs("screenshots", exist_ok=True)
frame_count = 0
fps = 0.0
fps_timer = time.time()
fps_counter = 0
paused = False
today_log_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed")
        break
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
        draw_hud(frame, fps, len(_results), today_log_count, paused = True)
        cv2.imshow("Face Recognition — Employee Detection", frame)
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
    draw_hud(frame, fps, len(detected), today_log_count, paused = False)
    cv2.imshow("Face Recognition — Employee Detection", frame)
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
        records = logger.today()
        today_log_count = len(records)
        print(f"\n--- Today's Attendance ({today_log_count} entries) ---")
        for emp_id, seen_at, distance in records:
            print(f"  {seen_at.strftime('%H:%M:%S')}  {emp_id:<20}  dist = {distance:.4f}")
        print("---\n")
    elif key in (ord('e'), ord('E')):
        print(f"Exported: {logger.export_csv()}")
cap.release()
cv2.destroyAllWindows()
logger.close()
cursor.close()
conn.close()