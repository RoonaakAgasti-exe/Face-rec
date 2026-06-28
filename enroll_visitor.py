import os
import sys
import cv2
import time
import numpy as np
import psycopg2
import threading
import argparse
import uuid
from pgvector.psycopg2 import register_vector
from insightface.app import FaceAnalysis
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "visit_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT", "55432")
LIVENESS_THRESHOLD = float(os.getenv("LIVENESS_THRESHOLD", "0.5"))
EMBEDDING_FOLDER = os.getenv("EMBEDDING_FOLDER", "embeddings/visitors")
parser = argparse.ArgumentParser(description = "Enroll a frequent visitor via live camera")
parser.add_argument("--camera", type = int, default = 0)
args = parser.parse_args()
GREEN = (0, 210, 0)
RED = (0, 40, 220)
YELLOW = (0, 200, 220)
WHITE = (220, 220, 220)
CYAN = (220, 210, 0)
BLUE = (220, 130, 0)
DARK = (30, 30, 30)
STATE_PREVIEW = "preview"
STATE_CAPTURED = "captured"
STATE_SUCCESS = "success"
STATE_FAILED = "failed"
state = STATE_PREVIEW
status_msg = ""
status_colour = WHITE
captured_frame = None
result_frame = None
visitor_id = ""
full_name = ""
phone = ""
purpose = ""
print("Connecting to database...")
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
    print("Database connected.")
except psycopg2.OperationalError as e:
    print(f"Cannot connect: {e}")
    sys.exit(1)
app = FaceAnalysis(name = "buffalo_l")
app.prepare(ctx_id = -1)
liveness_model = None
try:
    from insightface.model_zoo import get_model as _get_model
    liveness_model = _get_model("antispoof")
    liveness_model.prepare(ctx_id=-1)
except Exception as e:
    print(f"Anti-spoof unavailable: {e}")
os.makedirs(EMBEDDING_FOLDER, exist_ok = True)
os.makedirs("enrolled_photos/visitors", exist_ok = True)
def enroll_face(frame, vid, name, ph, purp):
    global state, status_msg, status_colour, result_frame
    faces = app.get(frame)
    if len(faces) == 0:
        status_msg, status_colour, state = "No face detected", RED, STATE_FAILED
        return
    if len(faces) > 1:
        status_msg, status_colour, state = f"{len(faces)} faces — only 1 allowed", RED, STATE_FAILED
        return
    face = faces[0]
    if liveness_model is not None:
        try:
            score = float(liveness_model.predict(frame, face))
            if score < LIVENESS_THRESHOLD:
                status_msg  = f"Spoof detected (score {score:.2f})"
                status_colour = YELLOW
                state = STATE_FAILED
                return
        except Exception as e:
            print(f"[LIVENESS ERROR] {e}")
    try:
        cursor.execute("SELECT visitor_id FROM visitor_embeddings WHERE visitor_id = %s", (vid,))
        if cursor.fetchone():
            status_msg  = f"ID '{vid}' already enrolled as visitor"
            status_colour = YELLOW
            state = STATE_FAILED
            return
    except Exception as e:
        conn.rollback()
        status_msg, status_colour, state = f"DB check failed: {e}", RED, STATE_FAILED
        return
    embedding = face.embedding.astype(np.float32)
    np.save(os.path.join(EMBEDDING_FOLDER, f"{vid}.npy"), embedding)
    try:
        cursor.execute(
            """
            INSERT INTO visitor_embeddings (visitor_id, full_name, phone, purpose, embedding)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (visitor_id) DO UPDATE
              SET full_name = EXCLUDED.full_name,
                  phone     = EXCLUDED.phone,
                  purpose   = EXCLUDED.purpose,
                  embedding = EXCLUDED.embedding;
            """, (vid, name, ph or None, purp or None, embedding)
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        status_msg, status_colour, state = f"DB insert failed: {e}", RED, STATE_FAILED
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    photo_path = f"enrolled_photos/visitors/{vid}_{ts}.jpg"
    cv2.imwrite(photo_path, frame)
    x1, y1, x2, y2 = face.bbox.astype(int)
    cv2.rectangle(result_frame, (x1, y1), (x2, y2), GREEN, 2)
    print(f"  Enrolled visitor: {vid}  ({name})")
    print(f"  Phone: {ph}")
    print(f"  Purpose: {purp}")
    print(f"  Photo: {photo_path}")
    status_msg = f"'{name}' enrolled as FREQUENT VISITOR"
    status_colour = GREEN
    state = STATE_SUCCESS
def draw_text_box(frame, lines, y_start, colour = WHITE, scale = 0.6):
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = y_start
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, font, scale, 1)
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, y - th - 4), (8 + tw + 8, y + 4), DARK, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, line, (12, y), font, scale, colour, 1, cv2.LINE_AA)
        y += th + 10
    return y
def draw_crosshair(frame):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    c = (180, 180, 180)
    cv2.line(frame, (cx - 30, cy), (cx + 30, cy), c, 1)
    cv2.line(frame, (cx, cy - 30), (cx, cy + 30), c, 1)
    cv2.ellipse(frame, (cx, cy), (110, 140), 0, 0, 360, c, 1)
cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    print(f"Cannot open camera {args.camera}")
    sys.exit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("\n" + "=" * 50)
print("  FREQUENT VISITOR ENROLLMENT")
print("=" * 50)
full_name = input("Full Name: ").strip()
phone = input("Phone Number: ").strip()
purpose = input("Purpose / Company: ").strip()
visitor_id = input("Visitor ID (or Enter for auto-generate): ").strip()
if not visitor_id:
    visitor_id = f"VIS-{uuid.uuid4().hex[:8].upper()}"
    print(f"  Auto-generated ID: {visitor_id}")
if not full_name:
    print("No name entered. Exiting.")
    cap.release(); cursor.close(); conn.close()
    sys.exit(0)
state = STATE_PREVIEW
print(f"\nEnrolling visitor: {visitor_id}  ({full_name})")
print("SPACE = capture   R = retake   Q = quit\n")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed"); break
    display = frame.copy()
    if state == STATE_PREVIEW:
        faces = app.get(frame)
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            cv2.rectangle(display, (x1, y1), (x2, y2), BLUE, 2)
        draw_crosshair(display)
        fc = len(faces)
        guide_col  = GREEN  if fc == 1 else (RED if fc == 0 else YELLOW)
        guide_text = ("Face detected — SPACE to capture" if fc == 1
                      else ("No face detected" if fc == 0 else f"{fc} faces detected"))
        draw_text_box(display,
                      [f"[VISITOR] Enrolling: {full_name}  ({visitor_id})",
                       guide_text,
                       "SPACE = capture   Q = quit"],
                      y_start=30, colour=guide_col)
    elif state == STATE_CAPTURED:
        display = result_frame.copy()
        draw_text_box(display, [f"Enrolling: {full_name}", "Processing..."], 30, CYAN)
    elif state == STATE_SUCCESS:
        display = result_frame.copy()
        draw_text_box(display, [status_msg, "SPACE = enroll another   Q = quit"], 30, GREEN, 0.65)
    elif state == STATE_FAILED:
        display = result_frame.copy() if result_frame is not None else display
        draw_text_box(display, [status_msg, "R = retake   Q = quit"], 30, RED)
    cv2.imshow("Visitor Enrollment", display)
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), ord('Q'), 27):
        print("Exiting."); break
    elif key == ord(' '):
        if state == STATE_PREVIEW:
            captured_frame = frame.copy()
            result_frame = frame.copy()
            state = STATE_CAPTURED
            threading.Thread(
                target = enroll_face,
                args = (captured_frame, visitor_id, full_name, phone, purpose),
                daemon = True,
            ).start()
        elif state == STATE_SUCCESS:
            print()
            full_name = input("Full Name: ").strip()
            phone = input("Phone Number: ").strip()
            purpose = input("Purpose / Company: ").strip()
            visitor_id = input("Visitor ID (or Enter for auto-generate): ").strip()
            if not visitor_id:
                visitor_id = f"VIS-{uuid.uuid4().hex[:8].upper()}"
                print(f"  Auto-generated ID: {visitor_id}")
            if not full_name:
                print("No name. Exiting."); break
            result_frame = None; status_msg = ""; state = STATE_PREVIEW
    elif key in (ord('r'), ord('R')):
        if state in (STATE_FAILED, STATE_SUCCESS, STATE_CAPTURED):
            result_frame = None; status_msg = ""; state = STATE_PREVIEW
cap.release()
cv2.destroyAllWindows()
cursor.close()
conn.close()