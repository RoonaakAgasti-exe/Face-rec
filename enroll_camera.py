import os
import sys
import cv2
import time
import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from insightface.app import FaceAnalysis
from dotenv import load_dotenv
from datetime import datetime
import argparse
import threading

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "visit_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT", "55432")
LIVENESS_THRESHOLD = float(os.getenv("LIVENESS_THRESHOLD", "0.5"))
EMBEDDING_FOLDER = os.getenv("EMBEDDING_FOLDER", "embeddings")
parser = argparse.ArgumentParser(description = "Enroll a new person via live camera")
parser.add_argument("--camera", type = int, default = 0)
args = parser.parse_args()
GREEN = (0, 210, 0)
RED = (0, 40, 220)
YELLOW = (0, 200, 220)
WHITE = (220, 220, 220)
CYAN = (220, 210, 0)
DARK = (30, 30, 30)
STATE_WAITING_FOR_ID = "waiting_for_id"
STATE_PREVIEW = "preview"
STATE_CAPTURED = "captured"
STATE_SUCCESS = "success"
STATE_FAILED = "failed"
state = STATE_WAITING_FOR_ID
emp_id = ""
status_msg = ""
status_colour = WHITE
captured_frame = None
result_frame = None
print("Connecting to database....")
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
    print("Database connected")
except psycopg2.OperationalError as e:
    print(f"Cannot connect to database: {e}")
    sys.exit(1)
app = FaceAnalysis(name = "buffalo_l")
app.prepare(ctx_id = -1)
liveness_model = None
try:
    from insightface.model_zoo import get_model as _get_model
    liveness_model = _get_model("antispoof")
    liveness_model.prepare(ctx_id = -1)
except Exception as e:
    print(f"Anti-spoof unavailable: {e}")
os.makedirs(EMBEDDING_FOLDER, exist_ok = True)
def enroll_face(frame, emp_id_value):
    global state, status_msg, status_colour, result_frame
    faces = app.get(frame)
    if len(faces) == 0:
        status_msg, status_colour, state = "No face detected", RED, STATE_FAILED
        return
    if len(faces) > 1:
        status_msg, status_colour, state = f"{len(faces)} faces detected", RED, STATE_FAILED
        return
    face = faces[0]
    if liveness_model is not None:
        try:
            score = float(liveness_model.predict(frame, face))
            if score < LIVENESS_THRESHOLD:
                status_msg = f"Spoof detected (score {score:.2f})"
                status_colour = YELLOW
                state = STATE_FAILED
                return
        except Exception as e:
            print(f"[LIVENESS ERROR] {e}")
    try:
        cursor.execute("SELECT emp_id FROM employee_embeddings WHERE emp_id = %s", (emp_id_value,))
        if cursor.fetchone():
            status_msg = f"ID '{emp_id_value}' already enrolled"
            status_colour = YELLOW
            state = STATE_FAILED
            return
    except Exception as e:
        conn.rollback()
        status_msg, status_colour, state = f"DB check failed: {e}", RED, STATE_FAILED
        return
    embedding = face.embedding.astype(np.float32)
    np.save(os.path.join(EMBEDDING_FOLDER, f"{emp_id_value}.npy"), embedding)
    try:
        cursor.execute(
            """
            INSERT INTO employee_embeddings (emp_id, embedding)
            VALUES (%s, %s)
            ON CONFLICT (emp_id) DO UPDATE SET embedding = EXCLUDED.embedding;
            """, (emp_id_value, embedding)
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        status_msg, status_colour, state = f"Database insert failed: {e}", RED, STATE_FAILED
        return
    photo_dir = "enrolled_photos"
    os.makedirs(photo_dir, exist_ok = True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    photo_path = os.path.join(photo_dir, f"{emp_id_value}_{ts}.jpg")
    cv2.imwrite(photo_path, frame)
    x1, y1, x2, y2 = face.bbox.astype(int)
    cv2.rectangle(result_frame, (x1, y1), (x2, y2), GREEN, 2)
    print(f"Enrolled : {emp_id_value}")
    print(f"Embedding : {EMBEDDING_FOLDER}/{emp_id_value}.npy")
    print(f"Photo : {photo_path}")
    status_msg = f"'{emp_id_value}' enrolled successfully"
    status_colour = GREEN
    state = STATE_SUCCESS
def draw_text_box(frame, lines, y_start, colour = WHITE, scale = 0.6):
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = y_start
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, font, scale, 1)
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, y-th-4), (8+tw+8, y+4), DARK, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, line, (12, y), font, scale, colour, 1, cv2.LINE_AA)
        y += th + 10
    return y
def draw_face_boxes(frame, faces):
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), CYAN, 2)
def draw_crosshair(frame):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    colour = (180, 180, 180)
    cv2.line(frame, (cx-30, cy), (cx+30, cy), colour, 1)
    cv2.line(frame, (cx, cy-30), (cx, cy+30), colour, 1)
    cv2.ellipse(frame, (cx, cy), (110, 140), 0, 0, 360, colour, 1)
cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    print(f"Cannot open camera {args.camera}")
    sys.exit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("\n" + "="*50)
print("  FACE ENROLLMENT")
print("="*50)
emp_id = input("Employee ID: ").strip()
if not emp_id:
    print("No ID entered. Exiting.")
    cap.release()
    cursor.close()
    conn.close()
    sys.exit(0)
state = STATE_PREVIEW
print(f"\nEnrolling: {emp_id}")
print("SPACE = capture   R = retake   Q = quit\n")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed")
        break
    display = frame.copy()
    h, w = display.shape[:2]
    if state == STATE_PREVIEW:
        faces = app.get(frame)
        draw_face_boxes(display, faces)
        draw_crosshair(display)
        face_count = len(faces)
        if face_count == 0:
            guide_colour, guide_text = RED, "No face detected: move closer"
        elif face_count == 1:
            guide_colour, guide_text = GREEN, "Face detected: SPACE to capture"
        else:
            guide_colour, guide_text = YELLOW, f"{face_count} faces detected"
        draw_text_box(display, [f"Enrolling: {emp_id}", guide_text, "SPACE = capture   R = new ID   Q = quit"], y_start = 30, colour = guide_colour)
    elif state == STATE_CAPTURED:
        display = result_frame.copy()
        draw_text_box(display, [f"Enrolling: {emp_id}", "Processing..."], y_start = 30, colour = CYAN)
    elif state == STATE_SUCCESS:
        display = result_frame.copy()
        draw_text_box(display, [status_msg, "SPACE = enroll another   Q = quit"], y_start = 30, colour = GREEN, scale = 0.65)
    elif state == STATE_FAILED:
        display = result_frame.copy() if result_frame is not None else display
        draw_text_box(display, [status_msg, "R = retake   Q = quit"], y_start = 30, colour = RED, scale = 0.6)
    cv2.imshow("Face Enrollment", display)
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), ord('Q'), 27):
        print("Exiting enrollment.")
        break
    elif key == ord(' '):
        if state == STATE_PREVIEW:
            captured_frame = frame.copy()
            result_frame = frame.copy()
            state = STATE_CAPTURED
            status_msg = ""
            threading.Thread(target = enroll_face, args = (captured_frame, emp_id), daemon = True).start()
        elif state == STATE_SUCCESS:
            emp_id = input("\nEmployee ID for next person: ").strip()
            if not emp_id:
                print("No ID entered. Exiting.")
                break
            print(f"Enrolling: {emp_id}\nSPACE to capture\n")
            result_frame = None
            status_msg = ""
            state = STATE_PREVIEW
    elif key in (ord('r'), ord('R')):
        if state in (STATE_FAILED, STATE_SUCCESS, STATE_CAPTURED):
            result_frame = None
            status_msg = ""
            state = STATE_PREVIEW
            print(f"Retaking for: {emp_id}")
cap.release()
cv2.destroyAllWindows()
cursor.close()
conn.close()