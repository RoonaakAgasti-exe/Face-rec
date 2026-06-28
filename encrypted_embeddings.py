import os
import sys
import cv2
import numpy as np
from cryptography.fernet import Fernet
from insightface.app import FaceAnalysis
from dotenv import load_dotenv

load_dotenv()

KEY_FILE = "keys/secret.key"
ENC_FOLDER = "encrypted_emb"
if len(sys.argv) < 3:
    sys.exit(1)
image_path, output_name = sys.argv[1], sys.argv[2]
os.makedirs("keys", exist_ok = True)
os.makedirs(ENC_FOLDER, exist_ok = True)
if not os.path.exists(KEY_FILE):
    key = Fernet.generate_key()
    with open(KEY_FILE, "wb") as f:
        f.write(key)
    print(f"New key generated: {KEY_FILE}")
else:
    with open(KEY_FILE, "rb") as f:
        key = f.read()
    print(f"Loaded key: {KEY_FILE}")
fernet = Fernet(key)
app = FaceAnalysis(name = "buffalo_l")
app.prepare(ctx_id = -1)
img = cv2.imread(image_path)
if img is None:
    print(f"Cannot read image: {image_path}")
    sys.exit(1)
faces = app.get(img)
if len(faces) == 0:
    print("No face detected")
    sys.exit(1)
if len(faces) > 1:
    print(f"{len(faces)} faces detected")
    sys.exit(1)
embedding = faces[0].embedding.astype(np.float32)
print(f"Embedding shape: {embedding.shape}")
data = embedding.tobytes()
encrypted = fernet.encrypt(data)
out_path = os.path.join(ENC_FOLDER, f"{output_name}.enc")
with open(out_path, "wb") as f:
    f.write(encrypted)
print(f"Saved: {out_path}  ({len(data)}b → {len(encrypted)}b)")
recovered = np.frombuffer(fernet.decrypt(open(out_path, "rb").read()), dtype = np.float32)
print("Verification passed" if np.allclose(embedding, recovered) else "Verification failed")