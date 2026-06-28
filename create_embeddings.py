import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from dotenv import load_dotenv

load_dotenv()

IMAGE_FOLDER = os.getenv("IMAGE_FOLDER", "faces")
EMBEDDING_FOLDER = os.getenv("EMBEDDING_FOLDER", "embeddings")
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1)
os.makedirs(EMBEDDING_FOLDER, exist_ok = True)
all_files = sorted(f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(".jpg"))
print(f"Found {len(all_files)} images in '{IMAGE_FOLDER}'")
saved = skipped = errors = 0
for file in all_files:
    out_path = os.path.join(EMBEDDING_FOLDER, f"{os.path.splitext(file)[0]}.npy")
    if os.path.exists(out_path):
        skipped += 1
        continue
    try:
        img = cv2.imread(os.path.join(IMAGE_FOLDER, file))
        if img is None:
            print(f"Cannot read: {file}")
            errors += 1
            continue
        faces = app.get(img)
        if len(faces) == 0:
            print(f"No face: {file}")
            errors += 1
            continue
        if len(faces) > 1:
            print(f"{len(faces)} faces found: {file}")
            errors += 1
            continue
        np.save(out_path, faces[0].embedding.astype(np.float32))
        saved += 1
    except Exception as e:
        print(f"{file}: {e}")
        errors += 1
print(f"Done\nSaved: {saved}  Already existed: {skipped}  Skipped: {errors}")