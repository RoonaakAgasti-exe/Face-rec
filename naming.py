import os
import csv
import random
import string
from dotenv import load_dotenv

load_dotenv()

FOLDER = os.getenv("EMBEDDING_FOLDER", "embeddings")
RANDOM_NAME_LENGTH = 12
MAPPING_FILE = "mapping.csv"
def random_name(length=12):
    return "".join(random.choices(string.ascii_letters + string.digits, k = length))
npy_files = [f for f in os.listdir(FOLDER) if f.endswith(".npy")]
if not npy_files:
    print(f"No .npy files found in '{FOLDER}'")
    raise SystemExit
if os.path.exists(MAPPING_FILE):
    print(f"Warning: {MAPPING_FILE} already exists. Press Ctrl+C to cancel.")
    try:
        input("Press Enter to continue...")
    except KeyboardInterrupt:
        print("\nAborted.")
        raise SystemExit
used_names = set()
mapping    = []
for filename in npy_files:
    while True:
        new_name = random_name(RANDOM_NAME_LENGTH) + ".npy"
        if new_name not in used_names and not os.path.exists(os.path.join(FOLDER, new_name)):
            used_names.add(new_name)
            break
    os.rename(os.path.join(FOLDER, filename), os.path.join(FOLDER, new_name))
    mapping.append([filename, new_name])
    print(f"{filename} → {new_name}")
with open(MAPPING_FILE, "w", newline = "") as f:
    writer = csv.writer(f)
    writer.writerow(["Original Filename", "Random Filename"])
    writer.writerows(mapping)
print(f"\nDone. {len(mapping)} files renamed. Mapping saved to: {MAPPING_FILE}")