import os
import sys
import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "visit_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT", "55432")
EMBEDDING_FOLDER = os.getenv("EMBEDDING_FOLDER", "embeddings")
EXPECTED_DIM = 512
print(f"Connecting to PostgreSQL at {DB_HOST}:{DB_PORT}/{DB_NAME}")
try:
    conn = psycopg2.connect(
        host = DB_HOST, 
        database = DB_NAME,
        user = DB_USER, 
        password = DB_PASSWORD, 
        port = DB_PORT
    )
except psycopg2.OperationalError as e:
    print(f"Cannot connect: {e}\nRun: docker compose up -d")
    sys.exit(1)
register_vector(conn)
cursor = conn.cursor()
print("Connected")
if not os.path.exists(EMBEDDING_FOLDER):
    os.makedirs(EMBEDDING_FOLDER, exist_ok=True)
files = sorted(f for f in os.listdir(EMBEDDING_FOLDER) if f.endswith(".npy"))
total = len(files)
print(f"Found {total} .npy files to insert")
inserted = skipped = errors = 0
for i, file in enumerate(files, 1):
    try:
        embedding = np.load(os.path.join(EMBEDDING_FOLDER, file)).astype(np.float32)
        if embedding.shape[0] != EXPECTED_DIM:
            print(f"{file}: wrong dimension {embedding.shape[0]}")
            skipped += 1
            continue
        cursor.execute(
            """
            INSERT INTO employee_embeddings (emp_id, embedding)
            VALUES (%s, %s)
            ON CONFLICT (emp_id) DO NOTHING;
            """, (os.path.splitext(file)[0], embedding)
        )
        inserted += 1
    except Exception as e:
        print(f"{file}: {e}")
        conn.rollback()
        errors += 1
        continue
    if i % 100 == 0:
        conn.commit()
        print(f"{i}/{total}  inserted: {inserted}  skipped: {skipped}  errors: {errors}")
conn.commit()
cursor.close()
conn.close()
print(f"{'='*50}\nMigration complete\nInserted: {inserted}  Skipped: {skipped}  Errors: {errors}")