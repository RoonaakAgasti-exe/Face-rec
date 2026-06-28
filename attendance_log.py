import os
import sys
import csv
import argparse
import psycopg2
from dotenv import load_dotenv
from datetime import datetime, date

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "visit_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT", "55432")
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS attendance_log (
    id         SERIAL PRIMARY KEY,
    emp_id     TEXT        NOT NULL,
    seen_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    distance   FLOAT,
    camera_idx INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_attendance_seen_at ON attendance_log (seen_at DESC);
CREATE INDEX IF NOT EXISTS idx_attendance_emp_id  ON attendance_log (emp_id);
"""
class AttendanceLogger:
    COOLDOWN_SECONDS = 60
    def __init__(self):
        self.conn = None
        self.cursor = None
        self._last_seen = {}
        self._connect()
        self._ensure_table()
    def _connect(self):
        self.conn = psycopg2.connect(
            host = DB_HOST, 
            database = DB_NAME,
            user = DB_USER, 
            password = DB_PASSWORD, 
            port = DB_PORT
        )
        self.cursor = self.conn.cursor()
    def _ensure_table(self):
        self.cursor.execute(CREATE_TABLE_SQL)
        self.conn.commit()
    def log(self, emp_id, distance, camera_idx = 0):
        now = datetime.now()
        if emp_id in self._last_seen:
            if (now - self._last_seen[emp_id]).total_seconds() < self.COOLDOWN_SECONDS:
                return False
        try:
            self.cursor.execute(
                """
                INSERT INTO attendance_log (emp_id, seen_at, distance, camera_idx)
                VALUES (%s, %s, %s, %s)
                """, (emp_id, now, float(distance), camera_idx)
            )
            self.conn.commit()
            self._last_seen[emp_id] = now
            print(f"{now.strftime('%H:%M:%S')} {emp_id} dist={distance:.4f}")
            return True
        except Exception as e:
            self.conn.rollback()
            print(f"[LOG ERROR] {e}")
            return False
    def today(self):
        self.cursor.execute(
            """
            SELECT emp_id, seen_at, distance
            FROM attendance_log
            WHERE seen_at::date = CURRENT_DATE
            ORDER BY seen_at ASC
            """
        )
        return self.cursor.fetchall()
    def all_records(self):
        self.cursor.execute(
            """
            SELECT emp_id, seen_at, distance, camera_idx
            FROM attendance_log
            ORDER BY seen_at DESC
            """
        )
        return self.cursor.fetchall()
    def export_csv(self, path = None):
        if path is None:
            path = f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        rows = self.all_records()
        with open(path, "w", newline = "") as f:
            writer = csv.writer(f)
            writer.writerow(["emp_id", "seen_at", "distance", "camera_idx"])
            writer.writerows(rows)
        print(f"Exported {len(rows)} records to: {path}")
        return path
    def clear(self):
        self.cursor.execute("Delete from attendance_log")
        self.conn.commit()
        self._last_seen.clear()
        print("All attendance records deleted")
    def close(self):
        self.cursor.close()
        self.conn.close()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Attendance log viewer / exporter")
    parser.add_argument("--export", action = "store_true")
    parser.add_argument("--clear",  action = "store_true")
    args = parser.parse_args()
    logger = AttendanceLogger()
    if args.clear:
        if input("Delete ALL records? Type YES to confirm: ").strip() == "YES":
            logger.clear()
        else:
            print("Aborted")
    elif args.export:
        logger.export_csv()
    else:
        records = logger.today()
        print(f"Attendance — {date.today().strftime('%Y-%m-%d')}")
        print("-" * 50)
        if not records:
            print("No records today")
        else:
            for emp_id, seen_at, distance in records:
                print(f"  {seen_at.strftime('%H:%M:%S')}  {emp_id:<20}  dist={distance:.4f}")
            print(f"Total: {len(records)} entries")
    logger.close()