import os
import csv
import psycopg2
from dotenv import load_dotenv
from datetime import datetime, date
import argparse

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "visit_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT", "55432")
class VisitLogger:
    """
    Dual-table logger.

    attendance_log    → employees and frequent visitors (recognised faces)
    new_visitor_log   → unknown walk-ins who typed their info
    """
    COOLDOWN_SECONDS = 60          
    def __init__(self):
        self.conn = None
        self.cursor = None
        self._last_seen: dict[str, datetime] = {}
        self._connect()
        self._ensure_tables()
    def _connect(self):
        self.conn = psycopg2.connect(
            host = DB_HOST,
            database = DB_NAME,
            user = DB_USER,
            password = DB_PASSWORD,
            port = DB_PORT,
        )
        self.cursor = self.conn.cursor()
    def _ensure_tables(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance_log (
                id          SERIAL PRIMARY KEY,
                person_id   TEXT        NOT NULL,
                person_type TEXT        NOT NULL CHECK (person_type IN ('employee','frequent_visitor')),
                full_name   TEXT,
                seen_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                distance    FLOAT,
                camera_idx  INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_att_seen_at   ON attendance_log (seen_at DESC);
            CREATE INDEX IF NOT EXISTS idx_att_person_id ON attendance_log (person_id);

            CREATE TABLE IF NOT EXISTS new_visitor_log (
                id          SERIAL PRIMARY KEY,
                full_name   TEXT        NOT NULL,
                phone       TEXT,
                purpose     TEXT,
                seen_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                camera_idx  INTEGER DEFAULT 0,
                promoted    BOOLEAN DEFAULT FALSE
            );
            CREATE INDEX IF NOT EXISTS idx_nv_seen_at ON new_visitor_log (seen_at DESC);
        """)
        self.conn.commit()
    def log_known(self, person_id: str, person_type: str,
                  full_name: str | None, distance: float,
                  camera_idx: int = 0) -> bool:
        """
        Log an employee or frequent visitor.
        person_type must be 'employee' or 'frequent_visitor'.
        Returns True if a new row was inserted, False if within cooldown.
        """
        now = datetime.now()
        cache_key = f"{person_type}:{person_id}"
        if cache_key in self._last_seen:
            if (now - self._last_seen[cache_key]).total_seconds() < self.COOLDOWN_SECONDS:
                return False
        try:
            self.cursor.execute(
                """
                INSERT INTO attendance_log (person_id, person_type, full_name, seen_at, distance, camera_idx)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (person_id, person_type, full_name, now, float(distance), camera_idx),
            )
            self.conn.commit()
            self._last_seen[cache_key] = now
            tag = "EMP" if person_type == "employee" else "VIS"
            print(f"[{now.strftime('%H:%M:%S')}] [{tag}] {person_id} ({full_name})  dist={distance:.4f}")
            return True
        except Exception as e:
            self.conn.rollback()
            print(f"[LOG ERROR] {e}")
            return False
    def log_new_visitor(self, full_name: str, phone: str,
                        purpose: str, camera_idx: int = 0) -> int:
        """
        Insert a new (unknown) visitor who typed their details.
        Returns the new row id.
        """
        now = datetime.now()
        try:
            self.cursor.execute(
                """
                INSERT INTO new_visitor_log (full_name, phone, purpose, seen_at, camera_idx)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (full_name, phone or "", purpose or "", now, camera_idx),
            )
            row_id = self.cursor.fetchone()[0]
            self.conn.commit()
            print(f"[{now.strftime('%H:%M:%S')}] [NEW] {full_name} | {phone} | {purpose}")
            return row_id
        except Exception as e:
            self.conn.rollback()
            print(f"[LOG ERROR] {e}")
            return -1
    def today_known(self):
        self.cursor.execute(
            """
            SELECT person_id, person_type, full_name, seen_at, distance
            FROM   attendance_log
            WHERE  seen_at::date = CURRENT_DATE
            ORDER  BY seen_at ASC
            """
        )
        return self.cursor.fetchall()
    def today_new_visitors(self):
        self.cursor.execute(
            """
            SELECT id, full_name, phone, purpose, seen_at
            FROM   new_visitor_log
            WHERE  seen_at::date = CURRENT_DATE
            ORDER  BY seen_at ASC
            """
        )
        return self.cursor.fetchall()
    def export_csv(self, path: str | None = None) -> str:
        if path is None:
            path = f"visit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.cursor.execute(
            """
            SELECT person_id, person_type, full_name, seen_at, distance, camera_idx
            FROM   attendance_log
            ORDER  BY seen_at DESC
            """
        )
        known = self.cursor.fetchall()
        self.cursor.execute(
            """
            SELECT 'NEW_VISITOR', 'new_visitor', full_name, seen_at, NULL, camera_idx
            FROM   new_visitor_log
            ORDER  BY seen_at DESC
            """
        )
        new_vis = self.cursor.fetchall()
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["person_id", "person_type", "full_name", "seen_at", "distance", "camera_idx"])
            writer.writerows(known)
            writer.writerows(new_vis)
        print(f"Exported {len(known) + len(new_vis)} records → {path}")
        return path
    def close(self):
        self.cursor.close()
        self.conn.close()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Visit log viewer / exporter")
    parser.add_argument("--export", action = "store_true", help = "Export full CSV")
    parser.add_argument("--today",  action = "store_true", help = "Print today's log")
    args = parser.parse_args()
    logger = VisitLogger()
    if args.export:
        logger.export_csv()
    else:
        known = logger.today_known()
        new_v = logger.today_new_visitors()
        print(f"\n=== Today's Log  ({date.today()}) ===")
        print(f"\n-- Known persons ({len(known)}) --")
        for pid, ptype, name, seen_at, dist in known:
            tag = "EMP" if ptype == "employee" else "VIS"
            print(f"  {seen_at.strftime('%H:%M:%S')}  [{tag}] {pid:<20} {name or '':<20}  dist={dist:.4f}")
        print(f"\n-- New visitors  ({len(new_v)}) --")
        for row_id, name, phone, purpose, seen_at in new_v:
            print(f"  {seen_at.strftime('%H:%M:%S')}  {name:<20} {phone:<15} {purpose}")
        print()
    logger.close()