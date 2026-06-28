# 🎯 Face Recognition Access Control System

A real-time, dual-table face recognition system for access control — distinguishing **employees**, **frequent visitors**, **new walk-ins**, and **spoof attempts**, all logged to PostgreSQL with pgvector.

---

## 📐 Architecture

```
live_camera.py
    │
    ├─► InsightFace (buffalo_l)   — detection + 512-dim embedding
    ├─► AntiSpoof model           — liveness check
    │
    ├─► employee_embeddings  (pgvector)  → GREEN  GATE OPEN
    ├─► visitor_embeddings   (pgvector)  → BLUE   LOGGED
    │
    ├─► LOW CONFIDENCE match             → ORANGE  re-enroll prompt
    ├─► NO MATCH                         → CYAN    terminal input → new_visitor_log
    └─► SPOOF                            → YELLOW  ACCESS DENIED
```

### Database Schema (4 tables)

| Table | Purpose |
|---|---|
| `employee_embeddings` | Permanent staff — emp_id, full_name, department, embedding |
| `visitor_embeddings` | Pre-registered frequent visitors — visitor_id, full_name, phone, purpose, embedding |
| `attendance_log` | Every verified pass-through (employee or frequent_visitor) |
| `new_visitor_log` | Unknown walk-ins who typed their details |

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.10+
- Docker Desktop

### 2. Clone & configure

```bash
git clone https://github.com/your-org/face-rec-v2.git
cd face-rec-v2
cp .env.example .env
# Edit .env — set DB_PASSWORD at minimum
```

### 3. Start the database

```bash
docker compose up -d
```

### 4. Create tables

```bash
# Linux / macOS
cat db_setup.sql | docker exec -i pgvec psql -U postgres -d visit_db

# Windows PowerShell
Get-Content db_setup.sql | docker exec -i pgvec psql -U postgres -d visit_db
```

### 5. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 6. Download the face model

```bash
python download_model.py
```

### 7. Enroll staff

```bash
# Permanent employee
python enroll_employee.py

# Pre-registered frequent visitor
python enroll_visitor.py
```

### 8. Run the camera

```bash
python live_camera.py
# Use a specific camera index:
python live_camera.py --camera 1
# Disable liveness check (useful for testing):
python live_camera.py --no-liveness
```

---

## ⌨️ Camera Controls

| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit |
| `S` | Save screenshot → `screenshots/` |
| `P` | Pause / resume |
| `L` | Print today's log to terminal |
| `E` | Export full log as CSV |

---

## 🎨 Visual Legend

| Colour | Meaning |
|--------|---------|
| 🟢 **Green** | Employee verified — gate open |
| 🔵 **Blue** | Frequent visitor verified |
| 🟠 **Orange** | Low confidence — re-enroll recommended |
| 🩵 **Cyan** | New / unknown visitor — fill details in terminal |
| 🟡 **Yellow** | Spoof detected — access denied |

---

## 🔧 Configuration (`.env`)

```env
DB_HOST=localhost
DB_NAME=visit_db
DB_USER=postgres
DB_PASSWORD=your_strong_password_here
DB_PORT=55432

LIVENESS_THRESHOLD=0.5    # below = spoof (0.0–1.0)
MATCH_THRESHOLD=0.35      # cosine distance — below = confirmed match
LOW_CONF_THRESHOLD=0.55   # between MATCH and this = low confidence
```

---

## 📁 File Map

```
face-rec-v2/
├── .env.example              # copy to .env and fill in passwords
├── docker-compose.yml        # pgvector/pgvector:pg17 service
├── db_setup.sql              # creates all 4 tables + indexes
├── requirements.txt
│
├── download_model.py         # one-time: downloads buffalo_l model
├── enroll_employee.py        # webcam enrollment for staff
├── enroll_visitor.py         # webcam enrollment for frequent visitors
├── convert_npy_to_psql.py    # migrate legacy .npy embeddings to DB
│
├── live_camera.py            # main recognition loop
└── visit_log.py              # VisitLogger class + CLI viewer/exporter
```

---

## 🗂️ Migrating Legacy `.npy` Embeddings

If you have existing embeddings stored as `.npy` files:

```bash
# Point EMBEDDING_FOLDER in .env to your folder, then:
python convert_npy_to_psql.py
```

---

## 📊 Log Viewer

```bash
# Print today's attendance
python visit_log.py --today

# Export full log as CSV
python visit_log.py --export
```

---

## 🛠️ Development Notes

- Recognition runs every **15 frames** in a background thread to keep the video feed smooth.
- New visitor terminal prompts run in a daemon thread — the camera never blocks.
- The `attendance_log` has a **60-second cooldown** per person to avoid duplicate entries.
- The `promoted` column in `new_visitor_log` lets you mark walk-ins who were later enrolled as frequent visitors.

---

## 📋 Requirements

- `insightface >= 0.7.3`
- `opencv-python >= 4.8.0`
- `psycopg2-binary >= 2.9.9`
- `pgvector >= 0.2.4`
- `onnxruntime >= 1.17.0`
- `python-dotenv >= 1.0.0`
- `numpy >= 1.24.0`

---

## 🔒 Security Notes

- Never commit your `.env` file — it's in `.gitignore`
- Change the default `DB_PASSWORD` before deployment
- The liveness (anti-spoof) check is enabled by default; only disable with `--no-liveness` for testing

---

## 📜 License

MIT
