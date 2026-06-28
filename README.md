# Face Recognition Attendance System

A real-time face recognition system built with **ArcFace** and **PostgreSQL + pgvector**. It detects and identifies faces from a live webcam feed, distinguishes existing employees from new (unenrolled) ones, and logs attendance automatically — with anti-spoofing protection to block photos, screen replays, and masks.

---

## Recognition States

![Recognition states](docs/recognition_states.png)

The camera classifies every detected face into one of four states in real time:

| Colour | State | Meaning |
|--------|-------|---------|
| 🟢 **Green** | EXISTING EMPLOYEE | Face matched in the database with high confidence (`dist < 0.35`) |
| 🔵 **Cyan** | NEW EMPLOYEE | Face not enrolled — prompt to run `enroll_camera.py` |
| 🟠 **Orange** | LOW CONFIDENCE | Weak match — may be an existing employee with poor angle or lighting (`0.35 ≤ dist < 0.55`) |
| 🟡 **Yellow** | SPOOF DETECTED | Anti-spoof liveness check failed — photo, screen, or mask |

---

## Architecture

![Architecture](docs/architecture.png)

```
Camera feed
    ↓  Face detection          every frame  (~20 ms)
    ↓  Anti-spoof liveness     every 15 frames
    ↓  ArcFace embedding       512-dim vector
    ↓  pgvector cosine search  <=> operator
    ├─ dist < 0.35   → EXISTING EMPLOYEE  → attendance_log (60 s cooldown)
    ├─ dist < 0.55   → LOW CONFIDENCE     → flag on screen
    └─ dist ≥ 0.55   → NEW EMPLOYEE       → prompt to enroll
```

---

## Features

- **New vs existing employee detection** — the camera distinguishes enrolled staff from completely new faces rather than just showing "unknown"
- **Four-state face classification** — verified, low confidence, new employee, spoof
- **Live enrollment** — add people directly from the webcam without pre-captured photos
- **Anti-spoofing** — rejects printed photos, screen replays, and 3D masks
- **Attendance logging** — every verified face is timestamped with a 60-second cooldown
- **CSV export** — full attendance history exportable on demand
- **Encrypted embeddings** — optional Fernet encryption for stored face vectors
- **Threaded recognition** — detection runs every frame; heavier embedding/DB work runs in a background thread to maintain smooth FPS

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Face recognition | InsightFace — ArcFace (buffalo_l) |
| Anti-spoofing | InsightFace — MiniFASNet (antispoof) |
| Vector search | PostgreSQL 17 + pgvector |
| Database client | psycopg2 + pgvector-python |
| Camera / video | OpenCV |
| Encryption | cryptography (Fernet) |
| Infrastructure | Docker + Docker Compose |

---

## Setup

### Prerequisites

- Python 3.9+
- Docker Desktop

### 1. Clone

```bash
git clone https://github.com/your-username/Face-rec.git
cd Face-rec
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set at minimum `DB_PASSWORD`. Tune the thresholds if needed (see [Threshold tuning](#threshold-tuning) below).

### 4. Start the database

```bash
docker compose up -d
```

### 5. Seed the schema

```bash
docker exec -i pgvec psql -U postgres visit_db < visit_db.sql
```

### 6. Download the model

```bash
python download_model.py
```

---

## Usage

### Enroll new employees

![Enrollment flow](docs/enrollment_flow.png)

```bash
python enroll_camera.py
```

Opens the camera. Type an Employee ID in the terminal, align the face inside the oval guide, and press **Space** to capture. The embedding is saved to both the `embeddings/` folder and PostgreSQL.

| Key | Action |
|-----|--------|
| `Space` | Capture and enroll |
| `R` | Retake / new ID |
| `Q` / `ESC` | Quit |

### Run live recognition with attendance logging

```bash
python live_camera_with_log.py
```

| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit |
| `S` | Screenshot |
| `P` | Pause / resume |
| `L` | Print today's attendance in terminal |
| `E` | Export full attendance log to CSV |

Optional flags:

```bash
python live_camera_with_log.py --camera 1        # use camera index 1
python live_camera_with_log.py --no-liveness     # disable anti-spoof check
python live_camera_with_log.py --width 1920 --height 1080
```

### View or export attendance

```bash
python attendance_log.py             # today's log
python attendance_log.py --export    # CSV export
python attendance_log.py --clear     # delete all records
```

---

## Enrollment pipeline (batch from images)

Run in order when enrolling from a folder of existing photos:

```bash
python create_embeddings.py      # generate .npy embeddings from faces/
python naming.py                 # anonymise filenames
python convert_npy_to_psql.py   # insert into database
```

---

## Threshold tuning

![Threshold diagram](docs/thresholds.png)

All thresholds are set in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `MATCH_THRESHOLD` | `0.35` | Cosine distance below this → **Existing Employee** (green) |
| `LOW_CONF_THRESHOLD` | `0.55` | Distance between MATCH and this → **Low Confidence** (orange); above this → **New Employee** (cyan) |
| `LIVENESS_THRESHOLD` | `0.5` | Anti-spoof score below this → **Spoof** (yellow) |

**Tips:**
- Lower `MATCH_THRESHOLD` (e.g. `0.30`) → stricter matching, fewer false positives
- Raise `LOW_CONF_THRESHOLD` (e.g. `0.65`) → wider "maybe existing" band before classifying as new
- If anti-spoof is too aggressive in poor lighting, raise `LIVENESS_THRESHOLD` slightly or use `--no-liveness`

---

## Configuration reference

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_HOST` | localhost | PostgreSQL host |
| `DB_NAME` | visit_db | Database name |
| `DB_USER` | postgres | DB user |
| `DB_PASSWORD` | — | Required |
| `DB_PORT` | 55432 | Exposed port |
| `IMAGE_FOLDER` | faces | Source JPEG folder for batch enrollment |
| `EMBEDDING_FOLDER` | embeddings | `.npy` face vector storage |
| `LIVENESS_THRESHOLD` | 0.5 | Anti-spoof cutoff (0–1) |
| `MATCH_THRESHOLD` | 0.35 | Cosine distance cutoff for verified match |
| `LOW_CONF_THRESHOLD` | 0.55 | Upper distance bound before classifying as new employee |

---

## Database schema

```sql
CREATE TABLE employee_embeddings (
    emp_id    TEXT PRIMARY KEY,
    embedding VECTOR(512)
);

CREATE TABLE attendance_log (
    id         SERIAL PRIMARY KEY,
    emp_id     TEXT        NOT NULL,
    seen_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    distance   FLOAT,
    camera_idx INTEGER DEFAULT 0
);
```

---

## Project structure

```
Face-rec/
├── live_camera_with_log.py   Main production script — camera + recognition + logging
├── enroll_camera.py          Live enrollment via webcam
├── attendance_log.py         Attendance viewer / exporter
├── create_embeddings.py      Batch embedding generation from image folder
├── naming.py                 Anonymise embedding filenames
├── convert_npy_to_psql.py   Migrate .npy files to PostgreSQL
├── download_model.py         Download ArcFace model weights
├── encrypted_embeddings.py   Encrypted embedding storage (optional)
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── visit_db.sql              Database schema seed
├── docs/                     README diagrams
├── faces/                    Source images (batch enrollment)
├── embeddings/               .npy face vectors
├── enrolled_photos/          Reference photos saved during enrollment
└── screenshots/              Camera captures (S key)
```
