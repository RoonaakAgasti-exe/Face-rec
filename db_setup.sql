CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS employee_embeddings (
    emp_id      TEXT PRIMARY KEY,
    full_name   TEXT,
    department  TEXT,
    enrolled_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    embedding   vector(512)
);

CREATE INDEX IF NOT EXISTS idx_emp_embedding
    ON employee_embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 50);

CREATE TABLE IF NOT EXISTS visitor_embeddings (
    visitor_id  TEXT PRIMARY KEY,
    full_name   TEXT NOT NULL,
    phone       TEXT,
    purpose     TEXT,
    enrolled_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    embedding   vector(512)
);

CREATE INDEX IF NOT EXISTS idx_vis_embedding
    ON visitor_embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 50);

CREATE TABLE IF NOT EXISTS attendance_log (
    id          SERIAL PRIMARY KEY,
    person_id   TEXT        NOT NULL,
    person_type TEXT        NOT NULL CHECK (person_type IN ('employee', 'frequent_visitor')),
    full_name   TEXT,
    seen_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    distance    FLOAT,
    camera_idx  INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_att_seen_at    ON attendance_log (seen_at DESC);
CREATE INDEX IF NOT EXISTS idx_att_person_id  ON attendance_log (person_id);
CREATE INDEX IF NOT EXISTS idx_att_type       ON attendance_log (person_type);

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
