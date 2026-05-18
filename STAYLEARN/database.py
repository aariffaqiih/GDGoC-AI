"""
database.py — SQLite persistence layer untuk fitur konselor StayLearn.

Konsep kunci:
- Trajectory dihitung dari selisih p_dropout (probabilitas mentah), bukan label risiko.
- Counseling record dibuat otomatis oleh save_prediction() sesuai aturan preventif.
- NIM adalah satu-satunya identity key. ON CONFLICT(nim) DO UPDATE memastikan tidak ada duplikasi.
- User authentication ditambahkan dengan tabel users dan hashed password (werkzeug security).
- p_dropout disimpan sebagai 0–1 (fraksi), bukan persen.
- Equity dashboard (SDG 4.5): agregasi berdasarkan lokasi dan kuintil pendapatan, dengan cache TTL.
- Thread-safe cache menggunakan RLock untuk menghindari race condition di multi-worker.
- Raw_features divalidasi sebelum insert dan direpair saat baca jika corrupt.
"""

import json
import os
import sqlite3
import logging
import time
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from flask import g, has_request_context

# Optional pandas untuk kuintil (akan fallback jika tidak ada)
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

# Threshold: penurunan p_dropout >= 5% dianggap "membaik"
IMPROVEMENT_THRESHOLD = 0.05
# Threshold risiko tinggi untuk equity dashboard (p_dropout >= 0.6)
HIGH_RISK_THRESHOLD = 0.6
# Cache TTL untuk equity data (detik)
EQUITY_CACHE_TTL = 300

_DB_PATH: Optional[str] = None
logger = logging.getLogger(__name__)

# ── Cache dengan thread-safe lock ─────────────────────────────────────────────
_equity_cache = {
    "data": None,
    "timestamp": 0,
    "lock": threading.RLock(),
}
_effectiveness_cache = {
    "data": None,
    "timestamp": 0,
    "lock": threading.RLock(),
}

# ─── Schema ───────────────────────────────────────────────────────────────────
SCHEMA = """
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS students (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    nim        TEXT    UNIQUE NOT NULL,
    nama       TEXT,
    created_at TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
    actual_dropout INTEGER DEFAULT 0   -- 0 = masih aktif, 1 = sudah dropout (diisi manual admin)
);

CREATE TABLE IF NOT EXISTS predictions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id   INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    risk_level   TEXT    NOT NULL,
    p_dropout    REAL    NOT NULL,
    raw_features TEXT    NOT NULL,
    source       TEXT    NOT NULL DEFAULT 'individual',
    trajectory   TEXT    NOT NULL DEFAULT 'baru',
    predicted_at TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))
);

CREATE TABLE IF NOT EXISTS counseling_records (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id    INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    prediction_id INTEGER NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    action_type   TEXT    NOT NULL DEFAULT 'konseling',
    status        TEXT    NOT NULL DEFAULT 'pending',
    scheduled_at  TEXT,
    completed_at  TEXT,
    notes         TEXT,
    created_at    TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))
);

-- Tabel user untuk autentikasi konselor
CREATE TABLE IF NOT EXISTS users (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    username   TEXT UNIQUE NOT NULL,
    password   TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);

-- Tabel mentor sukarelawan (SDG 4.a – peer support)
CREATE TABLE IF NOT EXISTS mentors (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    name                TEXT    NOT NULL,
    nim                 TEXT    UNIQUE NOT NULL,
    major               TEXT,
    availability_json   TEXT    DEFAULT '[]',  -- JSON list of available slots
    max_mentees         INTEGER DEFAULT 3,
    is_active           INTEGER DEFAULT 1,
    created_at          TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))
);

CREATE TABLE IF NOT EXISTS mentor_sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    mentor_id       INTEGER NOT NULL REFERENCES mentors(id) ON DELETE CASCADE,
    mentee_id       INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    session_date    TEXT    NOT NULL,
    notes           TEXT,
    rating          INTEGER CHECK(rating BETWEEN 1 AND 5),
    status          TEXT    NOT NULL DEFAULT 'scheduled',  -- scheduled|done|cancelled
    created_at      TEXT    NOT NULL DEFAULT (datetime('now', 'localtime'))
);

CREATE INDEX IF NOT EXISTS idx_pred_student      ON predictions(student_id);
CREATE INDEX IF NOT EXISTS idx_pred_dropout      ON predictions(p_dropout DESC);
CREATE INDEX IF NOT EXISTS idx_pred_student_time ON predictions(student_id, predicted_at DESC);
CREATE INDEX IF NOT EXISTS idx_cr_student        ON counseling_records(student_id);
CREATE INDEX IF NOT EXISTS idx_cr_status         ON counseling_records(status);
CREATE INDEX IF NOT EXISTS idx_cr_done           ON counseling_records(status, completed_at);
CREATE INDEX IF NOT EXISTS idx_ms_mentor         ON mentor_sessions(mentor_id);
CREATE INDEX IF NOT EXISTS idx_ms_mentee         ON mentor_sessions(mentee_id);
"""

# ─── Inisialisasi ─────────────────────────────────────────────────────────────
def init_db(db_path: str) -> None:
    global _DB_PATH
    db_dir = os.path.dirname(os.path.abspath(db_path))
    try:
        os.makedirs(db_dir, exist_ok=True)
    except OSError as e:
        raise RuntimeError(f"Tidak dapat membuat direktori database: {db_dir}") from e

    _DB_PATH = db_path
    try:
        with _get_conn() as conn:
            conn.executescript(SCHEMA)
            _upgrade_schema(conn)
            if os.path.exists(db_path):
                row = conn.execute("PRAGMA integrity_check").fetchone()
                if row[0] != "ok":
                    raise RuntimeError("Database file corrupt (integrity check gagal)")
    except Exception as e:
        raise RuntimeError(f"Gagal menginisialisasi database: {e}") from e

def _upgrade_schema(conn: sqlite3.Connection) -> None:
    """Tambahkan kolom yang mungkin hilang pada database lama (idempotent)."""
    pred_cols = [row[1] for row in conn.execute("PRAGMA table_info(predictions)").fetchall()]
    if "trajectory" not in pred_cols:
        conn.execute("ALTER TABLE predictions ADD COLUMN trajectory TEXT NOT NULL DEFAULT 'baru'")
        logger.info("Kolom 'trajectory' ditambahkan ke tabel predictions")

    student_cols = [row[1] for row in conn.execute("PRAGMA table_info(students)").fetchall()]
    if "actual_dropout" not in student_cols:
        conn.execute("ALTER TABLE students ADD COLUMN actual_dropout INTEGER DEFAULT 0")
        logger.info("Kolom 'actual_dropout' ditambahkan ke tabel students")
    if "birth_date" not in student_cols:
        conn.execute("ALTER TABLE students ADD COLUMN birth_date TEXT")
        logger.info("Kolom 'birth_date' ditambahkan ke tabel students")

    cr_cols = [row[1] for row in conn.execute("PRAGMA table_info(counseling_records)").fetchall()]
    if "intervention_type" not in cr_cols:
        conn.execute(
            "ALTER TABLE counseling_records ADD COLUMN intervention_type TEXT DEFAULT 'umum'"
        )
        logger.info("Kolom 'intervention_type' ditambahkan ke tabel counseling_records")

def verify_connection() -> bool:
    try:
        with _get_conn() as conn:
            conn.execute("SELECT 1").fetchone()
        return True
    except Exception as e:
        logger.error("Database connection verification failed: %s", e)
        return False

# ─── Connection management (thread‑safe per request) ──────────────────────────
@contextmanager
def _get_conn():
    """
    Context manager untuk koneksi database.
    - Jika dalam konteks request Flask, gunakan koneksi yang sudah ada di g.db.
    - Di luar request (CLI, test), buat koneksi baru dan tutup setelah selesai.
    """
    if has_request_context() and hasattr(g, 'db'):
        # Koneksi dikelola oleh Flask (app.teardown_appcontext)
        yield g.db
    else:
        # Untuk lingkungan tanpa Flask (script, test)
        if _DB_PATH is None:
            raise RuntimeError("DB belum diinisialisasi. Panggil init_db() terlebih dahulu.")
        conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

# ─── Students ─────────────────────────────────────────────────────────────────
def upsert_student(
    nim: str,
    nama: Optional[str] = None,
    birth_date: Optional[str] = None,
) -> int:
    nim = nim.strip()
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO students (nim, nama, birth_date) VALUES (?, ?, ?)
            ON CONFLICT(nim) DO UPDATE SET
                nama       = COALESCE(EXCLUDED.nama,       students.nama),
                birth_date = COALESCE(EXCLUDED.birth_date, students.birth_date)
            """,
            (nim, nama.strip() if nama else None, birth_date),
        )
        row = conn.execute(
            "SELECT id FROM students WHERE nim = ?", (nim,)
        ).fetchone()
        return row["id"]


def get_student_by_nim(nim: str) -> Optional[dict]:
    """Ambil data mahasiswa berdasarkan NIM. Returns None jika tidak ada."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT id, nim, nama, created_at, actual_dropout FROM students WHERE nim = ?",
            (nim.strip(),),
        ).fetchone()
    return dict(row) if row else None


def verify_student_birth_date(nim: str, birth_date: str) -> bool:
    """
    Verifikasi bahwa NIM dan tanggal lahir cocok di database.
    Digunakan untuk autentikasi halaman riwayat tanpa login konselor.
    Returns False jika NIM tidak ada, birth_date NULL, atau tidak cocok.
    Tidak membedakan kasus untuk mencegah enumerasi NIM.
    """
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT birth_date FROM students WHERE nim = ?", (nim.strip(),)
        ).fetchone()
    if row is None or row["birth_date"] is None:
        return False
    return row["birth_date"] == birth_date.strip()


def get_prediction_history_by_nim(nim: str) -> List[Dict[str, Any]]:
    """
    Riwayat prediksi mahasiswa berdasarkan NIM, diurutkan ASC (terlama ke terbaru).
    Digunakan untuk halaman riwayat mahasiswa (grafik time-series).
    Returns: list of dicts. Empty list jika belum ada prediksi.
    """
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT p.id, p.risk_level, p.p_dropout, p.trajectory,
                   p.source, p.predicted_at
            FROM   predictions p
            JOIN   students    s ON s.id = p.student_id
            WHERE  s.nim = ?
            ORDER  BY p.predicted_at ASC
            """,
            (nim.strip(),),
        ).fetchall()
    return [dict(r) for r in rows]


def update_actual_dropout(student_id: int, is_dropout: bool) -> bool:
    """Update status dropout aktual mahasiswa (digunakan untuk fairness metrics)."""
    with _get_conn() as conn:
        conn.execute(
            "UPDATE students SET actual_dropout = ? WHERE id = ?",
            (1 if is_dropout else 0, student_id)
        )
        return conn.total_changes > 0

# ─── Trajectory ───────────────────────────────────────────────────────────────
def _compute_trajectory(student_id: int, new_p_dropout: float) -> str:
    with _get_conn() as conn:
        row = conn.execute(
            """SELECT p_dropout FROM predictions
               WHERE student_id = ?
               ORDER BY predicted_at DESC LIMIT 1""",
            (student_id,),
        ).fetchone()

    if row is None:
        return "baru"

    prev = row["p_dropout"]
    diff = new_p_dropout - prev

    if diff > 0.005:
        return "memburuk"
    if diff < -IMPROVEMENT_THRESHOLD:
        return "membaik"
    return "stagnan"

# ─── Validasi dan repair raw_features ─────────────────────────────────────────
_REQUIRED_FEATURE_KEYS = {
    "location_type", "family_income", "financial_aid_status", "distance_to_institute",
    "internet_connectivity_issues", "motivation_score", "career_alignment",
    "stress_levels", "family_support", "attendance_rate", "test_scores_avg",
    "backlogs", "teaching_quality_rating"
}

_DEFAULT_FEATURE_VALUES = {
    "location_type": "Urban",
    "family_income": 10000.0,
    "financial_aid_status": 1,
    "distance_to_institute": 10.0,
    "internet_connectivity_issues": 0,
    "motivation_score": 5,
    "career_alignment": 2,
    "stress_levels": 2,
    "family_support": 2,
    "attendance_rate": 75.0,
    "test_scores_avg": 65.0,
    "backlogs": 0,
    "teaching_quality_rating": 6,
}

def _validate_and_repair_features(raw_features: Dict[str, Any]) -> Dict[str, Any]:
    """Pastikan dictionary memiliki semua field wajib, isi default jika kurang."""
    repaired = dict(raw_features)
    for key in _REQUIRED_FEATURE_KEYS:
        if key not in repaired or repaired[key] is None:
            repaired[key] = _DEFAULT_FEATURE_VALUES.get(key)
            logger.warning("Missing feature key '%s' in raw_features, using default", key)
    # Pastikan tipe data benar
    try:
        repaired["family_income"] = float(repaired["family_income"])
        repaired["distance_to_institute"] = float(repaired["distance_to_institute"])
        repaired["attendance_rate"] = float(repaired["attendance_rate"])
        repaired["test_scores_avg"] = float(repaired["test_scores_avg"])
        for k in ["financial_aid_status", "internet_connectivity_issues", "motivation_score",
                  "career_alignment", "stress_levels", "family_support", "backlogs",
                  "teaching_quality_rating"]:
            if k in repaired:
                repaired[k] = int(float(repaired[k]))
    except (ValueError, TypeError) as e:
        logger.warning("Type conversion error in raw_features: %s", e)
    return repaired

def _safe_parse_raw_features(raw_json: str) -> Optional[Dict[str, Any]]:
    """Parse JSON raw_features dengan fallback jika corrupt."""
    try:
        data = json.loads(raw_json)
        if not isinstance(data, dict):
            return None
        return data
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning("Corrupt raw_features JSON: %s", e)
        # Attempt repair: extract using regex (sederhana)
        import re
        repaired = {}
        for key in _REQUIRED_FEATURE_KEYS:
            pattern = rf'"{key}"\s*:\s*([^,]+)'
            match = re.search(pattern, raw_json)
            if match:
                val = match.group(1).strip().strip('"')
                try:
                    if key in ["location_type"]:
                        repaired[key] = val
                    else:
                        repaired[key] = float(val) if '.' in val else int(float(val))
                except ValueError:
                    repaired[key] = _DEFAULT_FEATURE_VALUES.get(key)
            else:
                repaired[key] = _DEFAULT_FEATURE_VALUES.get(key)
        return repaired if repaired else None

# ─── Predictions (dengan transaksi tunggal) ───────────────────────────────────
def save_prediction(
    student_id: int,
    risk_level: str,
    p_dropout: float,
    raw_features: dict,
    source: str = "individual",
) -> int:
    # Validasi dan repair raw_features sebelum disimpan
    repaired_features = _validate_and_repair_features(raw_features)
    features_json = json.dumps(repaired_features)

    trajectory = _compute_trajectory(student_id, p_dropout)
    prediction_id = None

    with _get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO predictions
                (student_id, risk_level, p_dropout, raw_features, source, trajectory)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                student_id,
                risk_level,
                p_dropout,
                features_json,
                source,
                trajectory,
            ),
        )
        prediction_id = cur.lastrowid
        _auto_create_counseling(conn, student_id, prediction_id, risk_level, trajectory)

    # Invalidate caches karena ada data baru
    invalidate_equity_cache()
    invalidate_effectiveness_cache()
    return prediction_id

def _auto_create_counseling(
    conn: sqlite3.Connection,
    student_id: int,
    prediction_id: int,
    risk_level: str,
    trajectory: str,
) -> None:
    if risk_level == "rendah" or trajectory == "membaik":
        return

    action_type = "konseling" if risk_level == "tinggi" else "monitoring"

    conn.execute(
        """
        INSERT INTO counseling_records
            (student_id, prediction_id, action_type, status)
        VALUES (?, ?, ?, 'pending')
        """,
        (student_id, prediction_id, action_type),
    )

# ─── Users (Flask-Login) ──────────────────────────────────────────────────────
class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username

    def check_password(self, plaintext):
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT password FROM users WHERE id = ?", (self.id,)
            ).fetchone()
            if row:
                return check_password_hash(row["password"], plaintext)
        return False

def get_user_by_id(user_id: int):
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT id, username FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if row:
            return User(row["id"], row["username"])
    return None

def get_user_by_username(username: str):
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT id, username FROM users WHERE username = ?", (username,)
        ).fetchone()
        if row:
            return User(row["id"], row["username"])
    return None

def create_user(username: str, password: str) -> bool:
    hashed = generate_password_hash(password)
    try:
        with _get_conn() as conn:
            conn.execute(
                "INSERT INTO users (username, password) VALUES (?, ?)",
                (username, hashed)
            )
        return True
    except sqlite3.IntegrityError:
        return False

# ─── Dashboard Queries ────────────────────────────────────────────────────────
def get_priority_queue(
    filter_risk: Optional[str] = None,
    filter_status: Optional[str] = None,
) -> list:
    params = []
    risk_clause = ""
    status_clause = ""

    if filter_risk and filter_risk != "semua":
        risk_clause = "AND p.risk_level = ?"
        params.append(filter_risk)

    if filter_status and filter_status in ("pending", "scheduled"):
        status_clause = "AND cr.status = ?"
        params.append(filter_status)
    else:
        status_clause = "AND cr.status IN ('pending', 'scheduled')"

    query = f"""
        SELECT
            s.id          AS student_id,
            s.nim,
            s.nama,
            p.id          AS prediction_id,
            p.risk_level,
            p.p_dropout,
            p.trajectory,
            p.predicted_at,
            p.source,
            cr.id         AS counseling_id,
            cr.action_type,
            cr.status,
            cr.scheduled_at
        FROM counseling_records cr
        JOIN predictions p ON cr.prediction_id = p.id
        JOIN students    s ON cr.student_id     = s.id
        WHERE 1=1
        {status_clause}
        {risk_clause}
        ORDER BY
            CASE p.risk_level
                WHEN 'tinggi' THEN 1
                WHEN 'sedang' THEN 2
                ELSE 3
            END,
            p.p_dropout DESC
    """
    with _get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]

def get_student_history(student_id: int) -> Optional[dict]:
    with _get_conn() as conn:
        student = conn.execute(
            "SELECT * FROM students WHERE id = ?", (student_id,)
        ).fetchone()
        if student is None:
            return None

        predictions = conn.execute(
            """
            SELECT
                p.*,
                cr.id           AS counseling_id,
                cr.action_type,
                cr.status       AS counseling_status,
                cr.scheduled_at,
                cr.completed_at,
                cr.notes
            FROM predictions p
            LEFT JOIN counseling_records cr ON cr.prediction_id = p.id
            WHERE p.student_id = ?
            ORDER BY p.predicted_at DESC
            """,
            (student_id,),
        ).fetchall()

    return {
        "student": dict(student),
        "predictions": [dict(p) for p in predictions],
    }

def get_counseling_record(counseling_id: int) -> Optional[dict]:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM counseling_records WHERE id = ?", (counseling_id,)
        ).fetchone()
    return dict(row) if row else None

def update_counseling(counseling_id: int, **kwargs) -> bool:
    allowed = {"status", "scheduled_at", "completed_at", "notes", "intervention_type"}
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return False

    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [counseling_id]

    with _get_conn() as conn:
        cursor = conn.execute(
            f"UPDATE counseling_records SET {set_clause} WHERE id = ?",
            values,
        )
        changed = cursor.rowcount > 0

    if changed and updates.get("status") == "done":
        invalidate_effectiveness_cache()

    return changed

def get_dashboard_stats() -> dict:
    with _get_conn() as conn:
        total_students = conn.execute(
            "SELECT COUNT(*) FROM students"
        ).fetchone()[0]

        pending_tinggi = conn.execute(
            """SELECT COUNT(*) FROM counseling_records cr
               JOIN predictions p ON cr.prediction_id = p.id
               WHERE cr.status = 'pending' AND p.risk_level = 'tinggi'"""
        ).fetchone()[0]

        pending_sedang = conn.execute(
            """SELECT COUNT(*) FROM counseling_records cr
               JOIN predictions p ON cr.prediction_id = p.id
               WHERE cr.status = 'pending' AND p.risk_level = 'sedang'"""
        ).fetchone()[0]

        done_this_month = conn.execute(
            """SELECT COUNT(*) FROM counseling_records
               WHERE status = 'done'
               AND strftime('%Y-%m', completed_at) = strftime('%Y-%m', 'now', 'localtime')"""
        ).fetchone()[0]

    return {
        "total_students": total_students,
        "pending_tinggi": pending_tinggi,
        "pending_sedang": pending_sedang,
        "done_this_month": done_this_month,
    }

# ─── Analytics Helpers (module-private) ──────────────────────────────────────
def _count_values(values: list) -> dict:
    """Hitung frekuensi tiap nilai dan kembalikan dict terurut."""
    counter: dict = {}
    for v in values:
        key = str(v)
        counter[key] = counter.get(key, 0) + 1
    return dict(sorted(counter.items()))

def _make_histogram(values: list, num_bins: int = 10) -> dict:
    """
    Buat data histogram untuk Chart.js bar chart.

    Args:
        values: list angka
        num_bins: jumlah bin

    Returns:
        dict dengan 'labels' (string rentang) dan 'counts' (integer per bin)
    """
    if not values:
        return {"labels": [], "counts": []}

    fv = [float(v) for v in values]
    min_val = min(fv)
    max_val = max(fv)

    if min_val == max_val:
        return {"labels": [f"{min_val:.1f}"], "counts": [len(fv)]}

    bin_width = (max_val - min_val) / num_bins
    counts = [0] * num_bins

    for v in fv:
        idx = min(int((v - min_val) / bin_width), num_bins - 1)
        counts[idx] += 1

    labels = [
        f"{min_val + i * bin_width:.1f}–{min_val + (i + 1) * bin_width:.1f}"
        for i in range(num_bins)
    ]
    return {"labels": labels, "counts": counts}

# ─── Analytics Query ──────────────────────────────────────────────────────────
def get_analytics_data() -> dict:
    """
    Kumpulkan dan agregasi data dari predictions & counseling_records
    untuk semua chart di halaman analytics konselor.

    Catatan:
        - p_dropout dikonversi dari fraksi (0–1) ke persen (0–100) untuk histogram.
        - raw_features di-parse dari JSON; baris yang gagal di-parse diabaikan.

    Returns:
        dict siap di-JSON-serialize untuk Chart.js.
    """
    with _get_conn() as conn:
        pred_rows = conn.execute(
            "SELECT raw_features, risk_level, p_dropout, trajectory FROM predictions"
        ).fetchall()

        action_rows = conn.execute(
            "SELECT action_type, COUNT(*) AS cnt FROM counseling_records GROUP BY action_type"
        ).fetchall()

        status_rows = conn.execute(
            "SELECT status, COUNT(*) AS cnt FROM counseling_records GROUP BY status"
        ).fetchall()

    # Kolom fitur yang ingin diekstrak dari raw_features JSON
    feature_keys = [
        "location_type",
        "family_income",
        "financial_aid_status",
        "distance_to_institute",
        "internet_connectivity_issues",
        "motivation_score",
        "stress_levels",
        "career_alignment",
        "family_support",
        "attendance_rate",
        "test_scores_avg",
        "backlogs",
        "teaching_quality_rating",
    ]

    feature_lists: dict = {k: [] for k in feature_keys}
    risk_levels: list = []
    p_dropouts: list = []   # disimpan sebagai persen
    trajectories: list = []

    for row in pred_rows:
        # Parse raw_features JSON dengan safe parser
        rf = _safe_parse_raw_features(row["raw_features"])
        if rf is None:
            continue
        for key in feature_keys:
            val = rf.get(key)
            if val is not None:
                feature_lists[key].append(val)

        risk_levels.append(row["risk_level"])
        p_dropouts.append(float(row["p_dropout"]) * 100.0)
        if row["trajectory"]:
            trajectories.append(row["trajectory"])

    return {
        "total_records": len(pred_rows),
        # ── Demografi & Ekonomi ──────────────────────────────────────────────
        "location_type":               _count_values(feature_lists["location_type"]),
        "family_income":               _make_histogram(feature_lists["family_income"], num_bins=8),
        "financial_aid_status":        _count_values(feature_lists["financial_aid_status"]),
        "distance_to_institute":       _make_histogram(feature_lists["distance_to_institute"], num_bins=8),
        "internet_connectivity_issues": _count_values(feature_lists["internet_connectivity_issues"]),
        "family_support":              _count_values(feature_lists["family_support"]),
        # ── Psikologis ───────────────────────────────────────────────────────
        "motivation_score":            _count_values(feature_lists["motivation_score"]),
        "stress_levels":               _count_values(feature_lists["stress_levels"]),
        "career_alignment":            _count_values(feature_lists["career_alignment"]),
        # ── Akademik ─────────────────────────────────────────────────────────
        "attendance_rate":             _make_histogram(feature_lists["attendance_rate"], num_bins=9),
        "test_scores_avg":             _make_histogram(feature_lists["test_scores_avg"], num_bins=8),
        "backlogs":                    _count_values(feature_lists["backlogs"]),
        "teaching_quality_rating":     _count_values(feature_lists["teaching_quality_rating"]),
        # ── Risiko & Konseling ───────────────────────────────────────────────
        "risk_level":   _count_values(risk_levels),
        "p_dropout":    _make_histogram(p_dropouts, num_bins=10),
        "trajectory":   _count_values(trajectories),
        "action_type":  {row["action_type"]: row["cnt"] for row in action_rows},
        "status":       {row["status"]: row["cnt"] for row in status_rows},
    }

# ─── Equity Dashboard (SDG 4.5) ───────────────────────────────────────────────
def _get_latest_prediction_per_student() -> List[Dict[str, Any]]:
    """
    Mengambil prediksi terbaru untuk setiap mahasiswa beserta lokasi, pendapatan,
    dan status dropout aktual (jika tersedia).
    Returns list of dicts with keys: student_id, location_type, family_income,
    p_dropout, actual_dropout.
    """
    query = """
        SELECT
            s.id AS student_id,
            s.actual_dropout,
            p.p_dropout,
            p.raw_features
        FROM students s
        JOIN predictions p ON s.id = p.student_id
        WHERE p.id IN (
            SELECT MAX(id) FROM predictions GROUP BY student_id
        )
    """
    with _get_conn() as conn:
        rows = conn.execute(query).fetchall()

    result = []
    for row in rows:
        rf = _safe_parse_raw_features(row["raw_features"])
        if rf is None:
            continue
        loc = rf.get("location_type")
        income = rf.get("family_income")
        if loc is None or income is None or income <= 0:
            continue
        result.append({
            "student_id": row["student_id"],
            "location_type": loc,
            "family_income": float(income),
            "p_dropout": float(row["p_dropout"]),
            "actual_dropout": row["actual_dropout"] if row["actual_dropout"] is not None else 0,
        })
    return result

def _compute_quintiles(incomes: List[float]) -> List[str]:
    """
    Hitung label kuintil berdasarkan daftar pendapatan.
    Menggunakan pandas.qcut jika tersedia, fallback ke manual.
    Returns list label kuintil (Q1..Q5) dengan panjang sama dengan incomes.
    """
    if not incomes:
        return []
    if HAS_PANDAS and pd:
        try:
            quintiles, bins = pd.qcut(incomes, q=5, labels=False, duplicates='drop', retbins=True)
            unique_labels = sorted(set(quintiles))
            label_map = {old: f"Q{idx+1}" for idx, old in enumerate(unique_labels)}
            return [label_map[q] for q in quintiles]
        except Exception as e:
            logger.warning("Gagal menghitung kuintil dengan pandas: %s, fallback ke manual", e)

    # Fallback manual
    sorted_incomes = sorted(incomes)
    n = len(sorted_incomes)
    quintile_size = n // 5
    labels = []
    for inc in incomes:
        pos = sorted_incomes.index(inc)
        if pos < quintile_size:
            labels.append("Q1")
        elif pos < 2 * quintile_size:
            labels.append("Q2")
        elif pos < 3 * quintile_size:
            labels.append("Q3")
        elif pos < 4 * quintile_size:
            labels.append("Q4")
        else:
            labels.append("Q5")
    return labels

def _aggregate_by_location(data: List[Dict]) -> Dict[str, Dict]:
    """Agregasi berdasarkan location_type."""
    result = {}
    for item in data:
        loc = item["location_type"]
        if loc not in result:
            result[loc] = {"p_dropout_sum": 0.0, "high_risk_count": 0, "count": 0}
        result[loc]["p_dropout_sum"] += item["p_dropout"]
        if item["p_dropout"] >= HIGH_RISK_THRESHOLD:
            result[loc]["high_risk_count"] += 1
        result[loc]["count"] += 1

    output = {}
    for loc, agg in result.items():
        output[loc] = {
            "avg_p_dropout": round(agg["p_dropout_sum"] / agg["count"], 4),
            "high_risk_ratio": round(agg["high_risk_count"] / agg["count"], 4),
            "count": agg["count"],
        }
    return output

def _aggregate_by_quintile(data: List[Dict], quintile_labels: List[str]) -> Dict[str, Dict]:
    """Agregasi berdasarkan label kuintil."""
    result = {}
    for idx, item in enumerate(data):
        q = quintile_labels[idx]
        if q not in result:
            result[q] = {"p_dropout_sum": 0.0, "high_risk_count": 0, "count": 0, "min_income": float('inf'), "max_income": float('-inf')}
        result[q]["p_dropout_sum"] += item["p_dropout"]
        if item["p_dropout"] >= HIGH_RISK_THRESHOLD:
            result[q]["high_risk_count"] += 1
        result[q]["count"] += 1
        inc = item["family_income"]
        if inc < result[q]["min_income"]:
            result[q]["min_income"] = inc
        if inc > result[q]["max_income"]:
            result[q]["max_income"] = inc

    output = {}
    for q, agg in result.items():
        output[q] = {
            "range": f"{int(agg['min_income'])}–{int(agg['max_income'])}",
            "avg_p_dropout": round(agg["p_dropout_sum"] / agg["count"], 4),
            "high_risk_ratio": round(agg["high_risk_count"] / agg["count"], 4),
            "count": agg["count"],
        }
    ordered = {}
    for i in range(1, 6):
        key = f"Q{i}"
        if key in output:
            ordered[key] = output[key]
    return ordered

def _compute_fairness_metrics(data: List[Dict]) -> Optional[Dict]:
    """
    Hitung fairness metrics per subgroup (lokasi) jika data aktual dropout tersedia.
    Metrics: false positive rate, false negative rate, dan peringatan bias.
    Returns dict dengan struktur:
    {
        "has_actual_data": bool,
        "warning": str or None,
        "by_location": { loc: {"fpr": ..., "fnr": ...} }
    }
    """
    has_actual = any(item.get("actual_dropout", 0) == 1 for item in data)
    if not has_actual:
        return {
            "has_actual_data": False,
            "warning": "Data dropout aktual belum tersedia. Fairness metrics tidak dapat dihitung.",
            "by_location": {}
        }

    location_stats = {}
    for item in data:
        loc = item["location_type"]
        if loc not in location_stats:
            location_stats[loc] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        pred_positive = item["p_dropout"] >= 0.5
        actual_positive = item["actual_dropout"] == 1
        if pred_positive and actual_positive:
            location_stats[loc]["tp"] += 1
        elif pred_positive and not actual_positive:
            location_stats[loc]["fp"] += 1
        elif not pred_positive and not actual_positive:
            location_stats[loc]["tn"] += 1
        else:
            location_stats[loc]["fn"] += 1

    by_location = {}
    for loc, stats in location_stats.items():
        total_actual_positive = stats["tp"] + stats["fn"]
        total_actual_negative = stats["fp"] + stats["tn"]
        fpr = stats["fp"] / total_actual_negative if total_actual_negative > 0 else 0
        fnr = stats["fn"] / total_actual_positive if total_actual_positive > 0 else 0
        by_location[loc] = {"fpr": round(fpr, 4), "fnr": round(fnr, 4)}

    warning = None
    if len(by_location) >= 2:
        fpr_values = [v["fpr"] for v in by_location.values()]
        fnr_values = [v["fnr"] for v in by_location.values()]
        if max(fpr_values) - min(fpr_values) > 0.2:
            warning = "Perbedaan False Positive Rate antar lokasi > 0.2. Model mungkin bias terhadap kelompok tertentu."
        elif max(fnr_values) - min(fnr_values) > 0.2:
            warning = "Perbedaan False Negative Rate antar lokasi > 0.2. Model mungkin bias terhadap kelompok tertentu."

    return {
        "has_actual_data": True,
        "warning": warning,
        "by_location": by_location,
    }

def get_equity_data(use_cache: bool = True) -> Dict[str, Any]:
    """
    Menghasilkan data untuk Equity Dashboard (SDG 4.5):
    - Agregasi berdasarkan lokasi tempat tinggal
    - Agregasi berdasarkan kuintil pendapatan keluarga
    - Fairness metrics (jika data aktual tersedia)

    Menggunakan cache dengan TTL 5 menit, thread-safe.
    """
    global _equity_cache
    now = time.time()

    if use_cache:
        with _equity_cache["lock"]:
            if _equity_cache["data"] is not None and (now - _equity_cache["timestamp"] < EQUITY_CACHE_TTL):
                return _equity_cache["data"]

    # Ambil data terbaru per mahasiswa
    student_data = _get_latest_prediction_per_student()
    if not student_data:
        result = {
            "by_location": {},
            "by_income_quintile": {},
            "fairness": {
                "has_actual_data": False,
                "warning": "Tidak ada data prediksi yang tersedia.",
                "by_location": {}
            },
            "summary": {
                "total_students": 0,
                "excluded_due_to_missing_income": 0,
                "quintile_method": "equal_frequency"
            }
        }
        with _equity_cache["lock"]:
            _equity_cache["data"] = result
            _equity_cache["timestamp"] = now
        return result

    by_location = _aggregate_by_location(student_data)

    incomes = [d["family_income"] for d in student_data]
    quintile_labels = _compute_quintiles(incomes)
    by_quintile = _aggregate_by_quintile(student_data, quintile_labels)

    fairness = _compute_fairness_metrics(student_data)

    result = {
        "by_location": by_location,
        "by_income_quintile": by_quintile,
        "fairness": fairness,
        "summary": {
            "total_students": len(student_data),
            "excluded_due_to_missing_income": 0,
            "quintile_method": "pandas_qcut" if HAS_PANDAS else "manual_rank"
        }
    }

    if use_cache:
        with _equity_cache["lock"]:
            _equity_cache["data"] = result
            _equity_cache["timestamp"] = now
    return result

def invalidate_equity_cache() -> None:
    """Invalidasi cache equity data (thread-safe)."""
    with _equity_cache["lock"]:
        _equity_cache["data"] = None
        _equity_cache["timestamp"] = 0
        logger.info("Equity cache invalidated")

# ─── Intervention Effectiveness Tracking (SDG 4.1) ───────────────────────────
EFFECTIVENESS_MIN_DAYS: int = 30
EFFECTIVENESS_UNCHANGED_THRESHOLD: float = 0.02
EFFECTIVENESS_CACHE_TTL: int = 300
EFFECTIVENESS_MIN_SAMPLE: int = 5

def invalidate_effectiveness_cache() -> None:
    """Invalidasi cache effectiveness (thread-safe)."""
    with _effectiveness_cache["lock"]:
        _effectiveness_cache["data"] = None
        _effectiveness_cache["timestamp"] = 0.0
        logger.info("Effectiveness cache invalidated")

def _fetch_effectiveness_rows(min_days: int) -> List[Dict[str, Any]]:
    """
    Ambil semua counseling_records berstatus 'done' beserta:
    - p_dropout dari prediksi "sebelum" (via FK prediction_id)
    - p_dropout dari prediksi "sesudah" (correlated subquery)
    """
    if min_days < 0:
        raise ValueError(f"min_days harus >= 0, diterima: {min_days}")

    plus_days_expr = f"+{min_days} days"

    sql = """
        SELECT
            cr.id                               AS counseling_id,
            cr.student_id,
            cr.completed_at,
            s.nim,
            s.nama,
            p_before.p_dropout                  AS p_dropout_before,
            p_before.predicted_at               AS before_predicted_at,
            (
                SELECT p_after.p_dropout
                FROM   predictions  p_after
                WHERE  p_after.student_id   = cr.student_id
                  AND  p_after.predicted_at > datetime(cr.completed_at, :plus_days)
                ORDER  BY p_after.predicted_at DESC
                LIMIT  1
            )                                   AS p_dropout_after,
            (
                SELECT p_after.predicted_at
                FROM   predictions  p_after
                WHERE  p_after.student_id   = cr.student_id
                  AND  p_after.predicted_at > datetime(cr.completed_at, :plus_days)
                ORDER  BY p_after.predicted_at DESC
                LIMIT  1
            )                                   AS after_predicted_at
        FROM   counseling_records cr
        JOIN   students           s        ON s.id        = cr.student_id
        JOIN   predictions        p_before ON p_before.id = cr.prediction_id
        WHERE  cr.status       = 'done'
          AND  cr.completed_at IS NOT NULL
        ORDER  BY cr.completed_at DESC
    """
    with _get_conn() as conn:
        rows = conn.execute(sql, {"plus_days": plus_days_expr}).fetchall()
    return [dict(row) for row in rows]

def _classify_effectiveness_row(
    row: Dict[str, Any],
    min_days: int,
) -> Dict[str, Any]:
    """
    Klasifikasi satu baris efektivitas konseling.
    """
    completed_at_str: Optional[str] = row.get("completed_at")
    p_before_raw = row.get("p_dropout_before")
    p_after_raw = row.get("p_dropout_after")

    try:
        completed_dt = datetime.strptime(str(completed_at_str)[:19], "%Y-%m-%d %H:%M:%S")
        days_elapsed = (datetime.now() - completed_dt).days
    except (ValueError, TypeError):
        return {
            "counseling_id": row.get("counseling_id"),
            "student_id": row.get("student_id"),
            "nim": row.get("nim") or "-",
            "nama": row.get("nama") or "-",
            "completed_at": completed_at_str,
            "days_elapsed": None,
            "p_dropout_before_pct": None,
            "p_dropout_after_pct": None,
            "delta_pct": None,
            "improvement_category": None,
            "evaluation_status": "invalid_date",
            "before_predicted_at": row.get("before_predicted_at"),
            "after_predicted_at": None,
        }

    try:
        p_before = float(p_before_raw)
    except (TypeError, ValueError):
        p_before = 0.0

    if days_elapsed < min_days:
        return {
            "counseling_id": row["counseling_id"],
            "student_id": row["student_id"],
            "nim": row.get("nim") or "-",
            "nama": row.get("nama") or "-",
            "completed_at": completed_at_str,
            "days_elapsed": days_elapsed,
            "p_dropout_before_pct": round(p_before * 100, 1),
            "p_dropout_after_pct": None,
            "delta_pct": None,
            "improvement_category": None,
            "evaluation_status": "too_recent",
            "before_predicted_at": row.get("before_predicted_at"),
            "after_predicted_at": None,
        }

    if p_after_raw is None:
        return {
            "counseling_id": row["counseling_id"],
            "student_id": row["student_id"],
            "nim": row.get("nim") or "-",
            "nama": row.get("nama") or "-",
            "completed_at": completed_at_str,
            "days_elapsed": days_elapsed,
            "p_dropout_before_pct": round(p_before * 100, 1),
            "p_dropout_after_pct": None,
            "delta_pct": None,
            "improvement_category": None,
            "evaluation_status": "pending_followup",
            "before_predicted_at": row.get("before_predicted_at"),
            "after_predicted_at": None,
        }

    try:
        p_after = float(p_after_raw)
    except (TypeError, ValueError):
        p_after = p_before

    delta = p_before - p_after
    delta_pct = round(delta * 100, 2)

    if delta >= EFFECTIVENESS_UNCHANGED_THRESHOLD:
        improvement_category = "improved"
    elif delta <= -EFFECTIVENESS_UNCHANGED_THRESHOLD:
        improvement_category = "worsened"
    else:
        improvement_category = "unchanged"

    return {
        "counseling_id": row["counseling_id"],
        "student_id": row["student_id"],
        "nim": row.get("nim") or "-",
        "nama": row.get("nama") or "-",
        "completed_at": completed_at_str,
        "days_elapsed": days_elapsed,
        "p_dropout_before_pct": round(p_before * 100, 1),
        "p_dropout_after_pct": round(p_after * 100, 1),
        "delta_pct": delta_pct,
        "improvement_category": improvement_category,
        "evaluation_status": "evaluated",
        "before_predicted_at": row.get("before_predicted_at"),
        "after_predicted_at": row.get("after_predicted_at"),
    }

def _compute_effectiveness_summary(
    classified_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Hitung agregat statistik dari semua record yang sudah diklasifikasi.
    """
    evaluated = [r for r in classified_records if r["evaluation_status"] == "evaluated"]
    too_recent = [r for r in classified_records if r["evaluation_status"] == "too_recent"]
    pending = [r for r in classified_records if r["evaluation_status"] == "pending_followup"]

    improved  = [r for r in evaluated if r["improvement_category"] == "improved"]
    worsened  = [r for r in evaluated if r["improvement_category"] == "worsened"]
    unchanged = [r for r in evaluated if r["improvement_category"] == "unchanged"]

    n_evaluated = len(evaluated)

    if n_evaluated > 0:
        avg_reduction_pct = round(sum(r["delta_pct"] for r in evaluated) / n_evaluated, 2)
        improved_pct = round(len(improved) / n_evaluated * 100, 1)
        delta_distribution = [r["delta_pct"] for r in evaluated if r["delta_pct"] is not None]
    else:
        avg_reduction_pct = None
        improved_pct = None
        delta_distribution = []

    return {
        "total_counseling_done": len(classified_records),
        "total_evaluated": n_evaluated,
        "total_too_recent": len(too_recent),
        "total_pending_followup": len(pending),
        "avg_dropout_reduction_pct": avg_reduction_pct,
        "improved_count": len(improved),
        "worsened_count": len(worsened),
        "unchanged_count": len(unchanged),
        "improved_pct": improved_pct,
        "data_sufficient": n_evaluated >= EFFECTIVENESS_MIN_SAMPLE,
        "min_sample_required": EFFECTIVENESS_MIN_SAMPLE,
        "delta_distribution": delta_distribution,
        "methodology_notes": [
            f"Evaluasi dilakukan minimal {EFFECTIVENESS_MIN_DAYS} hari setelah konseling selesai.",
            f"'Membaik' = penurunan p_dropout >= {EFFECTIVENESS_UNCHANGED_THRESHOLD * 100:.0f}pp.",
            "Selection bias mungkin ada: mahasiswa yang di-prediksi ulang bisa bukan sampel representatif.",
            "Korelasi bukan kausalitas — faktor lain mungkin berkontribusi pada perubahan risiko.",
        ],
    }

def get_intervention_effectiveness(
    min_days: int = EFFECTIVENESS_MIN_DAYS,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Hitung dan kembalikan data efektivitas intervensi konseling (SDG 4.1).
    Thread-safe dengan cache.
    """
    global _effectiveness_cache
    now = time.time()

    if use_cache:
        with _effectiveness_cache["lock"]:
            if _effectiveness_cache["data"] is not None and (now - _effectiveness_cache["timestamp"] < EFFECTIVENESS_CACHE_TTL):
                return _effectiveness_cache["data"]

    raw_rows = _fetch_effectiveness_rows(min_days)
    classified = [_classify_effectiveness_row(row, min_days) for row in raw_rows]
    summary = _compute_effectiveness_summary(classified)

    result = {
        "summary": summary,
        "records": classified,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "min_days": min_days,
    }

    if use_cache:
        with _effectiveness_cache["lock"]:
            _effectiveness_cache["data"] = result
            _effectiveness_cache["timestamp"] = now
    return result

# ─── Reset Data (Development only) ───────────────────────────────────────────
def reset_student_data() -> None:
    """Hapus semua data mahasiswa, prediksi, dan catatan konseling."""
    with _get_conn() as conn:
        conn.execute("DELETE FROM counseling_records")
        conn.execute("DELETE FROM predictions")
        conn.execute("DELETE FROM students")
        conn.execute(
            "DELETE FROM sqlite_sequence WHERE name IN ('students','predictions','counseling_records')"
        )
    invalidate_equity_cache()
    invalidate_effectiveness_cache()

# ─── Collective Wellbeing Early Warning (SDG 4.a) ────────────────────────────
def get_wellbeing_alerts(
    stress_threshold: float = 2.5,
    high_risk_spike_pct: float = 200.0,
    lookback_days: int = 7,
) -> Dict[str, Any]:
    """Hitungan alert wellbeing kolektif."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alerts: List[Dict[str, Any]] = []

    with _get_conn() as conn:
        stress_rows = conn.execute(
            """
            SELECT p.raw_features
            FROM   predictions p
            WHERE  p.id IN (
                SELECT MAX(id) FROM predictions GROUP BY student_id
            )
            """
        ).fetchall()

        stress_values: List[float] = []
        for row in stress_rows:
            rf = _safe_parse_raw_features(row["raw_features"])
            if rf is None:
                continue
            sl = rf.get("stress_levels")
            if sl is not None:
                stress_values.append(float(sl))

        avg_stress = sum(stress_values) / len(stress_values) if stress_values else 0.0

        from_this = f"datetime('now', '-{lookback_days} days')"
        from_prev = f"datetime('now', '-{lookback_days * 2} days')"

        count_this = conn.execute(
            f"""
            SELECT COUNT(*) FROM predictions
            WHERE  risk_level = 'tinggi'
              AND  predicted_at >= {from_this}
            """
        ).fetchone()[0]

        count_prev = conn.execute(
            f"""
            SELECT COUNT(*) FROM predictions
            WHERE  risk_level = 'tinggi'
              AND  predicted_at >= {from_prev}
              AND  predicted_at  < {from_this}
            """
        ).fetchone()[0]

    if avg_stress >= stress_threshold:
        alerts.append({
            "type": "high_stress",
            "severity": "warning",
            "message": (
                f"Rata-rata tingkat stres mahasiswa mencapai {avg_stress:.2f}/3.0 "
                f"(ambang batas: {stress_threshold}/3.0). Pertimbangkan program wellbeing kampus."
            ),
            "value": round(avg_stress, 2),
            "threshold": stress_threshold,
        })

    if count_prev > 0:
        spike_pct = (count_this - count_prev) / count_prev * 100
        if spike_pct >= high_risk_spike_pct:
            alerts.append({
                "type": "high_risk_spike",
                "severity": "danger",
                "message": (
                    f"Jumlah prediksi risiko tinggi naik {spike_pct:.0f}% "
                    f"({count_prev} → {count_this} kasus) dalam {lookback_days} hari terakhir."
                ),
                "value": round(spike_pct, 1),
                "threshold": high_risk_spike_pct,
            })
    elif count_prev == 0 and count_this >= 5:
        alerts.append({
            "type": "high_risk_spike",
            "severity": "warning",
            "message": (
                f"Terdeteksi {count_this} kasus risiko tinggi baru dalam "
                f"{lookback_days} hari (tidak ada data pembanding sebelumnya)."
            ),
            "value": count_this,
            "threshold": 5,
        })

    return {
        "alerts": alerts,
        "computed_at": now_str,
        "stats": {
            "avg_stress": round(avg_stress, 2),
            "n_stress_data": len(stress_values),
            "high_risk_this_period": count_this,
            "high_risk_prev_period": count_prev,
        },
    }

# ─── Teacher / Teaching Quality Analytics (SDG 4.c) ──────────────────────────
def get_teaching_analytics() -> Dict[str, Any]:
    """Agregasi data teaching_quality_rating."""
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT p.raw_features, p.risk_level, s.nim, s.nama
            FROM   predictions p
            JOIN   students    s ON s.id = p.student_id
            WHERE  p.id IN (
                SELECT MAX(id) FROM predictions GROUP BY student_id
            )
            """
        ).fetchall()

    ratings: List[float] = []
    low_rating_students: List[Dict] = []
    rating_dist: Dict[str, int] = {str(i): 0 for i in range(1, 11)}

    for row in rows:
        rf = _safe_parse_raw_features(row["raw_features"])
        if rf is None:
            continue
        r = rf.get("teaching_quality_rating")
        if r is not None:
            rv = float(r)
            ratings.append(rv)
            key = str(int(rv))
            if key in rating_dist:
                rating_dist[key] += 1
            if rv <= 4:
                low_rating_students.append({
                    "nim": row["nim"] or "-",
                    "nama": row["nama"] or "-",
                    "rating": rv,
                    "risk_level": row["risk_level"],
                })

    avg_rating = round(sum(ratings) / len(ratings), 2) if ratings else None
    high_risk_ratings = []
    low_risk_ratings = []
    with _get_conn() as conn:
        rows2 = conn.execute(
            """
            SELECT p.raw_features, p.risk_level
            FROM   predictions p
            WHERE  p.id IN (
                SELECT MAX(id) FROM predictions GROUP BY student_id
            )
            """
        ).fetchall()
    for row in rows2:
        rf = _safe_parse_raw_features(row["raw_features"])
        if rf is None:
            continue
        r = rf.get("teaching_quality_rating")
        if r is None:
            continue
        if row["risk_level"] == "tinggi":
            high_risk_ratings.append(float(r))
        else:
            low_risk_ratings.append(float(r))

    avg_high_risk = round(sum(high_risk_ratings)/len(high_risk_ratings), 2) if high_risk_ratings else None
    avg_low_risk  = round(sum(low_risk_ratings)/len(low_risk_ratings), 2) if low_risk_ratings else None

    if avg_rating is not None and avg_rating <= 5:
        training_recs = [
            "Active Learning & Student-Centered Teaching",
            "Digital Pedagogy untuk Generasi Z",
            "Strategi Asesmen & Umpan Balik Formatif",
            "Manajemen Kelas dan Motivasi Mahasiswa",
        ]
        compliance_note = f"Rating rata-rata {avg_rating}/10 di bawah standar. Diperlukan program peningkatan mutu dosen."
    elif avg_rating is not None and avg_rating <= 7:
        training_recs = [
            "Inovasi Metode Pengajaran",
            "Penggunaan Teknologi Pembelajaran",
        ]
        compliance_note = f"Rating rata-rata {avg_rating}/10 cukup. Lanjutkan program pengembangan berkelanjutan."
    else:
        training_recs = ["Berbagi Praktik Baik (Best Practice Sharing)"]
        compliance_note = f"Rating rata-rata {avg_rating}/10 baik. Pertahankan dan jadikan mentor bagi dosen lain."

    return {
        "avg_rating": avg_rating,
        "total_data_points": len(ratings),
        "rating_distribution": rating_dist,
        "low_rating_students": sorted(low_rating_students, key=lambda x: x["rating"])[:20],
        "avg_rating_by_risk": {
            "tinggi": avg_high_risk,
            "rendah_sedang": avg_low_risk,
        },
        "training_recommendations": training_recs,
        "sdg_4c_compliance_note": compliance_note,
        "computed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

# ─── Mentor / Peer Support (SDG 4.a) ─────────────────────────────────────────
def get_all_mentors(active_only: bool = True) -> List[Dict[str, Any]]:
    clause = "WHERE m.is_active = 1" if active_only else ""
    with _get_conn() as conn:
        rows = conn.execute(
            f"""
            SELECT m.*,
                   COUNT(ms.id) AS total_sessions,
                   SUM(CASE WHEN ms.status='done' THEN 1 ELSE 0 END) AS done_sessions
            FROM   mentors m
            LEFT   JOIN mentor_sessions ms ON ms.mentor_id = m.id
            {clause}
            GROUP  BY m.id
            ORDER  BY m.name
            """
        ).fetchall()
    return [dict(r) for r in rows]

def get_mentor_by_id(mentor_id: int) -> Optional[Dict[str, Any]]:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM mentors WHERE id = ?", (mentor_id,)
        ).fetchone()
    return dict(row) if row else None

def create_mentor(
    name: str,
    nim: str,
    major: Optional[str] = None,
    availability_json: str = "[]",
    max_mentees: int = 3,
) -> Optional[int]:
    try:
        with _get_conn() as conn:
            cur = conn.execute(
                """INSERT INTO mentors (name, nim, major, availability_json, max_mentees)
                   VALUES (?, ?, ?, ?, ?)""",
                (name.strip(), nim.strip(), major, availability_json, max_mentees),
            )
            return cur.lastrowid
    except sqlite3.IntegrityError:
        return None

def update_mentor(mentor_id: int, **kwargs) -> bool:
    allowed = {"name", "major", "availability_json", "max_mentees", "is_active"}
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return False
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    with _get_conn() as conn:
        cur = conn.execute(
            f"UPDATE mentors SET {set_clause} WHERE id = ?",
            list(updates.values()) + [mentor_id],
        )
        return cur.rowcount > 0

def match_mentor_for_student(student_id: int) -> Optional[Dict[str, Any]]:
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT m.*,
                   COUNT(CASE WHEN ms.status='scheduled' THEN 1 END) AS active_mentees
            FROM   mentors m
            LEFT   JOIN mentor_sessions ms ON ms.mentor_id = m.id
            WHERE  m.is_active = 1
              AND  m.id NOT IN (
                  SELECT mentor_id FROM mentor_sessions WHERE mentee_id = ?
              )
            GROUP  BY m.id
            HAVING active_mentees < m.max_mentees
            ORDER  BY active_mentees ASC
            LIMIT  1
            """,
            (student_id,),
        ).fetchone()
    return dict(rows) if rows else None

def create_mentor_session(
    mentor_id: int,
    mentee_id: int,
    session_date: str,
    notes: Optional[str] = None,
) -> int:
    with _get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO mentor_sessions (mentor_id, mentee_id, session_date, notes)
               VALUES (?, ?, ?, ?)""",
            (mentor_id, mentee_id, session_date, notes),
        )
        return cur.lastrowid

def update_mentor_session(session_id: int, **kwargs) -> bool:
    allowed = {"status", "notes", "rating", "session_date"}
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return False
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    with _get_conn() as conn:
        cur = conn.execute(
            f"UPDATE mentor_sessions SET {set_clause} WHERE id = ?",
            list(updates.values()) + [session_id],
        )
        return cur.rowcount > 0

def get_mentor_sessions(
    mentor_id: Optional[int] = None,
    mentee_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    clauses = []
    params: List = []
    if mentor_id is not None:
        clauses.append("ms.mentor_id = ?")
        params.append(mentor_id)
    if mentee_id is not None:
        clauses.append("ms.mentee_id = ?")
        params.append(mentee_id)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    with _get_conn() as conn:
        rows = conn.execute(
            f"""
            SELECT ms.*, m.name AS mentor_name, m.nim AS mentor_nim,
                   s.nim AS mentee_nim, s.nama AS mentee_nama
            FROM   mentor_sessions ms
            JOIN   mentors  m ON m.id = ms.mentor_id
            JOIN   students s ON s.id = ms.mentee_id
            {where}
            ORDER  BY ms.session_date DESC
            """,
            params,
        ).fetchall()
    return [dict(r) for r in rows]

# ─── Statistical Significance untuk Effectiveness (SDG 4.1) ──────────────────
def compute_statistical_significance(
    before_scores: List[float],
    after_scores: List[float],
) -> Dict[str, Any]:
    n = len(before_scores)
    if n < 3:
        return {
            "n": n,
            "p_value": None,
            "is_significant": None,
            "effect_size": None,
            "interpretation": "Data tidak cukup (minimal 3 pasang data).",
            "test_used": "none",
            "mean_before": None,
            "mean_after": None,
            "mean_difference": None,
        }

    mean_before = sum(before_scores) / n
    mean_after  = sum(after_scores)  / n
    mean_diff   = mean_before - mean_after

    differences = [b - a for b, a in zip(before_scores, after_scores)]
    mean_diffs = sum(differences) / n
    var_diffs  = sum((d - mean_diffs) ** 2 for d in differences) / (n - 1)
    std_diffs  = var_diffs ** 0.5

    cohen_d = mean_diffs / std_diffs if std_diffs > 0 else 0.0

    try:
        from scipy import stats as _stats
        t_stat, p_value = _stats.ttest_rel(before_scores, after_scores)
        test_used = "paired_t_test"
        p_val_rounded = round(float(p_value), 4)
        is_significant = p_val_rounded < 0.05
    except ImportError:
        import math
        se = std_diffs / math.sqrt(n)
        t_stat = mean_diffs / se if se > 0 else 0.0
        p_val_rounded = None
        is_significant = abs(t_stat) > 2.0
        test_used = "t_approximation"

    abs_d = abs(cohen_d)
    if abs_d >= 0.8:
        interpretation = "Efek besar — konseling berdampak signifikan pada penurunan risiko."
    elif abs_d >= 0.5:
        interpretation = "Efek sedang — konseling cukup berdampak."
    elif abs_d >= 0.2:
        interpretation = "Efek kecil — ada dampak, tapi perlu ditingkatkan."
    else:
        interpretation = "Efek sangat kecil atau tidak ada — pertimbangkan evaluasi metode konseling."

    return {
        "n": n,
        "p_value": p_val_rounded,
        "is_significant": is_significant,
        "effect_size": round(cohen_d, 3),
        "interpretation": interpretation,
        "test_used": test_used,
        "mean_before": round(mean_before * 100, 2),
        "mean_after":  round(mean_after  * 100, 2),
        "mean_difference": round(mean_diff * 100, 2),
    }

# ─── Recommendation Data Access (untuk recommender.py) ───────────────────────
def get_evaluated_interventions() -> List[Dict[str, Any]]:
    sql = """
        SELECT
            cr.id                AS counseling_id,
            cr.student_id,
            cr.intervention_type,
            cr.notes,
            cr.completed_at,
            p_before.raw_features AS raw_features_before,
            p_before.p_dropout    AS p_dropout_before,
            (
                SELECT p2.p_dropout
                FROM   predictions p2
                WHERE  p2.student_id   = cr.student_id
                  AND  p2.predicted_at > datetime(cr.completed_at, '+30 days')
                ORDER  BY p2.predicted_at DESC
                LIMIT  1
            ) AS p_dropout_after
        FROM   counseling_records cr
        JOIN   predictions p_before ON p_before.id = cr.prediction_id
        WHERE  cr.status       = 'done'
          AND  cr.completed_at IS NOT NULL
    """
    with _get_conn() as conn:
        rows = conn.execute(sql).fetchall()

    result = []
    for row in rows:
        if row["p_dropout_after"] is None:
            continue
        rf = _safe_parse_raw_features(row["raw_features_before"])
        if rf is None:
            continue
        result.append({
            "counseling_id": row["counseling_id"],
            "student_id": row["student_id"],
            "intervention_type": row["intervention_type"] or "umum",
            "notes": row["notes"] or "",
            "p_dropout_before": float(row["p_dropout_before"]),
            "p_dropout_after": float(row["p_dropout_after"]),
            "delta": float(row["p_dropout_before"]) - float(row["p_dropout_after"]),
            "features": rf,
        })
    return result

def get_latest_features_for_student(student_id: int) -> Optional[Dict[str, Any]]:
    with _get_conn() as conn:
        row = conn.execute(
            """SELECT raw_features, p_dropout, risk_level
               FROM predictions WHERE student_id = ?
               ORDER BY predicted_at DESC LIMIT 1""",
            (student_id,),
        ).fetchone()
    if row is None:
        return None
    rf = _safe_parse_raw_features(row["raw_features"])
    if rf is None:
        return None
    rf["p_dropout"] = float(row["p_dropout"])
    rf["risk_level"] = row["risk_level"]
    return rf

# ─── Infrastructure Survey (SDG 4.a) ─────────────────────────────────────────
def _ensure_infrastructure_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS infrastructure_reports (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            campus        TEXT    NOT NULL DEFAULT 'Kampus Utama',
            faculty       TEXT,
            reported_by   INTEGER REFERENCES students(id) ON DELETE SET NULL,
            electricity   INTEGER DEFAULT 0,
            clean_water   INTEGER DEFAULT 0,
            sanitation    INTEGER DEFAULT 0,
            separate_wc   INTEGER DEFAULT 0,
            computers     INTEGER DEFAULT 0,
            internet      INTEGER DEFAULT 0,
            disability_access INTEGER DEFAULT 0,
            safe_environment  INTEGER DEFAULT 0,
            notes         TEXT,
            created_at    TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_infra_campus ON infrastructure_reports(campus)"
    )

def _ensure_equity_columns(conn: sqlite3.Connection) -> None:
    student_cols = [r[1] for r in conn.execute("PRAGMA table_info(students)").fetchall()]
    extras = {
        "gender":           "TEXT",
        "disability_status": "INTEGER DEFAULT 0",
        "indigenous":       "INTEGER DEFAULT 0",
        "conflict_affected":"INTEGER DEFAULT 0",
    }
    for col, typedef in extras.items():
        if col not in student_cols:
            conn.execute(f"ALTER TABLE students ADD COLUMN {col} {typedef}")
            logger.info("Kolom '%s' ditambahkan ke tabel students", col)

def save_infrastructure_report(
    campus: str,
    faculty: Optional[str],
    reported_by: Optional[int],
    electricity: bool,
    clean_water: bool,
    sanitation: bool,
    separate_wc: bool,
    computers: bool,
    internet: bool,
    disability_access: bool,
    safe_environment: bool,
    notes: Optional[str] = None,
) -> int:
    with _get_conn() as conn:
        _ensure_infrastructure_table(conn)
        cur = conn.execute(
            """INSERT INTO infrastructure_reports
               (campus, faculty, reported_by, electricity, clean_water, sanitation,
                separate_wc, computers, internet, disability_access, safe_environment, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (campus, faculty, reported_by,
             int(electricity), int(clean_water), int(sanitation), int(separate_wc),
             int(computers), int(internet), int(disability_access),
             int(safe_environment), notes),
        )
        return cur.lastrowid

def get_infrastructure_summary() -> Dict[str, Any]:
    with _get_conn() as conn:
        _ensure_infrastructure_table(conn)
        rows = conn.execute(
            """SELECT campus, COUNT(*) AS total,
               AVG(electricity) AS electricity,
               AVG(clean_water) AS clean_water,
               AVG(sanitation) AS sanitation,
               AVG(separate_wc) AS separate_wc,
               AVG(computers) AS computers,
               AVG(internet) AS internet,
               AVG(disability_access) AS disability_access,
               AVG(safe_environment) AS safe_environment
               FROM infrastructure_reports
               GROUP BY campus
               ORDER BY campus"""
        ).fetchall()
        total_reports = conn.execute(
            "SELECT COUNT(*) FROM infrastructure_reports"
        ).fetchone()[0]

    FACILITIES = [
        "electricity", "clean_water", "sanitation", "separate_wc",
        "computers", "internet", "disability_access", "safe_environment",
    ]

    by_campus = []
    for row in rows:
        r = dict(row)
        campus_data = {"campus": r["campus"], "total_reports": r["total"]}
        for f in FACILITIES:
            campus_data[f] = round(float(r[f] or 0), 3)
        by_campus.append(campus_data)

    overall: Dict[str, Any] = {"total_reports": total_reports}
    if by_campus:
        for f in FACILITIES:
            vals = [c[f] for c in by_campus]
            overall[f] = round(sum(vals) / len(vals), 3)
    else:
        for f in FACILITIES:
            overall[f] = None

    return {
        "by_campus": by_campus,
        "overall": overall,
        "total_reports": total_reports,
        "computed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

# ─── Equity Dashboard Extended (SDG 4.5) ─────────────────────────────────────
def update_student_equity_info(
    student_id: int,
    gender: Optional[str] = None,
    disability_status: Optional[bool] = None,
    indigenous: Optional[bool] = None,
    conflict_affected: Optional[bool] = None,
) -> bool:
    with _get_conn() as conn:
        _ensure_equity_columns(conn)
    updates: Dict[str, Any] = {}
    if gender is not None:
        updates["gender"] = gender.upper()[:1] if gender else None
    if disability_status is not None:
        updates["disability_status"] = int(disability_status)
    if indigenous is not None:
        updates["indigenous"] = int(indigenous)
    if conflict_affected is not None:
        updates["conflict_affected"] = int(conflict_affected)
    if not updates:
        return False
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    with _get_conn() as conn:
        cur = conn.execute(
            f"UPDATE students SET {set_clause} WHERE id = ?",
            list(updates.values()) + [student_id],
        )
        changed = cur.rowcount > 0
    # Invalidate equity cache because gender/disability etc. changed
    if changed:
        invalidate_equity_cache()
    return changed

def get_equity_data_extended() -> Dict[str, Any]:
    with _get_conn() as conn:
        _ensure_equity_columns(conn)
        rows = conn.execute(
            """SELECT s.gender, s.disability_status, s.indigenous,
                      s.conflict_affected, p.p_dropout, p.risk_level
               FROM   students s
               JOIN   predictions p ON p.id = (
                   SELECT MAX(id) FROM predictions WHERE student_id = s.id
               )"""
        ).fetchall()

    if not rows:
        return {
            "gender": {}, "disability": {}, "indigenous": {}, "conflict": {},
            "parity_indices": {}, "alerts": [],
            "total_with_gender_data": 0, "total_students": 0,
            "computed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _aggregate(key_fn, data):
        groups: Dict[str, Dict] = {}
        for row in data:
            k = key_fn(row)
            if k not in groups:
                groups[k] = {"count": 0, "p_dropout_sum": 0.0, "high_risk": 0}
            groups[k]["count"] += 1
            groups[k]["p_dropout_sum"] += float(row["p_dropout"])
            if row["risk_level"] == "tinggi":
                groups[k]["high_risk"] += 1
        result = {}
        for k, v in groups.items():
            n = v["count"]
            result[k] = {
                "count": n,
                "avg_p_dropout": round(v["p_dropout_sum"] / n, 4),
                "high_risk_pct": round(v["high_risk"] / n * 100, 1),
            }
        return result

    gender_agg  = _aggregate(lambda r: r["gender"] or "Tidak diisi", rows)
    disab_agg   = _aggregate(lambda r: "Disabilitas" if r["disability_status"] else "Non-disabilitas", rows)
    indig_agg   = _aggregate(lambda r: "Masyarakat Adat" if r["indigenous"] else "Bukan Masyarakat Adat", rows)
    conflict_agg= _aggregate(lambda r: "Daerah Konflik" if r["conflict_affected"] else "Daerah Normal", rows)

    parity: Dict[str, Any] = {}
    gl = gender_agg.get("L")
    gp = gender_agg.get("P")
    if gl and gp and gl["avg_p_dropout"] > 0:
        pi = round(gp["avg_p_dropout"] / gl["avg_p_dropout"], 3)
        parity["gender"] = {
            "value": pi,
            "interpretation": (
                "Setara" if 0.97 <= pi <= 1.03
                else ("Perempuan lebih berisiko" if pi > 1.03 else "Laki-laki lebih berisiko")
            ),
            "target": "0.97 – 1.03",
        }

    nd = disab_agg.get("Non-disabilitas")
    d  = disab_agg.get("Disabilitas")
    if nd and d and nd["avg_p_dropout"] > 0:
        pi_d = round(d["avg_p_dropout"] / nd["avg_p_dropout"], 3)
        parity["disability"] = {
            "value": pi_d,
            "interpretation": "Akses setara" if 0.9 <= pi_d <= 1.1 else "Ada kesenjangan akses",
            "target": "0.90 – 1.10",
        }

    alerts = []
    gpi = parity.get("gender", {})
    if gpi and (gpi["value"] < 0.97 or gpi["value"] > 1.03):
        alerts.append({
            "type": "gender_parity",
            "message": f"Indeks paritas gender = {gpi['value']} ({gpi['interpretation']}). Target SDG: 0.97–1.03.",
            "severity": "warning",
        })

    total_with_gender = sum(1 for r in rows if r["gender"] in ("L", "P"))

    return {
        "gender": gender_agg,
        "disability": disab_agg,
        "indigenous": indig_agg,
        "conflict": conflict_agg,
        "parity_indices": parity,
        "alerts": alerts,
        "total_with_gender_data": total_with_gender,
        "total_students": len(rows),
        "computed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

# ─── Wellbeing Time-Series (SDG 4.a) ─────────────────────────────────────────
def get_wellbeing_trend(weeks: int = 8) -> Dict[str, Any]:
    if weeks < 1 or weeks > 52:
        weeks = 8

    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
                strftime('%Y-W%W', predicted_at) AS week_label,
                AVG(CAST(json_extract(raw_features, '$.stress_levels') AS REAL)) AS avg_stress,
                COUNT(CASE WHEN risk_level='tinggi' THEN 1 END) AS high_risk_count,
                COUNT(*) AS total
            FROM predictions
            WHERE predicted_at >= datetime('now', ?)
            GROUP BY week_label
            ORDER BY week_label ASC
            """,
            (f"-{weeks * 7} days",),
        ).fetchall()

    labels = []
    avg_stress_series = []
    high_risk_series = []

    for row in rows:
        labels.append(row["week_label"] or "?")
        avg_stress_series.append(round(float(row["avg_stress"] or 0), 2))
        high_risk_series.append(int(row["high_risk_count"] or 0))

    return {
        "labels": labels,
        "avg_stress": avg_stress_series,
        "high_risk_count": high_risk_series,
        "weeks_requested": weeks,
        "computed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

# ─── Intervention Effectiveness Trend (SDG 4.1) ───────────────────────────────
def get_intervention_trend() -> Dict[str, Any]:
    with _get_conn() as conn:
        monthly_rows = conn.execute(
            """
            SELECT
                strftime('%Y-%m', cr.completed_at) AS month_label,
                COUNT(*) AS n_cases,
                AVG(p_before.p_dropout - COALESCE((
                    SELECT p2.p_dropout
                    FROM   predictions p2
                    WHERE  p2.student_id = cr.student_id
                      AND  p2.predicted_at > datetime(cr.completed_at, '+30 days')
                    ORDER  BY p2.predicted_at DESC LIMIT 1
                ), p_before.p_dropout)) AS avg_delta
            FROM   counseling_records cr
            JOIN   predictions p_before ON p_before.id = cr.prediction_id
            WHERE  cr.status = 'done'
              AND  cr.completed_at IS NOT NULL
              AND  cr.completed_at >= datetime('now', '-12 months')
            GROUP  BY month_label
            ORDER  BY month_label ASC
            """
        ).fetchall()

        type_rows = conn.execute(
            """
            SELECT
                cr.intervention_type,
                p_before.risk_level,
                COUNT(*) AS n_cases,
                AVG(p_before.p_dropout - COALESCE((
                    SELECT p2.p_dropout
                    FROM   predictions p2
                    WHERE  p2.student_id = cr.student_id
                      AND  p2.predicted_at > datetime(cr.completed_at, '+30 days')
                    ORDER  BY p2.predicted_at DESC LIMIT 1
                ), p_before.p_dropout)) AS avg_delta
            FROM   counseling_records cr
            JOIN   predictions p_before ON p_before.id = cr.prediction_id
            WHERE  cr.status = 'done'
              AND  cr.completed_at IS NOT NULL
            GROUP  BY cr.intervention_type, p_before.risk_level
            """
        ).fetchall()

    monthly = {
        "labels": [r["month_label"] for r in monthly_rows],
        "avg_delta_pct": [round(float(r["avg_delta"] or 0) * 100, 2) for r in monthly_rows],
        "n_cases": [int(r["n_cases"]) for r in monthly_rows],
    }

    heatmap: Dict[str, Dict[str, Any]] = {}
    for row in type_rows:
        itype = row["intervention_type"] or "umum"
        rlevel = row["risk_level"] or "sedang"
        if itype not in heatmap:
            heatmap[itype] = {}
        heatmap[itype][rlevel] = {
            "avg_delta_pct": round(float(row["avg_delta"] or 0) * 100, 2),
            "n_cases": int(row["n_cases"]),
        }

    return {
        "monthly_trend": monthly,
        "heatmap_by_type_and_risk": heatmap,
        "computed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

# ─── Mentor Effectiveness (SDG 4.a) ──────────────────────────────────────────
def get_mentor_effectiveness() -> Dict[str, Any]:
    with _get_conn() as conn:
        mentored = conn.execute(
            """
            SELECT DISTINCT ms.mentee_id
            FROM mentor_sessions ms
            WHERE ms.status = 'done'
            """
        ).fetchall()
        mentored_ids = {r["mentee_id"] for r in mentored}

        if not mentored_ids:
            return {
                "mentored_count": 0,
                "avg_delta_pct_mentored": None,
                "avg_delta_pct_unmentored": None,
                "parity_note": "Belum ada sesi mentoring yang selesai.",
                "mentor_stats": [],
                "computed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

        all_deltas = conn.execute(
            """
            SELECT s.id AS student_id,
                   first_p.p_dropout AS p_first,
                   last_p.p_dropout  AS p_last
            FROM   students s
            JOIN   predictions first_p ON first_p.id = (
                SELECT MIN(id) FROM predictions WHERE student_id = s.id
            )
            JOIN   predictions last_p ON last_p.id = (
                SELECT MAX(id) FROM predictions WHERE student_id = s.id
            )
            WHERE  first_p.id != last_p.id
            """
        ).fetchall()

    mentored_deltas = []
    unmentored_deltas = []
    for row in all_deltas:
        delta = (float(row["p_first"]) - float(row["p_last"])) * 100
        if row["student_id"] in mentored_ids:
            mentored_deltas.append(delta)
        else:
            unmentored_deltas.append(delta)

    avg_m = round(sum(mentored_deltas) / len(mentored_deltas), 2) if mentored_deltas else None
    avg_u = round(sum(unmentored_deltas) / len(unmentored_deltas), 2) if unmentored_deltas else None

    note = ""
    if avg_m is not None and avg_u is not None:
        if avg_m > avg_u:
            note = f"Mahasiswa yang didampingi mentor mengalami penurunan risiko rata-rata {avg_m:.1f}pp, lebih baik dari yang tidak ({avg_u:.1f}pp)."
        elif avg_m < avg_u:
            note = f"Mahasiswa tidak didampingi mengalami penurunan lebih besar ({avg_u:.1f}pp vs {avg_m:.1f}pp). Evaluasi kualitas sesi mentoring."
        else:
            note = "Tidak ada perbedaan signifikan antara kelompok mentored dan non-mentored."

    with _get_conn() as conn:
        mentor_rows = conn.execute(
            """
            SELECT m.id, m.name, m.nim,
                   COUNT(DISTINCT ms.mentee_id) AS total_mentees,
                   AVG(ms.rating) AS avg_rating,
                   SUM(CASE WHEN ms.status='done' THEN 1 ELSE 0 END) AS done_sessions
            FROM   mentors m
            LEFT   JOIN mentor_sessions ms ON ms.mentor_id = m.id
            WHERE  m.is_active = 1
            GROUP  BY m.id
            ORDER  BY avg_rating DESC NULLS LAST
            """
        ).fetchall()

    mentor_stats = []
    for row in mentor_rows:
        mentor_stats.append({
            "name": row["name"],
            "nim": row["nim"],
            "total_mentees": row["total_mentees"],
            "avg_rating": round(float(row["avg_rating"]), 2) if row["avg_rating"] else None,
            "done_sessions": row["done_sessions"],
        })

    return {
        "mentored_count": len(mentored_ids),
        "unmentored_sample": len(unmentored_deltas),
        "avg_delta_pct_mentored": avg_m,
        "avg_delta_pct_unmentored": avg_u,
        "parity_note": note,
        "mentor_stats": mentor_stats,
        "computed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }