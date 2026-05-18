import os
import warnings
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    # ── Database & Model ───────────────────────────────────────────────────────
    DB_PATH = os.path.join(BASE_DIR, "data", "staylearn.db")
    MODEL_PATH = os.path.join(BASE_DIR, "models", "staylearn_model.joblib")
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

    # ── Environment ────────────────────────────────────────────────────────────
    ENVIRONMENT = os.environ.get("FLASK_ENV", "production")

    # ── Debug mode ─────────────────────────────────────────────────────────────
    DEBUG = os.environ.get("FLASK_ENV", "production") != "production"

    # ── Secret Key ─────────────────────────────────────────────────────────────
    SECRET_KEY = os.environ.get("SECRET_KEY")
    if not SECRET_KEY:
        if ENVIRONMENT == "production":
            warnings.warn(
                "SECRET_KEY tidak diatur via environment variable. "
                "Aplikasi TIDAK AKAN BERJALAN di mode produksi tanpa SECRET_KEY. "
                "Setel: export SECRET_KEY='<string-acak-minimal-32-karakter>'",
                RuntimeWarning,
                stacklevel=2,
            )
        import secrets as _secrets
        SECRET_KEY = _secrets.token_hex(32)

    # ── Request limits ─────────────────────────────────────────────────────────
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024   # 16 MB max upload
    MAX_BATCH_ROWS = 10_000
    CACHE_MAX_SIZE = 128

    # ── Rate Limiting ──────────────────────────────────────────────────────────
    RATELIMIT_DEFAULT = "100 per minute"
    RATELIMIT_STORAGE_URL = os.environ.get("RATELIMIT_STORAGE_URL", "memory://")

    # ── CSRF ───────────────────────────────────────────────────────────────────
    CSRF_TIME_LIMIT = 3600   # Token CSRF berlaku 1 jam

    # ── Session Cookie Security ─────────────────────────────────────────────────
    SESSION_COOKIE_SECURE = (ENVIRONMENT == "production")
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"

    # ── Career Advice (SDG 4.4) ────────────────────────────────────────────────
    ENABLE_CAREER_ADVICE = os.environ.get("ENABLE_CAREER_ADVICE", "True").lower() == "true"

    # ── Collective Wellbeing Early Warning (SDG 4.a) ───────────────────────────
    # Rata-rata stress level (skala 1–3) dianggap berbahaya jika melebihi batas ini
    WELLBEING_STRESS_THRESHOLD = 2.5
    # Persentase kenaikan jumlah mahasiswa risiko tinggi dalam 7 hari dibanding 7 hari sebelumnya
    WELLBEING_HIGH_RISK_SPIKE_THRESHOLD = 200   # 200% = naik 2x lipat
    WELLBEING_LOOKBACK_DAYS = 7