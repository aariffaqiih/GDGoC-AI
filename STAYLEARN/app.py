import database as db
from counselor_routes import counselor_bp
import io
import os
import secrets
import logging
import hashlib
import uuid
from logging.handlers import RotatingFileHandler
from functools import wraps
import pandas as pd
from flask import Flask, g, jsonify, render_template, request, send_file, session, redirect, url_for
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_swagger_ui import get_swaggerui_blueprint
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_wtf.csrf import CSRFProtect
from config import Config
from predictor import Predictor
from chatbot import get_chatbot
from datetime import datetime, timedelta
import re
import sqlite3

# --- AI Engine (LLM-powered: Groq/HuggingFace) ---
from ai_engine import generate_career_advice, generate_recommendation, chat as ai_chat, reset_chat_session, get_ai_status

# --- Logging dengan request ID ---
class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.request_id = g.get("request_id", "-")
        except RuntimeError:
            record.request_id = "-"
        return True

log_formatter = logging.Formatter(
    "%(asctime)s %(levelname)s [%(request_id)s] %(message)s"
)
log_handler = RotatingFileHandler("app.log", maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
log_handler.setFormatter(log_formatter)
log_handler.addFilter(RequestIdFilter())
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)

def add_request_id() -> str:
    request_id = str(uuid.uuid4())
    g.request_id = request_id
    return request_id

def log_request(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        add_request_id()
        logger.info("%s %s accessed", request.method, request.path)
        return func(*args, **kwargs)
    return wrapper

# --- Factory ---
def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)

    # ── Guard: tolak startup produksi tanpa SECRET_KEY yang eksplisit ──────────
    if app.config.get("ENVIRONMENT") == "production" and not os.environ.get("SECRET_KEY"):
        raise RuntimeError(
            "SECRET_KEY WAJIB diatur via environment variable di lingkungan produksi. "
            "Jalankan: export SECRET_KEY='<string-acak-panjang>'"
        )

    # ── Guard: debug mode TIDAK boleh aktif di produksi ────────────────────────
    if app.config.get("ENVIRONMENT") == "production" and app.debug:
        logger.critical("DEBUG MODE AKTIF DI LINGKUNGAN PRODUKSI! Matikan FLASK_DEBUG.")
        raise RuntimeError("Debug mode tidak boleh aktif di produksi.")

    # Konfigurasi keamanan cookie sesi
    app.config['SESSION_COOKIE_SECURE'] = app.config['SESSION_COOKIE_SECURE']
    app.config['SESSION_COOKIE_HTTPONLY'] = app.config['SESSION_COOKIE_HTTPONLY']
    app.config['SESSION_COOKIE_SAMESITE'] = app.config['SESSION_COOKIE_SAMESITE']
    # Session timeout untuk riwayat mahasiswa (30 menit)
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)

    # Middleware untuk mendapatkan IP asli di belakang proxy
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)

    # CSRF Protection
    csrf = CSRFProtect()
    csrf.init_app(app)
    app.config['WTF_CSRF_HEADERS'] = ['X-CSRFToken']

    # Inisialisasi database
    try:
        db.init_db(app.config["DB_PATH"])
        if not db.verify_connection():
            raise RuntimeError("Database tidak dapat diakses setelah inisialisasi")
        logger.info("Database berhasil diinisialisasi dan diverifikasi")
    except Exception as e:
        logger.critical("Kesalahan fatal pada database: %s", e)
        raise RuntimeError(f"Gagal menginisialisasi database: {e}") from e

    app.register_blueprint(counselor_bp)

    # --- Flask-Login ---
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = "login"
    login_manager.login_message = "Silakan login untuk mengakses halaman konselor."

    @login_manager.user_loader
    def load_user(user_id):
        return db.get_user_by_id(int(user_id))

    # --- Rate Limiting ---
    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=[app.config["RATELIMIT_DEFAULT"]],
        storage_uri=app.config["RATELIMIT_STORAGE_URL"],
    )

    predictor = Predictor(
        app.config["MODEL_PATH"],
        max_batch_rows=app.config["MAX_BATCH_ROWS"],
    )
    # Simpan referensi predictor agar bisa diakses di blueprint routes (XAI)
    app.config["_predictor_ref"] = predictor

    # AI Engine (Groq/HuggingFace) - no initialization needed, uses API calls
    logger.info("AI Engine initialized (Groq/HF backends)")

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    # --- Swagger ---
    SWAGGER_URL = "/api/docs"
    API_URL = "/swagger.json"
    swaggerui_blueprint = get_swaggerui_blueprint(
        SWAGGER_URL,
        API_URL,
        config={"app_name": "StayLearn API"},
    )
    app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

    @app.route("/swagger.json")
    def swagger_json():
        spec = {
            "swagger": "2.0",
            "info": {
                "title": "StayLearn API",
                "description": "API for predicting student dropout risk",
                "version": "1.0.0",
            },
            "host": request.host,
            "basePath": "/api",
            "schemes": ["http", "https"],
            "paths": {
                "/predict": {
                    "post": {
                        "summary": "Predict dropout risk for a single student",
                        "consumes": ["application/json"],
                        "produces": ["application/json"],
                        "parameters": [
                            {
                                "in": "body",
                                "name": "body",
                                "required": True,
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "location_type": {
                                            "type": "string",
                                            "enum": ["Urban", "Rural", "Semi-urban"],
                                        },
                                        "family_income": {"type": "number"},
                                        "financial_aid_status": {
                                            "type": "integer",
                                            "enum": [0, 1, 2],
                                        },
                                        "distance_to_institute": {"type": "number"},
                                        "internet_connectivity_issues": {
                                            "type": "integer",
                                            "enum": [0, 1, 2],
                                        },
                                        "motivation_score": {
                                            "type": "integer",
                                            "minimum": 1,
                                            "maximum": 10,
                                        },
                                        "career_alignment": {
                                            "type": "integer",
                                            "enum": [1, 2, 3],
                                        },
                                        "stress_levels": {
                                            "type": "integer",
                                            "enum": [1, 2, 3],
                                        },
                                        "family_support": {
                                            "type": "integer",
                                            "enum": [1, 2, 3],
                                        },
                                        "attendance_rate": {
                                            "type": "number",
                                            "minimum": 10,
                                            "maximum": 100,
                                        },
                                        "test_scores_avg": {
                                            "type": "number",
                                            "minimum": 35,
                                            "maximum": 100,
                                        },
                                        "backlogs": {
                                            "type": "integer",
                                            "minimum": 0,
                                            "maximum": 9,
                                        },
                                        "teaching_quality_rating": {
                                            "type": "integer",
                                            "minimum": 1,
                                            "maximum": 10,
                                        },
                                    },
                                    "required": [
                                        "location_type",
                                        "family_income",
                                        "financial_aid_status",
                                        "distance_to_institute",
                                        "internet_connectivity_issues",
                                        "motivation_score",
                                        "career_alignment",
                                        "stress_levels",
                                        "family_support",
                                        "attendance_rate",
                                        "test_scores_avg",
                                        "backlogs",
                                        "teaching_quality_rating",
                                    ],
                                },
                            }
                        ],
                        "responses": {
                            "200": {"description": "Successful prediction"},
                            "400": {"description": "Invalid input"},
                            "422": {"description": "Validation error"},
                            "500": {"description": "Internal server error"},
                        },
                    }
                },
                "/batch": {
                    "post": {
                        "summary": "Predict dropout risk for multiple students (CSV upload)",
                        "consumes": ["multipart/form-data"],
                        "produces": ["application/json"],
                        "parameters": [
                            {
                                "in": "formData",
                                "name": "file",
                                "type": "file",
                                "required": True,
                                "description": "CSV file containing student data",
                            }
                        ],
                        "responses": {
                            "200": {
                                "description": "Batch prediction results (partial results if some rows fail)"
                            },
                            "400": {"description": "Invalid file"},
                            "413": {"description": "File too large"},
                            "422": {"description": "Validation error"},
                        },
                    }
                },
                "/random": {
                    "get": {
                        "summary": "Generate random student data for testing",
                        "responses": {"200": {"description": "Random data object"}},
                    }
                },
                "/template": {
                    "get": {
                        "summary": "Download CSV template for batch prediction",
                        "produces": ["text/csv"],
                        "responses": {
                            "200": {"description": "CSV file"},
                            "404": {"description": "Template not found"},
                        },
                    }
                },
                "/health": {
                    "get": {
                        "summary": "Health check, confirms server and model are operational",
                        "produces": ["application/json"],
                        "responses": {
                            "200": {"description": "Service is healthy"},
                            "503": {"description": "Service is unhealthy"},
                        },
                    }
                },
            },
        }
        return jsonify(spec)

    # ── CSP Nonce: dibangkitkan per-request ──────────────────────────────────
    @app.before_request
    def generate_csp_nonce():
        g.csp_nonce = secrets.token_urlsafe(16)

    @app.context_processor
    def inject_csp_nonce():
        return dict(csp_nonce=g.get("csp_nonce", ""))

    _is_dev = os.environ.get("FLASK_ENV", "production") != "production"

    @app.context_processor
    def inject_debug():
        return dict(is_debug=_is_dev)

    # ─── Database connection per request (thread‑safe) ────────────────────────
    @app.before_request
    def before_request_db():
        """Buka koneksi database sebelum setiap request, simpan di g.db."""
        g.db = sqlite3.connect(app.config["DB_PATH"])
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
        # Mode WAL sudah di-set di schema, tapi pastikan
        g.db.execute("PRAGMA journal_mode = WAL")

    @app.teardown_appcontext
    def teardown_db(exception=None):
        """Tutup koneksi database setelah request selesai, commit jika sukses."""
        db_conn = g.pop('db', None)
        if db_conn is not None:
            if exception is None:
                db_conn.commit()
            else:
                db_conn.rollback()
            db_conn.close()

    # --- Halaman publik ---
    @app.route("/")
    @log_request
    def index():
        return render_template("index.html")

    @app.route("/batch")
    @log_request
    def batch():
        return render_template("batch.html")

    # ─── Halaman Riwayat Mahasiswa (Verifikasi NIM + Tanggal Lahir) ───────────
    @app.route("/riwayat", methods=["GET", "POST"])
    @limiter.limit("10 per minute", methods=["POST"])
    def riwayat_verifikasi():
        """Halaman verifikasi akses riwayat."""
        if request.method == "POST":
            nim = request.form.get("nim", "").strip()
            birth_date = request.form.get("birth_date", "").strip()

            if not nim or not birth_date:
                return render_template("riwayat_verifikasi.html", error="NIM dan tanggal lahir wajib diisi.")

            # Validasi format tanggal lahir YYYY-MM-DD
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", birth_date):
                return render_template("riwayat_verifikasi.html", error="Format tanggal lahir harus YYYY-MM-DD.")

            # Verifikasi ke database
            if db.verify_student_birth_date(nim, birth_date):
                # Buat session sementara
                session.permanent = True
                session["riwayat_authorized"] = True
                session["riwayat_nim"] = nim
                # Redirect ke halaman grafik
                return redirect(url_for("riwayat_tampil", nim=nim))
            else:
                # Pesan generik untuk mencegah enumerasi
                return render_template("riwayat_verifikasi.html", error="NIM atau tanggal lahir tidak valid.")

        return render_template("riwayat_verifikasi.html")

    @app.route("/riwayat/<nim>")
    def riwayat_tampil(nim):
        """Halaman tampilan grafik riwayat, memerlukan session terverifikasi."""
        if not session.get("riwayat_authorized") or session.get("riwayat_nim") != nim:
            # Redirect ke halaman verifikasi dengan pesan
            flash("Silakan verifikasi NIM dan tanggal lahir Anda terlebih dahulu.", "warning")
            return redirect(url_for("riwayat_verifikasi"))
        # Ambil data mahasiswa untuk ditampilkan namanya
        student = db.get_student_by_nim(nim)
        if not student:
            return render_template("404.html"), 404
        return render_template("riwayat.html", nim=nim, student=student)

    # ─── API Riwayat (JSON untuk grafik) ─────────────────────────────────────
    @app.route("/api/riwayat/<nim>")
    @limiter.limit("30 per minute")
    def api_riwayat(nim):
        """Endpoint JSON untuk data time-series p_dropout."""
        # Verifikasi session
        if not session.get("riwayat_authorized") or session.get("riwayat_nim") != nim:
            return jsonify({"status": "error", "message": "Unauthorized"}), 401

        predictions = db.get_prediction_history_by_nim(nim)
        if not predictions:
            return jsonify({"status": "success", "data": []})

        # Format untuk Chart.js: labels (tanggal) dan values (p_dropout persen)
        data = {
            "labels": [],
            "values": [],
            "details": []  # optional untuk tooltip
        }
        for p in predictions:
            # Format tanggal: DD-MM-YYYY HH:MM
            try:
                dt = datetime.strptime(p["predicted_at"], "%Y-%m-%d %H:%M:%S")
                label = dt.strftime("%d-%m-%Y %H:%M")
            except:
                label = p["predicted_at"]
            data["labels"].append(label)
            data["values"].append(round(p["p_dropout"] * 100, 1))  # persen
            data["details"].append({
                "risk_level": p["risk_level"],
                "trajectory": p["trajectory"],
                "source": p["source"]
            })
        return jsonify({"status": "success", "data": data})

    # --- Autentikasi ---
    @app.route("/login", methods=["GET", "POST"])
    @limiter.limit("5 per minute", methods=["POST"])
    def login():
        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")

            if not username or not password:
                return render_template("login.html", error="Username dan password wajib diisi")

            user = db.get_user_by_username(username)
            if user and user.check_password(password):
                login_user(user)
                next_page = request.args.get("next")
                if next_page and next_page.startswith("/") and not next_page.startswith("//"):
                    return redirect(next_page)
                return redirect(url_for("counselor.analytics"))
            else:
                logger.warning("Gagal login untuk username: %s dari IP: %s", username, request.remote_addr)
                return render_template("login.html", error="Username atau password salah")
        return render_template("login.html")

    @app.route("/logout", methods=["POST"])
    @login_required
    def logout():
        logout_user()
        return redirect(url_for("index"))

    # --- API ---
    @app.route("/api/health")
    def health():
        try:
            model_ok = predictor.model is not None
            db_ok = db.verify_connection()
            status_code = 200 if (model_ok and db_ok) else 503
            from sklearn import __version__ as sklearn_ver
            return (
                jsonify(
                    {
                        "status": "ok" if (model_ok and db_ok) else "degraded",
                        "model_loaded": model_ok,
                        "database_ok": db_ok,
                        "sklearn_version": sklearn_ver,
                    }
                ),
                status_code,
            )
        except Exception as exc:
            logger.exception("Health check failed: %s", exc)
            return jsonify({"status": "error", "model_loaded": False, "database_ok": False}), 503

    @app.route("/api/predict", methods=["POST"])
    @limiter.limit("10 per minute")
    @log_request
    def predict():
        payload = request.get_json(silent=True)
        if payload is None:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Format permintaan tidak valid. Kirim data dalam format JSON.",
                    }
                ),
                400,
            )
        try:
            result = predictor.predict_single(payload)
            nim = payload.get("nim")
            nama = payload.get("nama")
            birth_date = payload.get("birth_date")  # optional, untuk verifikasi riwayat
            if nim:
                try:
                    # Validasi birth_date jika diberikan
                    if birth_date and not re.match(r"^\d{4}-\d{2}-\d{2}$", birth_date):
                        birth_date = None  # ignore invalid format
                    student_id = db.upsert_student(str(nim), nama, birth_date)
                    db.save_prediction(
                        student_id=student_id,
                        risk_level=result["risk_level"],
                        p_dropout=result["p_dropout"] / 100,
                        raw_features={k: payload[k] for k in payload if k not in ("nim", "nama", "birth_date")},
                        source="individual",
                    )
                except Exception as exc:
                    logger.warning("Gagal menyimpan prediksi ke DB: %s", exc)

            # --- AI Career Advice (Groq / HuggingFace) ---
            try:
                clean_features = {k: payload[k] for k in payload if k not in ("nim", "nama", "birth_date")}
                result["career_advice"] = generate_career_advice(
                    features=clean_features,
                    risk_level=result["risk_level"],
                )
            except Exception as exc:
                logger.exception("Career advice generation failed: %s", exc)
                result["career_advice"] = None

            # --- AI Recommendation (Groq / HuggingFace) ---
            try:
                clean_features = {k: payload[k] for k in payload if k not in ("nim", "nama", "birth_date")}
                ai_rec = generate_recommendation(
                    features=clean_features,
                    risk_level=result["risk_level"],
                    concerns=result.get("concerns", []),
                    strengths=result.get("strengths", []),
                    p_dropout=result["p_dropout"],
                )
                if ai_rec:
                    result["recommendation"] = ai_rec
            except Exception as exc:
                logger.warning("AI recommendation failed (keeping rule-based): %s", exc)

            # --- Add XAI explanation (SHAP / koefisien) ---
            try:
                from xai import explain_prediction
                clean_features = {k: payload[k] for k in payload if k not in ("nim", "nama", "birth_date")}
                result["xai"] = explain_prediction(
                    predictor.model,
                    clean_features,
                    result["p_dropout"] / 100,
                )
            except Exception as exc:
                logger.warning("XAI explanation failed (non-fatal): %s", exc)
                result["xai"] = None

            # --- Auto-invalidate recommender after new prediction ---
            try:
                from recommender import invalidate_recommender_cache
                invalidate_recommender_cache()
            except Exception:
                pass

            logger.info(
                "Prediksi individu berhasil, risk_level=%s p_stay=%.1f",
                result.get("risk_level"),
                result.get("p_stay", 0),
            )
            return jsonify({"status": "success", "result": result})
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("Kesalahan validasi pada prediksi individu: %s", exc)
            return jsonify({"status": "error", "message": "Data tidak valid"}), 422
        except Exception as exc:
            logger.exception("Kesalahan tak terduga pada prediksi individu: %s", exc)
            return (
                jsonify({"status": "error", "message": "Terjadi kesalahan internal server"}),
                500,
            )

    @app.route("/api/random")
    @limiter.limit("30 per minute")
    @log_request
    def random_data():
        try:
            data = predictor.generate_random()
            logger.info("Random data generated")
            return jsonify(data)
        except Exception as exc:
            logger.exception("Gagal generate random: %s", exc)
            return (
                jsonify({"status": "error", "message": "Gagal generate data random"}),
                500,
            )

    def get_file_hash(file_bytes: bytes) -> str:
        return hashlib.sha256(file_bytes).hexdigest()

    @app.route("/api/batch", methods=["POST"])
    @limiter.limit("5 per minute")
    @log_request
    def batch_predict():
        if "file" not in request.files:
            return (
                jsonify({"status": "error", "message": "Tidak ada file yang dikirim"}),
                400,
            )
        file = request.files["file"]
        if not file.filename:
            return (
                jsonify({"status": "error", "message": "Tidak ada file yang dipilih"}),
                400,
            )
        if not file.filename.lower().endswith(".csv"):
            return (
                jsonify(
                    {"status": "error", "message": "File harus berformat CSV (.csv)"}
                ),
                400,
            )
        content_type = file.mimetype
        BLOCKED_MIME = {"text/html", "application/javascript", "application/json",
                        "application/xml", "image/", "audio/", "video/"}
        if any(content_type.startswith(b) for b in BLOCKED_MIME):
            return (
                jsonify({"status": "error", "message": "Tipe file tidak diizinkan."}),
                400,
            )
        try:
            content = file.read()
        except Exception as exc:
            logger.error("Gagal membaca file: %s", exc)
            return jsonify({"status": "error", "message": "Gagal membaca file"}), 400

        file_hash = get_file_hash(content)
        try:
            try:
                decoded = content.decode("utf-8-sig")
            except UnicodeDecodeError:
                try:
                    decoded = content.decode("latin-1")
                except UnicodeDecodeError:
                    return (
                        jsonify(
                            {
                                "status": "error",
                                "message": "File tidak dapat dibaca. Pastikan enkoding file adalah UTF-8 atau Latin-1",
                            }
                        ),
                        400,
                    )
            df = pd.read_csv(io.StringIO(decoded))
            if df.empty:
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": "File CSV kosong atau tidak memiliki baris data",
                        }
                    ),
                    400,
                )
            if len(df) > app.config["MAX_BATCH_ROWS"]:
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": (
                                f"File terlalu besar: {len(df)} baris. "
                                f"Maksimal {app.config['MAX_BATCH_ROWS']} baris."
                            ),
                        }
                    ),
                    413,
                )
            results, errors = predictor.predict_batch_with_errors(df)

            sanitized_errors = [
                {"row": e["row"], "error": e["error"]}
                for e in errors
            ]

            _RISK_MAP = {
                "Risiko Tinggi": "tinggi",
                "Risiko Sedang": "sedang",
                "Risiko Rendah": "rendah",
            }
            from predictor import FEATURE_NAMES
            for r in results:
                nim = r.get("nim")
                if not nim:
                    continue
                try:
                    # Batch upload tidak memerlukan birth_date
                    student_id = db.upsert_student(str(nim), r.get("nama"))
                    risk_label = r.get("tingkat_risiko", "")
                    risk_level = _RISK_MAP.get(risk_label, "rendah")
                    db.save_prediction(
                        student_id=student_id,
                        risk_level=risk_level,
                        p_dropout=float(r.get("kemungkinan_dropout_pct", 0)) / 100,
                        raw_features={k: r[k] for k in FEATURE_NAMES if k in r},
                        source="batch",
                    )
                except Exception as exc:
                    logger.warning("Gagal simpan batch ke DB, nim=%s: %s", nim, exc)
            logger.info(
                "Batch prediksi selesai, sukses=%d gagal=%d hash=%s",
                len(results),
                len(errors),
                file_hash,
            )
            return jsonify(
                {
                    "status": "success",
                    "results": results,
                    "errors": sanitized_errors,
                    "total": len(results) + len(errors),
                }
            )
        except ValueError as exc:
            logger.warning("Kesalahan validasi batch: %s", exc)
            return jsonify({"status": "error", "message": "Kesalahan validasi data. Periksa format kolom."}), 422
        except Exception as exc:
            logger.exception("Kesalahan tak terduga pada batch: %s", exc)
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Terjadi kesalahan saat memproses file. Coba lagi nanti.",
                    }
                ),
                500,
            )

    @app.route("/api/template")
    def download_template():
        template_path = os.path.join(app.root_path, "data", "template_batch.csv")
        if not os.path.exists(template_path):
            logger.error("Template tidak ditemukan: %s", template_path)
            return (
                jsonify({"status": "error", "message": "File template tidak ditemukan"}),
                404,
            )
        return send_file(
            template_path,
            as_attachment=True,
            download_name="template_batch.csv",
            mimetype="text/csv",
        )

    @app.route("/api/chat", methods=["POST"])
    @limiter.limit("30 per minute")
    def chat():
        data = request.get_json(silent=True) or {}
        message = (data.get("message") or "").strip()[:500]
        if not message:
            return jsonify({"status": "error", "message": "Pesan tidak boleh kosong."}), 400
        # Gunakan session ID per user session untuk history percakapan
        session_id = session.get("riwayat_nim", request.remote_addr or "anon")
        try:
            result = ai_chat(message, session_id=session_id)
            return jsonify({"status": "success", "data": result})
        except Exception as exc:
            logger.error("AI chat error: %s", exc)
            fallback = {"response": "Maaf, asisten AI sedang tidak tersedia. Coba lagi nanti.", "model": "fallback"}
            return jsonify({"status": "success", "data": fallback})

    @app.route("/api/chat/reset", methods=["POST"])
    def chat_reset():
        session_id = session.get("riwayat_nim", request.remote_addr or "anon")
        try:
            reset_chat_session(session_id)
            return jsonify({"status": "success"})
        except Exception:
            return jsonify({"status": "error"}), 500

    @app.route("/riwayat/logout", methods=["POST"])
    def riwayat_logout():
        session.pop("riwayat_authorized", None)
        session.pop("riwayat_nim", None)
        return redirect(url_for("riwayat_verifikasi"))

    @app.route("/api/ai-status")
    def ai_status():
        """Endpoint debug: cek apakah Groq/HF API terhubung."""
        try:
            status = get_ai_status()
            return jsonify({"status": "success", "data": status})
        except Exception as exc:
            return jsonify({"status": "error", "message": str(exc)}), 500

    @app.errorhandler(404)
    def not_found(_):

        return render_template("404.html"), 404

    @app.errorhandler(500)
    def server_error(_):
        return render_template("500.html"), 500

    @app.errorhandler(RequestEntityTooLarge)
    def file_too_large(_):
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Ukuran file terlalu besar. Maksimum yang diizinkan adalah 16 MB",
                }
            ),
            413,
        )

    # ── Security Headers ─────────────────────────────────────────────────────
    @app.after_request
    def add_security_headers(response):
        nonce = g.get("csp_nonce", "")
        csp_directives = [
            "default-src 'self'",
            f"script-src 'self' 'nonce-{nonce}' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com",
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
            "font-src 'self' https://fonts.gstatic.com",
            "img-src 'self' data:",
            "object-src 'none'",
            "base-uri 'self'",
            "connect-src 'self' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com"
        ]
        response.headers["Content-Security-Policy"] = "; ".join(csp_directives)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        if app.config.get("ENVIRONMENT") == "production":
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains; preload"
            )
        return response

    return app

if __name__ == "__main__":
    app = create_app()
    is_production = os.environ.get("FLASK_ENV", "production") == "production"
    debug_mode = (not is_production) and os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    if debug_mode:
        logger.warning("Aplikasi berjalan dalam mode DEBUG. Jangan gunakan di produksi!")
    app.run(debug=debug_mode, host="0.0.0.0", port=5000)