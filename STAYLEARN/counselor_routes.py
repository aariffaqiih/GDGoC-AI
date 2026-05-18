"""
counselor_routes.py — Blueprint Flask untuk fitur konselor StayLearn.

URL prefix: /konselor
Routes:
  GET  /konselor/                       → Antrian prioritas / list mahasiswa (login required)
  GET  /konselor/analytics              → Dashboard chart analytics (login required)
  GET  /konselor/api/analytics          → JSON data chart analytics (login required)
  GET  /konselor/api/equity             → JSON data equity dashboard (SDG 4.5) (login required)
  GET  /konselor/effectiveness          → Halaman efektivitas intervensi (SDG 4.1) (login required)
  GET  /konselor/api/effectiveness      → JSON data efektivitas intervensi (login required)
  GET  /konselor/mahasiswa/<id>         → Detail + history satu mahasiswa (login required)
  PATCH /konselor/api/konseling/<id>    → Update status/jadwal/catatan (login required)
  POST /konselor/reset-data             → Reset data (DEVELOPMENT ONLY — guarded)
  POST /konselor/logout                 → Hapus sesi dan redirect ke login (CSRF-protected)

Security notes:
  - Logout menggunakan POST + CSRF token untuk mencegah CSRF logout attack (Finding #2).
  - reset-data dibatasi hanya untuk mode debug/development (Finding #4).
  - Semua endpoint dilindungi @login_required via before_request hook.
"""

from datetime import datetime
from flask import Blueprint, abort, current_app, jsonify, render_template, request, redirect, url_for
from flask_login import login_required, logout_user
import database as db

counselor_bp = Blueprint("counselor", __name__, url_prefix="/konselor")

# ── Proteksi seluruh blueprint dengan @login_required ─────────────────────────
@counselor_bp.before_request
@login_required
def protect():
    pass

# ── Halaman List (Priority Queue) ─────────────────────────────────────────────
@counselor_bp.route("/")
def dashboard():
    filter_risk = request.args.get("risk", "semua")
    filter_status = request.args.get("status", "semua")

    queue = db.get_priority_queue(filter_risk, filter_status)
    stats = db.get_dashboard_stats()

    return render_template(
        "counselor/dashboard.html",
        queue=queue,
        stats=stats,
        filter_risk=filter_risk,
        filter_status=filter_status,
    )

# ── Halaman Analytics (Dashboard Chart) ───────────────────────────────────────
@counselor_bp.route("/analytics")
def analytics():
    stats = db.get_dashboard_stats()
    return render_template("counselor/analytics.html", stats=stats)

# ── API: Data Analytics untuk Chart.js ────────────────────────────────────────
@counselor_bp.route("/api/analytics")
def api_analytics():
    """
    Endpoint JSON yang menyediakan data agregat untuk semua chart di halaman analytics.
    Data di-compute on-the-fly; pertimbangkan Redis-caching jika dataset sangat besar.
    """
    try:
        data = db.get_analytics_data()
        return jsonify({"status": "success", "data": data})
    except Exception as exc:
        import logging
        logging.getLogger(__name__).error("Gagal mengambil analytics data: %s", exc)
        # Finding #8: Jangan bocorkan detail error ke klien.
        return jsonify({
            "status": "error",
            "message": "Gagal memuat data analitik. Coba lagi nanti."
        }), 500

# ── API: Equity Dashboard (SDG 4.5) ───────────────────────────────────────────
@counselor_bp.route("/api/equity")
def api_equity():
    """
    Endpoint JSON untuk Equity Dashboard.
    Menyediakan agregasi risiko dropout berdasarkan lokasi dan kuintil pendapatan.
    Juga menyertakan fairness metrics jika data aktual dropout tersedia.
    """
    try:
        data = db.get_equity_data()
        return jsonify({"status": "success", "data": data})
    except Exception as exc:
        import logging
        logging.getLogger(__name__).error("Gagal mengambil equity data: %s", exc)
        return jsonify({
            "status": "error",
            "message": "Gagal memuat data equity dashboard. Coba lagi nanti."
        }), 500

# ── Halaman Efektivitas Intervensi (SDG 4.1) ──────────────────────────────────
@counselor_bp.route("/effectiveness")
def effectiveness():
    """
    Halaman dashboard Intervention Effectiveness Tracking (SDG 4.1).
    Data diambil secara async oleh JS via /api/effectiveness agar halaman
    tetap responsif saat DB query berjalan.
    """
    return render_template("counselor/effectiveness.html")

# ── API: Efektivitas Intervensi (SDG 4.1) ─────────────────────────────────────
@counselor_bp.route("/api/effectiveness")
def api_effectiveness():
    """
    Endpoint JSON untuk Intervention Effectiveness Dashboard.

    Query parameter (opsional):
        min_days (int): Override minimal hari evaluasi. Default: 30.
                        Dibatasi ke rentang [7, 180] untuk mencegah abuse.

    Returns JSON dengan struktur:
        {
            "status": "success",
            "data": {
                "summary": { ... },
                "records": [ ... ],
                "generated_at": "...",
                "min_days": int
            }
        }
    """
    # Validasi dan sanitasi query param min_days
    raw_min_days = request.args.get("min_days", "")
    if raw_min_days:
        try:
            min_days = int(raw_min_days)
            # Batasi ke rentang yang masuk akal untuk mencegah query scan besar
            min_days = max(7, min(180, min_days))
        except (ValueError, TypeError):
            return jsonify({
                "status": "error",
                "message": "Parameter min_days harus berupa angka bulat antara 7 dan 180."
            }), 400
        # Custom min_days — jangan gunakan cache yang mungkin pakai nilai berbeda
        use_cache = False
    else:
        min_days = db.EFFECTIVENESS_MIN_DAYS
        use_cache = True

    try:
        data = db.get_intervention_effectiveness(min_days=min_days, use_cache=use_cache)
        return jsonify({"status": "success", "data": data})
    except Exception as exc:
        import logging
        logging.getLogger(__name__).error(
            "Gagal mengambil effectiveness data: %s", exc
        )
        return jsonify({
            "status": "error",
            "message": "Gagal memuat data efektivitas. Coba lagi nanti."
        }), 500

# ── Detail Mahasiswa ───────────────────────────────────────────────────────────
@counselor_bp.route("/mahasiswa/<int:student_id>")
def student_detail(student_id):
    history = db.get_student_history(student_id)
    if history is None:
        return render_template("404.html"), 404
    return render_template("counselor/student.html", **history)

# ── API: Update Counseling Record ──────────────────────────────────────────────
@counselor_bp.route("/api/konseling/<int:counseling_id>", methods=["PATCH"])
def update_counseling(counseling_id):
    data = request.get_json(silent=True) or {}

    allowed = {"status", "scheduled_at", "completed_at", "notes"}
    updates = {k: v for k, v in data.items() if k in allowed}

    if not updates:
        return jsonify({
            "status": "error",
            "message": "Tidak ada field yang valid. Gunakan: status, scheduled_at, completed_at, atau notes."
        }), 400

    # Validasi status
    valid_statuses = {"pending", "scheduled", "done", "skipped"}
    if "status" in updates and updates["status"] not in valid_statuses:
        return jsonify({
            "status": "error",
            "message": f"Status tidak valid. Pilihan: {', '.join(sorted(valid_statuses))}"
        }), 400

    # Validasi format datetime
    def validate_datetime(dt_str: str) -> bool:
        try:
            datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
            return True
        except (ValueError, TypeError):
            return False

    if "scheduled_at" in updates and not validate_datetime(updates["scheduled_at"]):
        return jsonify({
            "status": "error",
            "message": "Format scheduled_at tidak valid. Gunakan YYYY-MM-DD HH:MM:SS"
        }), 400

    if "completed_at" in updates and not validate_datetime(updates["completed_at"]):
        return jsonify({
            "status": "error",
            "message": "Format completed_at tidak valid. Gunakan YYYY-MM-DD HH:MM:SS"
        }), 400

    # Auto-set completed_at saat status → done
    if updates.get("status") == "done" and "completed_at" not in updates:
        updates["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Auto-set: wajib sertakan scheduled_at saat status → scheduled (jika belum ada)
    if updates.get("status") == "scheduled" and "scheduled_at" not in updates:
        existing = db.get_counseling_record(counseling_id)
        if existing and not existing.get("scheduled_at"):
            return jsonify({
                "status": "error",
                "message": "Sertakan scheduled_at (format: YYYY-MM-DD HH:MM:SS) saat mengubah status ke 'scheduled'."
            }), 400

    success = db.update_counseling(counseling_id, **updates)
    if not success:
        return jsonify({
            "status": "error",
            "message": "Record tidak ditemukan atau tidak ada perubahan."
        }), 404

    record = db.get_counseling_record(counseling_id)
    return jsonify({"status": "success", "record": record})

# ── Reset Data (DEVELOPMENT ONLY) ─────────────────────────────────────────────
@counselor_bp.route("/reset-data", methods=["POST"])
def reset_data():
    """
    Hapus semua data mahasiswa, prediksi, dan catatan konseling.

    Finding #4: Endpoint ini HANYA tersedia saat aplikasi berjalan dalam
    mode debug/development. Di produksi (current_app.debug == False),
    endpoint ini mengembalikan 404 seolah-olah tidak ada.
    Ini mencegah wipe database oleh konselor yang nakal atau sesi yang
    dibobol di lingkungan produksi.
    """
    # Guard: hanya tersedia di luar produksi (FLASK_ENV != 'production')
    import os as _os
    if _os.environ.get("FLASK_ENV", "production") == "production":
        abort(404)

    db.reset_student_data()
    # Invalidate equity cache after reset
    db.invalidate_equity_cache()
    return jsonify({"status": "success", "message": "Data mahasiswa berhasil direset."})

# ── Logout (POST only, CSRF-protected) ────────────────────────────────────────
@counselor_bp.route("/logout", methods=["POST"])
def logout():
    """
    Hapus sesi login konselor dan redirect ke halaman login.

    Finding #2: Logout menggunakan POST bukan GET.
    GET logout rentan terhadap CSRF attack — penyerang cukup menyisipkan
    <img src="/konselor/logout"> di halaman lain untuk memaksa korban logout.
    Dengan POST + CSRF token (dikirim dari form di sidebar), serangan ini
    diblokir oleh Flask-WTF secara otomatis.
    """
    logout_user()
    return redirect(url_for("login"))
# ── API: Collective Wellbeing Early Warning (SDG 4.a) ─────────────────────────
@counselor_bp.route("/api/wellbeing")
def api_wellbeing():
    """
    Endpoint JSON untuk Collective Wellbeing Early Warning.
    Memeriksa rata-rata stres kolektif dan lonjakan risiko tinggi.
    """
    try:
        from config import Config
        data = db.get_wellbeing_alerts(
            stress_threshold=Config.WELLBEING_STRESS_THRESHOLD,
            high_risk_spike_pct=Config.WELLBEING_HIGH_RISK_SPIKE_THRESHOLD,
            lookback_days=Config.WELLBEING_LOOKBACK_DAYS,
        )
        return jsonify({"status": "success", "data": data})
    except Exception as exc:
        import logging
        logging.getLogger(__name__).error("Gagal mengambil wellbeing data: %s", exc)
        return jsonify({
            "status": "error",
            "message": "Gagal memuat data wellbeing."
        }), 500


# ── API: Intervention Recommendation (SDG 4.1 & 4.4) ─────────────────────────
@counselor_bp.route("/api/recommend/<int:student_id>")
def api_recommend(student_id: int):
    """
    Rekomendasikan intervensi berdasarkan profil mahasiswa dan historis
    intervensi yang berhasil pada profil serupa (KMeans clustering).

    Returns JSON:
        recommendations: list intervensi yang direkomendasikan
        method:          "kmeans_cluster" | "rule_based_fallback"
        similar_cases_count: jumlah kasus historis yang digunakan
    """
    features = db.get_latest_features_for_student(student_id)
    if features is None:
        return jsonify({
            "status": "error",
            "message": "Tidak ada prediksi untuk mahasiswa ini."
        }), 404

    risk_level = features.get("risk_level", "sedang")

    try:
        from recommender import get_recommender
        rec = get_recommender()
        result = rec.recommend(features, risk_level=risk_level)
        return jsonify({"status": "success", "data": result})
    except Exception as exc:
        import logging
        logging.getLogger(__name__).error("Recommendation error: %s", exc)
        return jsonify({
            "status": "error",
            "message": "Gagal menghasilkan rekomendasi."
        }), 500


# ── API: XAI — Feature Impact & Counterfactual (SDG 4.1) ─────────────────────
@counselor_bp.route("/api/explain/<int:student_id>")
def api_explain(student_id: int):
    """
    Jelaskan faktor yang paling mempengaruhi risiko dropout mahasiswa (SHAP/koefisien)
    beserta skenario counterfactual ("jika X berubah maka risiko turun Y%").

    Returns JSON:
        feature_impacts:  list fitur diurutkan berdasarkan dampak
        counterfactuals:  list skenario perubahan konkret
        top_risk_factor:  faktor utama risiko
        method:           "shap" atau "coefficient"
    """
    features = db.get_latest_features_for_student(student_id)
    if features is None:
        return jsonify({
            "status": "error",
            "message": "Tidak ada prediksi untuk mahasiswa ini."
        }), 404

    p_dropout = features.pop("p_dropout", 0.5)
    features.pop("risk_level", None)

    try:
        # Ambil model dari g (sudah diinisialisasi di create_app)
        from flask import current_app, g
        predictor = current_app.config.get("_predictor_ref")
        if predictor is None:
            return jsonify({
                "status": "error",
                "message": "Model tidak tersedia."
            }), 503

        from xai import explain_prediction
        result = explain_prediction(predictor.model, features, p_dropout)
        return jsonify({"status": "success", "data": result})
    except Exception as exc:
        import logging
        logging.getLogger(__name__).error("XAI error: %s", exc)
        return jsonify({
            "status": "error",
            "message": "Gagal menghasilkan penjelasan model."
        }), 500


# ── API: Update Intervention Type (SDG 4.1) ───────────────────────────────────
@counselor_bp.route("/api/konseling/<int:counseling_id>/intervention-type", methods=["PATCH"])
def update_intervention_type(counseling_id: int):
    """
    Update jenis intervensi untuk satu sesi konseling.
    Bisa dipanggil otomatis (dari notes) atau manual dari frontend.
    """
    data = request.get_json(silent=True) or {}
    intervention_type = data.get("intervention_type", "").strip()

    valid_types = {
        "motivasi", "bimbingan_akademik", "bantuan_finansial",
        "pengembangan_karir", "dukungan_keluarga", "kesehatan_mental",
        "akses_teknologi", "manajemen_waktu", "umum",
    }
    if not intervention_type:
        # Auto-extract dari notes
        record = db.get_counseling_record(counseling_id)
        if not record:
            return jsonify({"status": "error", "message": "Record tidak ditemukan."}), 404
        from recommender import extract_intervention_type
        intervention_type = extract_intervention_type(record.get("notes", ""))
    elif intervention_type not in valid_types:
        return jsonify({
            "status": "error",
            "message": f"Tipe tidak valid. Pilih: {', '.join(sorted(valid_types))}"
        }), 400

    success = db.update_counseling(counseling_id, intervention_type=intervention_type)
    if not success:
        return jsonify({"status": "error", "message": "Record tidak ditemukan."}), 404

    return jsonify({
        "status": "success",
        "intervention_type": intervention_type,
        "counseling_id": counseling_id,
    })


# ── API: Teacher Analytics (SDG 4.c) ─────────────────────────────────────────
@counselor_bp.route("/api/teacher-analytics")
def api_teacher_analytics():
    """
    Agregasi data kualitas pengajaran dari perspektif mahasiswa (SDG 4.c).
    Data bersumber dari teaching_quality_rating yang diinput mahasiswa saat prediksi.
    """
    try:
        data = db.get_teaching_analytics()
        return jsonify({"status": "success", "data": data})
    except Exception as exc:
        import logging
        logging.getLogger(__name__).error("Teacher analytics error: %s", exc)
        return jsonify({
            "status": "error",
            "message": "Gagal memuat data kualitas pengajaran."
        }), 500


# ── Halaman Teacher Analytics ──────────────────────────────────────────────────
@counselor_bp.route("/teacher-analytics")
def teacher_analytics():
    return render_template("counselor/teacher_analytics.html")


# ── API: Mentor CRUD (SDG 4.a) ────────────────────────────────────────────────
@counselor_bp.route("/api/mentors", methods=["GET"])
def api_mentors_list():
    """Daftar semua mentor aktif."""
    try:
        mentors = db.get_all_mentors(active_only=True)
        return jsonify({"status": "success", "data": mentors})
    except Exception as exc:
        import logging
        logging.getLogger(__name__).error("Mentors list error: %s", exc)
        return jsonify({"status": "error", "message": "Gagal memuat data mentor."}), 500


@counselor_bp.route("/api/mentors", methods=["POST"])
def api_mentor_create():
    """Daftarkan mentor baru."""
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    nim  = (data.get("nim") or "").strip()
    if not name or not nim:
        return jsonify({
            "status": "error",
            "message": "name dan nim wajib diisi."
        }), 400
    mentor_id = db.create_mentor(
        name=name,
        nim=nim,
        major=data.get("major"),
        availability_json=data.get("availability_json", "[]"),
        max_mentees=int(data.get("max_mentees", 3)),
    )
    if mentor_id is None:
        return jsonify({"status": "error", "message": "NIM mentor sudah terdaftar."}), 409
    return jsonify({"status": "success", "mentor_id": mentor_id}), 201


@counselor_bp.route("/api/mentors/<int:mentor_id>", methods=["PATCH"])
def api_mentor_update(mentor_id: int):
    """Update data mentor."""
    data = request.get_json(silent=True) or {}
    allowed = {"name", "major", "availability_json", "max_mentees", "is_active"}
    updates = {k: v for k, v in data.items() if k in allowed}
    if not updates:
        return jsonify({"status": "error", "message": "Tidak ada field yang valid."}), 400
    success = db.update_mentor(mentor_id, **updates)
    if not success:
        return jsonify({"status": "error", "message": "Mentor tidak ditemukan."}), 404
    return jsonify({"status": "success", "mentor_id": mentor_id})


@counselor_bp.route("/api/mentors/match/<int:student_id>", methods=["GET"])
def api_mentor_match(student_id: int):
    """Auto-match mentor untuk mahasiswa tertentu."""
    mentor = db.match_mentor_for_student(student_id)
    if mentor is None:
        return jsonify({
            "status": "success",
            "data": None,
            "message": "Tidak ada mentor yang tersedia saat ini.",
        })
    return jsonify({"status": "success", "data": mentor})


@counselor_bp.route("/api/mentor-sessions", methods=["POST"])
def api_mentor_session_create():
    """Buat sesi mentor baru."""
    data = request.get_json(silent=True) or {}
    try:
        mentor_id   = int(data["mentor_id"])
        mentee_id   = int(data["mentee_id"])
        session_date = str(data["session_date"])
    except (KeyError, ValueError, TypeError):
        return jsonify({
            "status": "error",
            "message": "mentor_id, mentee_id, dan session_date wajib diisi."
        }), 400
    session_id = db.create_mentor_session(
        mentor_id=mentor_id,
        mentee_id=mentee_id,
        session_date=session_date,
        notes=data.get("notes"),
    )
    return jsonify({"status": "success", "session_id": session_id}), 201


@counselor_bp.route("/api/mentor-sessions/<int:session_id>", methods=["PATCH"])
def api_mentor_session_update(session_id: int):
    """Update sesi mentor (status, notes, rating)."""
    data = request.get_json(silent=True) or {}
    allowed = {"status", "notes", "rating", "session_date"}
    updates = {k: v for k, v in data.items() if k in allowed}
    if "rating" in updates:
        try:
            r = int(updates["rating"])
            if not 1 <= r <= 5:
                raise ValueError
            updates["rating"] = r
        except (ValueError, TypeError):
            return jsonify({"status": "error", "message": "Rating harus 1–5."}), 400
    success = db.update_mentor_session(session_id, **updates)
    if not success:
        return jsonify({"status": "error", "message": "Sesi tidak ditemukan."}), 404
    return jsonify({"status": "success", "session_id": session_id})


@counselor_bp.route("/api/mentor-sessions", methods=["GET"])
def api_mentor_sessions_list():
    """Daftar sesi mentor, bisa filter per mentor atau mentee."""
    mentor_id = request.args.get("mentor_id", type=int)
    mentee_id = request.args.get("mentee_id", type=int)
    try:
        sessions = db.get_mentor_sessions(mentor_id=mentor_id, mentee_id=mentee_id)
        return jsonify({"status": "success", "data": sessions})
    except Exception as exc:
        import logging
        logging.getLogger(__name__).error("Mentor sessions error: %s", exc)
        return jsonify({"status": "error", "message": "Gagal memuat sesi mentor."}), 500


# ── Halaman Mentor Dashboard ───────────────────────────────────────────────────
@counselor_bp.route("/mentors")
def mentors():
    return render_template("counselor/mentors.html")


# ── API: Statistical Significance untuk Effectiveness ─────────────────────────
@counselor_bp.route("/api/effectiveness/significance")
def api_effectiveness_significance():
    """
    Hitung signifikansi statistik (paired t-test + Cohen's d) dari data efektivitas.
    Digunakan untuk mendukung publikasi ilmiah / laporan akreditasi BAN-PT.
    """
    try:
        eff_data = db.get_intervention_effectiveness(use_cache=False)
        evaluated = [
            r for r in eff_data.get("records", [])
            if r["evaluation_status"] == "evaluated"
               and r["p_dropout_before_pct"] is not None
               and r["p_dropout_after_pct"] is not None
        ]
        if len(evaluated) < 3:
            return jsonify({
                "status": "success",
                "data": {
                    "n": len(evaluated),
                    "p_value": None,
                    "is_significant": None,
                    "effect_size": None,
                    "interpretation": "Data tidak cukup untuk analisis statistik (minimal 3 kasus terevaluasi).",
                    "test_used": "none",
                    "mean_before": None,
                    "mean_after": None,
                    "mean_difference": None,
                }
            })

        before = [r["p_dropout_before_pct"] / 100 for r in evaluated]
        after  = [r["p_dropout_after_pct"]  / 100 for r in evaluated]
        stats  = db.compute_statistical_significance(before, after)
        return jsonify({"status": "success", "data": stats})
    except Exception as exc:
        import logging
        logging.getLogger(__name__).error("Statistical significance error: %s", exc)
        return jsonify({
            "status": "error",
            "message": "Gagal menghitung signifikansi statistik."
        }), 500

# ── Infrastructure Survey (SDG 4.a) ───────────────────────────────────────────
@counselor_bp.route("/infrastructure")
def infrastructure():
    return render_template("counselor/infrastructure.html")


@counselor_bp.route("/api/infrastructure", methods=["GET"])
def api_infrastructure_get():
    """Ringkasan survei infrastruktur per kampus (SDG 4.a.1)."""
    try:
        data = db.get_infrastructure_summary()
        return jsonify({"status": "success", "data": data})
    except Exception as exc:
        import logging
        logging.getLogger(__name__).error("Infrastructure summary error: %s", exc)
        return jsonify({"status": "error", "message": "Gagal memuat data infrastruktur."}), 500


@counselor_bp.route("/api/infrastructure", methods=["POST"])
def api_infrastructure_post():
    """Simpan laporan survei infrastruktur baru."""
    data = request.get_json(silent=True) or {}
    required = ["campus", "electricity", "clean_water", "sanitation", "separate_wc",
                "computers", "internet", "disability_access", "safe_environment"]
    for field in required:
        if field not in data:
            return jsonify({"status": "error", "message": f"Field '{field}' wajib diisi."}), 400
    try:
        report_id = db.save_infrastructure_report(
            campus=str(data["campus"]).strip(),
            faculty=data.get("faculty"),
            reported_by=data.get("reported_by"),
            electricity=bool(data["electricity"]),
            clean_water=bool(data["clean_water"]),
            sanitation=bool(data["sanitation"]),
            separate_wc=bool(data["separate_wc"]),
            computers=bool(data["computers"]),
            internet=bool(data["internet"]),
            disability_access=bool(data["disability_access"]),
            safe_environment=bool(data["safe_environment"]),
            notes=data.get("notes"),
        )
        return jsonify({"status": "success", "report_id": report_id}), 201
    except Exception as exc:
        import logging
        logging.getLogger(__name__).error("Infrastructure save error: %s", exc)
        return jsonify({"status": "error", "message": "Gagal menyimpan laporan."}), 500


# ── API: Equity Extended dengan Gender Parity (SDG 4.5) ─────────────────────
@counselor_bp.route("/api/equity-extended")
def api_equity_extended():
    """Equity dashboard diperluas: gender, disabilitas, indigeneity, konflik."""
    try:
        data = db.get_equity_data_extended()
        return jsonify({"status": "success", "data": data})
    except Exception as exc:
        import logging
        logging.getLogger(__name__).error("Equity extended error: %s", exc)
        return jsonify({"status": "error", "message": "Gagal memuat data equity."}), 500


@counselor_bp.route("/api/students/<int:student_id>/equity", methods=["PATCH"])
def api_student_equity_update(student_id: int):
    """Update data equity mahasiswa (gender, disability, dsb.) oleh konselor."""
    data = request.get_json(silent=True) or {}
    valid_genders = {"L", "P", None}
    gender = data.get("gender")
    if gender is not None and gender.upper()[:1] not in {"L", "P"}:
        return jsonify({"status": "error", "message": "gender harus 'L' atau 'P'."}), 400
    success = db.update_student_equity_info(
        student_id=student_id,
        gender=data.get("gender"),
        disability_status=data.get("disability_status"),
        indigenous=data.get("indigenous"),
        conflict_affected=data.get("conflict_affected"),
    )
    if not success:
        return jsonify({"status": "error", "message": "Tidak ada perubahan atau student tidak ditemukan."}), 404
    return jsonify({"status": "success", "student_id": student_id})


# ── API: Wellbeing Time-Series Trend (SDG 4.a) ───────────────────────────────
@counselor_bp.route("/api/wellbeing-trend")
def api_wellbeing_trend():
    """
    Data time-series tren stres kolektif dan risiko tinggi per minggu.
    Query param: weeks (int, 1–52, default 8)
    """
    weeks = request.args.get("weeks", 8, type=int)
    weeks = max(1, min(52, weeks))
    try:
        data = db.get_wellbeing_trend(weeks=weeks)
        return jsonify({"status": "success", "data": data})
    except Exception as exc:
        import logging
        logging.getLogger(__name__).error("Wellbeing trend error: %s", exc)
        return jsonify({"status": "error", "message": "Gagal memuat tren wellbeing."}), 500


# ── API: Intervention Effectiveness Trend + Heatmap (SDG 4.1) ────────────────
@counselor_bp.route("/api/intervention-trend")
def api_intervention_trend():
    """Tren efektivitas intervensi per bulan + heatmap per jenis intervensi."""
    try:
        data = db.get_intervention_trend()
        return jsonify({"status": "success", "data": data})
    except Exception as exc:
        import logging
        logging.getLogger(__name__).error("Intervention trend error: %s", exc)
        return jsonify({"status": "error", "message": "Gagal memuat tren intervensi."}), 500


# ── API: Mentor Effectiveness (SDG 4.a) ──────────────────────────────────────
@counselor_bp.route("/api/mentor-effectiveness")
def api_mentor_effectiveness():
    """Efektivitas program mentoring vs non-mentoring."""
    try:
        data = db.get_mentor_effectiveness()
        return jsonify({"status": "success", "data": data})
    except Exception as exc:
        import logging
        logging.getLogger(__name__).error("Mentor effectiveness error: %s", exc)
        return jsonify({"status": "error", "message": "Gagal memuat data efektivitas mentor."}), 500


# ── Halaman Equity Extended ────────────────────────────────────────────────────
@counselor_bp.route("/equity")
def equity():
    return render_template("counselor/equity.html")

# ── Halaman Wellbeing Trend + Intervention Trend ──────────────────────────────
@counselor_bp.route("/trends")
def trends():
    return render_template("counselor/wellbeing_trend.html")
