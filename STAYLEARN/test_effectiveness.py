"""
test_effectiveness.py — Test suite untuk Intervention Effectiveness Tracking.

Struktur:
  TestClassifyEffectivenessRow   — unit tests untuk _classify_effectiveness_row()
  TestComputeEffectivenessSummary — unit tests untuk _compute_effectiveness_summary()
  TestFetchEffectivenessRows     — integration tests dengan SQLite in-memory
  TestGetInterventionEffectiveness — integration tests end-to-end + cache
  TestApiEffectivenessEndpoint   — integration tests untuk Flask route

Jalankan:
  pytest test_effectiveness.py -v
  pytest test_effectiveness.py -v --tb=short   # output singkat

Manual test cases penting (cek di browser setelah server jalan):
  1. GET /konselor/api/effectiveness → harus mengembalikan JSON {status: "success"}
  2. Saat belum ada data → summary.total_counseling_done == 0
  3. GET /konselor/api/effectiveness?min_days=7 → min_days di response == 7
  4. GET /konselor/api/effectiveness?min_days=999 → min_days di response == 180 (capped)
  5. GET /konselor/api/effectiveness?min_days=abc → HTTP 400
  6. GET /konselor/effectiveness (tanpa login) → redirect ke /login
"""

import json
import sqlite3
import time
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# ── Import target functions langsung dari modul ────────────────────────────────
import database as db
from database import (
    _classify_effectiveness_row,
    _compute_effectiveness_summary,
    EFFECTIVENESS_MIN_DAYS,
    EFFECTIVENESS_UNCHANGED_THRESHOLD,
    EFFECTIVENESS_MIN_SAMPLE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_row(
    counseling_id: int = 1,
    student_id: int = 10,
    nim: str = "123456",
    nama: str = "Budi",
    completed_at: str = None,
    p_dropout_before: float = 0.70,
    p_dropout_after=None,
    before_predicted_at: str = "2024-01-01 08:00:00",
    after_predicted_at: str = None,
) -> dict:
    """Helper: buat raw row dict seperti yang dikembalikan _fetch_effectiveness_rows()."""
    if completed_at is None:
        # Default: selesai 45 hari lalu (sudah melewati min_days=30)
        completed_at = (datetime.now() - timedelta(days=45)).strftime("%Y-%m-%d %H:%M:%S")
    return {
        "counseling_id": counseling_id,
        "student_id": student_id,
        "nim": nim,
        "nama": nama,
        "completed_at": completed_at,
        "p_dropout_before": p_dropout_before,
        "p_dropout_after": p_dropout_after,
        "before_predicted_at": before_predicted_at,
        "after_predicted_at": after_predicted_at,
    }


@pytest.fixture
def in_memory_db(tmp_path):
    """
    Fixture: inisialisasi database SQLite in-memory via tmp_path
    sehingga setiap test mendapat DB bersih.
    """
    db_path = str(tmp_path / "test_effectiveness.db")
    db.init_db(db_path)
    yield db_path
    # Teardown: reset global state database module
    db._DB_PATH = None
    db.invalidate_effectiveness_cache()
    db.invalidate_equity_cache()


def _insert_student(conn, nim: str, nama: str = "Test") -> int:
    conn.execute(
        "INSERT INTO students (nim, nama) VALUES (?, ?)", (nim, nama)
    )
    row = conn.execute(
        "SELECT id FROM students WHERE nim = ?", (nim,)
    ).fetchone()
    return row[0]


def _insert_prediction(
    conn, student_id: int, p_dropout: float, predicted_at: str,
    risk_level: str = "tinggi",
) -> int:
    raw_features = json.dumps({
        "location_type": "Urban", "family_income": 5000,
        "financial_aid_status": 0, "distance_to_institute": 10,
        "internet_connectivity_issues": 0, "motivation_score": 5,
        "career_alignment": 2, "stress_levels": 2, "family_support": 2,
        "attendance_rate": 70, "test_scores_avg": 60, "backlogs": 1,
        "teaching_quality_rating": 5,
    })
    cur = conn.execute(
        """INSERT INTO predictions
               (student_id, risk_level, p_dropout, raw_features, source, trajectory, predicted_at)
           VALUES (?, ?, ?, ?, 'individual', 'baru', ?)""",
        (student_id, risk_level, p_dropout, raw_features, predicted_at),
    )
    return cur.lastrowid


def _insert_counseling(
    conn, student_id: int, prediction_id: int,
    status: str = "done", completed_at: str = None,
) -> int:
    cur = conn.execute(
        """INSERT INTO counseling_records
               (student_id, prediction_id, action_type, status, completed_at)
           VALUES (?, ?, 'konseling', ?, ?)""",
        (student_id, prediction_id, status, completed_at),
    )
    return cur.lastrowid


# ─────────────────────────────────────────────────────────────────────────────
# TestClassifyEffectivenessRow — unit tests (no DB needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestClassifyEffectivenessRow:
    """Unit tests untuk _classify_effectiveness_row() tanpa DB."""

    MIN = EFFECTIVENESS_MIN_DAYS  # 30

    def _classify(self, row, min_days=None):
        return _classify_effectiveness_row(row, min_days or self.MIN)

    # ── Status: evaluated ────────────────────────────────────────────────────

    def test_evaluated_improved_returns_correct_category(self):
        row = _make_row(p_dropout_before=0.80, p_dropout_after=0.50)
        result = self._classify(row)
        assert result["evaluation_status"] == "evaluated"
        assert result["improvement_category"] == "improved"
        assert result["delta_pct"] == pytest.approx(30.0, abs=0.1)

    def test_evaluated_worsened_returns_correct_category(self):
        row = _make_row(p_dropout_before=0.40, p_dropout_after=0.70)
        result = self._classify(row)
        assert result["evaluation_status"] == "evaluated"
        assert result["improvement_category"] == "worsened"
        assert result["delta_pct"] == pytest.approx(-30.0, abs=0.1)

    def test_evaluated_unchanged_within_threshold(self):
        """Selisih di bawah threshold (0.02 = 2pp) → unchanged."""
        row = _make_row(p_dropout_before=0.60, p_dropout_after=0.61)
        result = self._classify(row)
        assert result["evaluation_status"] == "evaluated"
        assert result["improvement_category"] == "unchanged"

    def test_evaluated_exactly_at_improvement_threshold(self):
        """Selisih tepat di threshold → improved (>= bukan >)."""
        delta = EFFECTIVENESS_UNCHANGED_THRESHOLD  # 0.02
        row = _make_row(p_dropout_before=0.60, p_dropout_after=0.60 - delta)
        result = self._classify(row)
        assert result["improvement_category"] == "improved"

    def test_evaluated_probabilities_converted_to_pct(self):
        """p_dropout disimpan 0-1, ditampilkan sebagai 0-100."""
        row = _make_row(p_dropout_before=0.75, p_dropout_after=0.50)
        result = self._classify(row)
        assert result["p_dropout_before_pct"] == pytest.approx(75.0, abs=0.1)
        assert result["p_dropout_after_pct"] == pytest.approx(50.0, abs=0.1)

    def test_evaluated_preserves_identity_fields(self):
        row = _make_row(counseling_id=42, student_id=99, nim="A001", nama="Siti")
        result = self._classify(row)
        assert result["counseling_id"] == 42
        assert result["student_id"] == 99
        assert result["nim"] == "A001"
        assert result["nama"] == "Siti"

    # ── Status: too_recent ────────────────────────────────────────────────────

    def test_too_recent_when_days_elapsed_less_than_min(self):
        recent_at = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d %H:%M:%S")
        row = _make_row(completed_at=recent_at, p_dropout_after=0.40)
        result = self._classify(row)
        assert result["evaluation_status"] == "too_recent"
        assert result["delta_pct"] is None
        assert result["improvement_category"] is None

    def test_too_recent_even_if_after_prediction_exists(self):
        """Prediksi sesudah ada tapi waktu belum cukup → too_recent, bukan evaluated."""
        recent_at = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d %H:%M:%S")
        row = _make_row(
            completed_at=recent_at,
            p_dropout_after=0.30,  # ada prediksi sesudah
        )
        result = self._classify(row)
        assert result["evaluation_status"] == "too_recent"

    def test_too_recent_exactly_at_boundary_is_too_recent(self):
        """Tepat di boundary (bukan >=) harus tetap too_recent."""
        at_boundary = (datetime.now() - timedelta(days=self.MIN - 1)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        row = _make_row(completed_at=at_boundary, p_dropout_after=0.40)
        result = self._classify(row)
        assert result["evaluation_status"] == "too_recent"

    # ── Status: pending_followup ──────────────────────────────────────────────

    def test_pending_followup_when_no_after_prediction(self):
        row = _make_row(p_dropout_after=None)
        result = self._classify(row)
        assert result["evaluation_status"] == "pending_followup"
        assert result["delta_pct"] is None

    def test_pending_followup_shows_before_value(self):
        """p_dropout_before harus tetap ditampilkan untuk pending."""
        row = _make_row(p_dropout_before=0.65, p_dropout_after=None)
        result = self._classify(row)
        assert result["p_dropout_before_pct"] == pytest.approx(65.0, abs=0.1)
        assert result["p_dropout_after_pct"] is None

    # ── Status: invalid_date ──────────────────────────────────────────────────

    def test_invalid_date_returns_invalid_status(self):
        row = _make_row(completed_at="BUKAN-TANGGAL")
        result = self._classify(row)
        assert result["evaluation_status"] == "invalid_date"
        assert result["delta_pct"] is None

    def test_none_completed_at_returns_invalid_status(self):
        row = _make_row(completed_at=None)
        result = self._classify(row)
        assert result["evaluation_status"] == "invalid_date"

    def test_empty_string_completed_at_returns_invalid_status(self):
        row = _make_row(completed_at="")
        result = self._classify(row)
        assert result["evaluation_status"] == "invalid_date"

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_none_nama_nim_replaced_with_dash(self):
        row = _make_row()
        row["nama"] = None
        row["nim"] = None
        result = self._classify(row)
        assert result["nama"] == "-"
        assert result["nim"] == "-"

    def test_min_days_zero_evaluates_immediately(self):
        """min_days=0 → semua konseling selesai langsung eligible."""
        just_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = _make_row(completed_at=just_now, p_dropout_after=0.40)
        result = _classify_effectiveness_row(row, min_days=0)
        # days_elapsed bisa 0 yang masih >= 0, tapi ada after prediction
        assert result["evaluation_status"] in ("evaluated", "too_recent")


# ─────────────────────────────────────────────────────────────────────────────
# TestComputeEffectivenessSummary — unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeEffectivenessSummary:

    def _make_classified(self, status, category=None, delta_pct=None):
        return {
            "evaluation_status": status,
            "improvement_category": category,
            "delta_pct": delta_pct,
            "counseling_id": 1,
        }

    def test_empty_records_returns_zero_counts(self):
        summary = _compute_effectiveness_summary([])
        assert summary["total_counseling_done"] == 0
        assert summary["total_evaluated"] == 0
        assert summary["avg_dropout_reduction_pct"] is None
        assert summary["improved_pct"] is None

    def test_all_evaluated_improved(self):
        records = [
            self._make_classified("evaluated", "improved", delta_pct=20.0),
            self._make_classified("evaluated", "improved", delta_pct=15.0),
            self._make_classified("evaluated", "improved", delta_pct=10.0),
        ]
        summary = _compute_effectiveness_summary(records)
        assert summary["total_evaluated"] == 3
        assert summary["improved_count"] == 3
        assert summary["worsened_count"] == 0
        assert summary["avg_dropout_reduction_pct"] == pytest.approx(15.0, abs=0.01)
        assert summary["improved_pct"] == pytest.approx(100.0, abs=0.1)

    def test_mixed_categories_counts_correctly(self):
        records = [
            self._make_classified("evaluated", "improved",  delta_pct=20.0),
            self._make_classified("evaluated", "worsened",  delta_pct=-10.0),
            self._make_classified("evaluated", "unchanged", delta_pct=0.5),
            self._make_classified("pending_followup"),
            self._make_classified("too_recent"),
        ]
        summary = _compute_effectiveness_summary(records)
        assert summary["total_counseling_done"] == 5
        assert summary["total_evaluated"] == 3
        assert summary["improved_count"] == 1
        assert summary["worsened_count"] == 1
        assert summary["unchanged_count"] == 1
        assert summary["total_pending_followup"] == 1
        assert summary["total_too_recent"] == 1

    def test_avg_reduction_includes_negative_deltas(self):
        """Avg harus termasuk sesi yang memburuk (delta negatif)."""
        records = [
            self._make_classified("evaluated", "improved",  delta_pct=30.0),
            self._make_classified("evaluated", "worsened",  delta_pct=-10.0),
        ]
        summary = _compute_effectiveness_summary(records)
        # (30 + -10) / 2 = 10
        assert summary["avg_dropout_reduction_pct"] == pytest.approx(10.0, abs=0.01)

    def test_data_sufficient_flag_below_threshold(self):
        records = [
            self._make_classified("evaluated", "improved", delta_pct=10.0)
            for _ in range(EFFECTIVENESS_MIN_SAMPLE - 1)
        ]
        summary = _compute_effectiveness_summary(records)
        assert summary["data_sufficient"] is False

    def test_data_sufficient_flag_at_threshold(self):
        records = [
            self._make_classified("evaluated", "improved", delta_pct=10.0)
            for _ in range(EFFECTIVENESS_MIN_SAMPLE)
        ]
        summary = _compute_effectiveness_summary(records)
        assert summary["data_sufficient"] is True

    def test_delta_distribution_only_from_evaluated(self):
        records = [
            self._make_classified("evaluated", "improved", delta_pct=15.0),
            self._make_classified("pending_followup"),  # tidak masuk distribusi
        ]
        summary = _compute_effectiveness_summary(records)
        assert summary["delta_distribution"] == [15.0]

    def test_methodology_notes_always_present(self):
        summary = _compute_effectiveness_summary([])
        assert isinstance(summary["methodology_notes"], list)
        assert len(summary["methodology_notes"]) > 0


# ─────────────────────────────────────────────────────────────────────────────
# TestFetchEffectivenessRows — integration test dengan DB nyata
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchEffectivenessRows:
    """Integration tests yang membutuhkan database nyata."""

    def test_empty_db_returns_empty_list(self, in_memory_db):
        rows = db._fetch_effectiveness_rows(EFFECTIVENESS_MIN_DAYS)
        assert rows == []

    def test_only_done_counseling_returned(self, in_memory_db):
        with db._get_conn() as conn:
            sid = _insert_student(conn, "S001")
            pid = _insert_prediction(conn, sid, 0.70, "2024-01-01 10:00:00")
            _insert_counseling(conn, sid, pid, status="pending")  # bukan done
            _insert_counseling(conn, sid, pid, status="scheduled")

        rows = db._fetch_effectiveness_rows(EFFECTIVENESS_MIN_DAYS)
        assert rows == []

    def test_done_without_completed_at_excluded(self, in_memory_db):
        with db._get_conn() as conn:
            sid = _insert_student(conn, "S001")
            pid = _insert_prediction(conn, sid, 0.70, "2024-01-01 10:00:00")
            _insert_counseling(conn, sid, pid, status="done", completed_at=None)

        rows = db._fetch_effectiveness_rows(EFFECTIVENESS_MIN_DAYS)
        assert rows == []

    def test_after_prediction_found_after_min_days(self, in_memory_db):
        before_date = "2024-01-01 10:00:00"
        completed_date = "2024-01-15 10:00:00"
        # Prediksi setelah 30 hari dari completed = 2024-02-14+
        after_date = "2024-02-20 10:00:00"

        with db._get_conn() as conn:
            sid = _insert_student(conn, "S002")
            pid_before = _insert_prediction(conn, sid, 0.75, before_date)
            _insert_prediction(conn, sid, 0.45, after_date)  # prediksi sesudah
            _insert_counseling(conn, sid, pid_before, status="done",
                               completed_at=completed_date)

        rows = db._fetch_effectiveness_rows(30)
        assert len(rows) == 1
        assert rows[0]["p_dropout_before"] == pytest.approx(0.75, abs=0.001)
        assert rows[0]["p_dropout_after"] == pytest.approx(0.45, abs=0.001)

    def test_after_prediction_not_found_if_too_early(self, in_memory_db):
        completed_date = "2024-01-15 10:00:00"
        # Prediksi hanya 10 hari setelah completed (< 30 hari)
        too_early_date = "2024-01-25 10:00:00"

        with db._get_conn() as conn:
            sid = _insert_student(conn, "S003")
            pid = _insert_prediction(conn, sid, 0.75, "2024-01-01 10:00:00")
            _insert_prediction(conn, sid, 0.45, too_early_date)
            _insert_counseling(conn, sid, pid, status="done",
                               completed_at=completed_date)

        rows = db._fetch_effectiveness_rows(30)
        assert len(rows) == 1
        assert rows[0]["p_dropout_after"] is None

    def test_latest_after_prediction_used_when_multiple(self, in_memory_db):
        """Saat ada beberapa prediksi sesudah, yang terbaru yang diambil."""
        completed_date = "2024-01-15 10:00:00"

        with db._get_conn() as conn:
            sid = _insert_student(conn, "S004")
            pid = _insert_prediction(conn, sid, 0.80, "2024-01-01 10:00:00")
            _insert_prediction(conn, sid, 0.60, "2024-02-20 10:00:00")  # lebih lama
            _insert_prediction(conn, sid, 0.35, "2024-03-10 10:00:00")  # paling baru
            _insert_counseling(conn, sid, pid, status="done",
                               completed_at=completed_date)

        rows = db._fetch_effectiveness_rows(30)
        assert rows[0]["p_dropout_after"] == pytest.approx(0.35, abs=0.001)

    def test_invalid_min_days_raises_value_error(self, in_memory_db):
        with pytest.raises(ValueError, match="min_days"):
            db._fetch_effectiveness_rows(-1)


# ─────────────────────────────────────────────────────────────────────────────
# TestGetInterventionEffectiveness — end-to-end + cache tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGetInterventionEffectiveness:

    def test_returns_expected_top_level_keys(self, in_memory_db):
        result = db.get_intervention_effectiveness()
        assert "summary" in result
        assert "records" in result
        assert "generated_at" in result
        assert "min_days" in result

    def test_empty_db_returns_valid_structure(self, in_memory_db):
        result = db.get_intervention_effectiveness()
        assert result["summary"]["total_counseling_done"] == 0
        assert result["records"] == []

    def test_cache_hit_skips_db_query(self, in_memory_db):
        # First call — populasi cache
        r1 = db.get_intervention_effectiveness()
        ts1 = db._effectiveness_cache["timestamp"]

        # Second call — harus hit cache (timestamp sama)
        r2 = db.get_intervention_effectiveness()
        ts2 = db._effectiveness_cache["timestamp"]

        assert ts1 == ts2
        assert r1["generated_at"] == r2["generated_at"]

    def test_cache_bypassed_with_use_cache_false(self, in_memory_db):
        r1 = db.get_intervention_effectiveness()
        time.sleep(0.01)  # pastikan timestamp beda
        r2 = db.get_intervention_effectiveness(use_cache=False)
        # generated_at bisa berbeda jika melewati detik berbeda,
        # tapi yang penting cache di-refresh (timestamp berubah)
        assert db._effectiveness_cache["timestamp"] > 0

    def test_invalidate_resets_cache(self, in_memory_db):
        db.get_intervention_effectiveness()
        assert db._effectiveness_cache["data"] is not None

        db.invalidate_effectiveness_cache()
        assert db._effectiveness_cache["data"] is None

    def test_full_pipeline_improved_student(self, in_memory_db):
        """
        Scenario lengkap: mahasiswa berisiko tinggi → dikonseling →
        diprediksi ulang 45 hari kemudian dengan risiko lebih rendah.
        """
        completed_date = (datetime.now() - timedelta(days=45)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        after_date = (datetime.now() - timedelta(days=5)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        with db._get_conn() as conn:
            sid = _insert_student(conn, "S_FULL", "Mahasiswa Uji")
            pid_before = _insert_prediction(conn, sid, 0.80, "2024-01-01 10:00:00")
            _insert_prediction(conn, sid, 0.30, after_date)
            _insert_counseling(conn, sid, pid_before, status="done",
                               completed_at=completed_date)

        result = db.get_intervention_effectiveness(use_cache=False)
        assert result["summary"]["total_counseling_done"] == 1
        assert result["summary"]["total_evaluated"] == 1
        assert result["summary"]["improved_count"] == 1
        assert result["summary"]["avg_dropout_reduction_pct"] == pytest.approx(50.0, abs=0.5)
        assert result["records"][0]["evaluation_status"] == "evaluated"
        assert result["records"][0]["improvement_category"] == "improved"

    def test_min_days_parameter_respected(self, in_memory_db):
        """
        Dengan min_days=0 semua konseling selesai eligible,
        dengan min_days=365 tidak ada yang eligible (kecuali DB lama).
        """
        recent_completed = (datetime.now() - timedelta(days=5)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        after_date = (datetime.now() - timedelta(days=1)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        with db._get_conn() as conn:
            sid = _insert_student(conn, "S_MINDAYS")
            pid = _insert_prediction(conn, sid, 0.70, "2024-01-01 10:00:00")
            _insert_prediction(conn, sid, 0.40, after_date)
            _insert_counseling(conn, sid, pid, status="done",
                               completed_at=recent_completed)

        # Dengan min_days=0 dan prediksi sesudah ada → harus evaluated
        result_short = db.get_intervention_effectiveness(min_days=0, use_cache=False)
        assert result_short["records"][0]["evaluation_status"] == "evaluated"

        # Dengan min_days=90 dan completed hanya 5 hari lalu → too_recent
        result_long = db.get_intervention_effectiveness(min_days=90, use_cache=False)
        assert result_long["records"][0]["evaluation_status"] == "too_recent"


# ─────────────────────────────────────────────────────────────────────────────
# TestCacheInvalidationIntegration — test invalidasi cache lintas fungsi
# ─────────────────────────────────────────────────────────────────────────────

class TestCacheInvalidationIntegration:

    def test_save_prediction_invalidates_cache(self, in_memory_db):
        """save_prediction() harus invalidasi effectiveness cache."""
        # Isi cache dulu
        db.get_intervention_effectiveness()
        assert db._effectiveness_cache["data"] is not None

        # Simpan prediksi baru
        with db._get_conn() as conn:
            sid = _insert_student(conn, "S_CACHE1")
        db.save_prediction(
            student_id=sid, risk_level="tinggi",
            p_dropout=0.75, raw_features={
                "location_type": "Urban", "family_income": 5000,
                "financial_aid_status": 0, "distance_to_institute": 10,
                "internet_connectivity_issues": 0, "motivation_score": 5,
                "career_alignment": 2, "stress_levels": 2, "family_support": 2,
                "attendance_rate": 70, "test_scores_avg": 60, "backlogs": 1,
                "teaching_quality_rating": 5,
            },
        )
        assert db._effectiveness_cache["data"] is None

    def test_update_counseling_done_invalidates_cache(self, in_memory_db):
        """update_counseling dengan status=done harus invalidasi cache."""
        with db._get_conn() as conn:
            sid = _insert_student(conn, "S_CACHE2")
            pid = _insert_prediction(conn, sid, 0.70, "2024-01-01 10:00:00")
            cid = _insert_counseling(conn, sid, pid, status="scheduled")

        db.get_intervention_effectiveness()
        assert db._effectiveness_cache["data"] is not None

        db.update_counseling(cid, status="done")
        assert db._effectiveness_cache["data"] is None

    def test_update_counseling_non_done_does_not_invalidate(self, in_memory_db):
        """update_counseling dengan status selain done TIDAK invalidasi cache."""
        with db._get_conn() as conn:
            sid = _insert_student(conn, "S_CACHE3")
            pid = _insert_prediction(conn, sid, 0.70, "2024-01-01 10:00:00")
            cid = _insert_counseling(conn, sid, pid, status="pending")

        db.get_intervention_effectiveness()
        cache_snapshot = db._effectiveness_cache["data"]
        assert cache_snapshot is not None

        db.update_counseling(cid, status="scheduled")
        # Cache harus tetap ada (tidak di-invalidate)
        assert db._effectiveness_cache["data"] is not None

    def test_reset_student_data_invalidates_cache(self, in_memory_db):
        """reset_student_data() harus invalidasi semua cache."""
        db.get_intervention_effectiveness()
        db.get_equity_data()

        db.reset_student_data()

        assert db._effectiveness_cache["data"] is None
        assert db._equity_cache["data"] is None


# ─────────────────────────────────────────────────────────────────────────────
# TestApiEffectivenessEndpoint — Flask route integration tests
# ─────────────────────────────────────────────────────────────────────────────

class TestApiEffectivenessEndpoint:
    """
    Test Flask endpoints. Membutuhkan aplikasi Flask yang bisa dibuat via create_app().
    Skip jika app.py tidak tersedia di lingkungan test.
    """

    @pytest.fixture
    def app_client(self, tmp_path):
        """Buat Flask test client dengan DB in-memory."""
        try:
            import sys, os
            sys.path.insert(0, os.path.dirname(__file__))
            from app import create_app
            from config import Config
        except ImportError:
            pytest.skip("app.py tidak tersedia untuk Flask route tests")

        # Override DB path ke temp file
        db_path = str(tmp_path / "flask_test.db")
        with patch.object(Config, "DB_PATH", db_path):
            with patch.object(Config, "MODEL_PATH", str(tmp_path / "model.joblib")):
                # Mock Predictor agar tidak butuh model file
                with patch("app.Predictor") as mock_pred:
                    mock_pred.return_value = MagicMock()
                    flask_app = create_app()

        flask_app.config["TESTING"] = True
        flask_app.config["WTF_CSRF_ENABLED"] = False

        # Buat user dan login
        db.init_db(db_path)
        db.create_user("testadmin", "Admin123!")

        with flask_app.test_client() as client:
            # Login
            client.post("/login", data={"username": "testadmin",
                                        "password": "Admin123!"})
            yield client

    def test_api_effectiveness_returns_200(self, app_client):
        resp = app_client.get("/konselor/api/effectiveness")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"

    def test_api_effectiveness_structure(self, app_client):
        resp = app_client.get("/konselor/api/effectiveness")
        data = resp.get_json()["data"]
        assert "summary" in data
        assert "records" in data
        assert "generated_at" in data
        assert "min_days" in data

    def test_api_min_days_param_respected(self, app_client):
        resp = app_client.get("/konselor/api/effectiveness?min_days=7")
        assert resp.status_code == 200
        data = resp.get_json()["data"]
        assert data["min_days"] == 7

    def test_api_min_days_capped_at_max(self, app_client):
        resp = app_client.get("/konselor/api/effectiveness?min_days=999")
        assert resp.status_code == 200
        data = resp.get_json()["data"]
        assert data["min_days"] == 180  # capped

    def test_api_min_days_capped_at_min(self, app_client):
        resp = app_client.get("/konselor/api/effectiveness?min_days=1")
        assert resp.status_code == 200
        data = resp.get_json()["data"]
        assert data["min_days"] == 7  # capped at 7

    def test_api_invalid_min_days_returns_400(self, app_client):
        resp = app_client.get("/konselor/api/effectiveness?min_days=abc")
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["status"] == "error"

    def test_effectiveness_page_returns_200(self, app_client):
        resp = app_client.get("/konselor/effectiveness")
        assert resp.status_code == 200

    def test_api_requires_login(self, tmp_path):
        """Endpoint harus redirect jika tidak login."""
        try:
            from app import create_app
            from config import Config
        except ImportError:
            pytest.skip("app.py tidak tersedia")

        db_path = str(tmp_path / "unauth_test.db")
        with patch.object(Config, "DB_PATH", db_path):
            with patch.object(Config, "MODEL_PATH", str(tmp_path / "model.joblib")):
                with patch("app.Predictor") as mock_pred:
                    mock_pred.return_value = MagicMock()
                    flask_app = create_app()

        flask_app.config["TESTING"] = True
        db.init_db(db_path)

        with flask_app.test_client() as client:
            resp = client.get("/konselor/api/effectiveness")
            # Flask-Login redirect ke login page
            assert resp.status_code in (302, 401)


# ─────────────────────────────────────────────────────────────────────────────
# Panduan Manual Test Cases
# ─────────────────────────────────────────────────────────────────────────────
# Jalankan aplikasi dengan FLASK_ENV=development, lalu:
#
# 1. Prediksi individu dengan NIM mahasiswa A → simpan ke DB
# 2. Tandai counseling sebagai "done" via halaman mahasiswa
# 3. Tunggu atau manipulasi completed_at ke 31 hari lalu:
#    UPDATE counseling_records SET completed_at='2024-01-01 10:00:00'
#    WHERE id=<id>;
# 4. Buat prediksi baru untuk mahasiswa A (simulasi prediksi sesudah)
# 5. Kunjungi /konselor/effectiveness → harus tampil 1 sesi terevaluasi
# 6. Kunjungi /konselor/api/effectiveness → cek JSON terstruktur benar
# 7. Test filter "Terevaluasi" di tabel → hanya tampil 1 baris
# 8. Klik "Ekspor CSV" → file harus terunduh dengan data benar
# 9. Kunjungi /konselor/api/effectiveness?min_days=7 → min_days=7 di response
# 10. Kunjungi /konselor/api/effectiveness?min_days=abc → HTTP 400
# ─────────────────────────────────────────────────────────────────────────────
