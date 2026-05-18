import pytest
from database import init_db, upsert_student, verify_student_birth_date, get_prediction_history_by_nim
from config import Config
import tempfile
import os
import sqlite3

@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    init_db(path)
    yield path
    os.unlink(path)

def test_upsert_student_with_birth_date(temp_db):
    student_id = upsert_student("12345", "Budi", "2000-01-01")
    assert student_id > 0
    # Verifikasi birth_date tersimpan
    from database import _get_conn
    with _get_conn() as conn:
        row = conn.execute("SELECT birth_date FROM students WHERE nim = ?", ("12345",)).fetchone()
        assert row["birth_date"] == "2000-01-01"

def test_verify_student_birth_date_success(temp_db):
    upsert_student("12345", "Budi", "2000-01-01")
    assert verify_student_birth_date("12345", "2000-01-01") is True

def test_verify_student_birth_date_wrong(temp_db):
    upsert_student("12345", "Budi", "2000-01-01")
    assert verify_student_birth_date("12345", "1999-12-31") is False

def test_verify_student_birth_date_no_birth_date(temp_db):
    upsert_student("12345", "Budi")  # tanpa birth_date
    assert verify_student_birth_date("12345", "2000-01-01") is False

def test_get_prediction_history_empty(temp_db):
    upsert_student("12345", "Budi")
    history = get_prediction_history_by_nim("12345")
    assert history == []

def test_get_prediction_history_ordered(temp_db):
    from database import save_prediction
    student_id = upsert_student("12345", "Budi", "2000-01-01")
    save_prediction(student_id, "tinggi", 0.8, {"test": 1}, source="individual")
    save_prediction(student_id, "sedang", 0.5, {"test": 2}, source="individual")
    history = get_prediction_history_by_nim("12345")
    assert len(history) == 2
    # Pastikan urutan ASC (terlama ke terbaru)
    assert history[0]["p_dropout"] == 0.8
    assert history[1]["p_dropout"] == 0.5