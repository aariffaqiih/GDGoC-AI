"""
generate_dummy_data.py - FIXED VERSION (tanpa weights error)
============================================
"""

import sqlite3
import random
import json
from datetime import datetime, timedelta
import sys

# ==============================================
# KONFIGURASI DATABASE
# ==============================================
DB_PATH = "data/staylearn.db"

# ==============================================
# PARAMETER GENERASI DATA
# ==============================================
NUM_STUDENTS = 500
NUM_PREDICTIONS_PER_STUDENT = (4, 10)
NUM_COUNSELING = 400
NUM_MENTORS = 30
NUM_MENTOR_SESSIONS = 200
NUM_INFRA_REPORTS = 100

# Thresholds
RISK_THRESH = {"rendah": 0.60, "sedang": 0.30}
TRAJECTORY_THRESHOLD = 0.05

# ==============================================
# FUNGSI BANTU (diperbaiki)
# ==============================================
def random_date(start, end):
    delta = end - start
    int_delta = int(delta.total_seconds())
    random_second = random.randint(0, max(1, int_delta))
    return start + timedelta(seconds=random_second)

def random_float(min_val, max_val, decimals=1):
    return round(random.uniform(min_val, max_val), decimals)

def random_int(min_val, max_val):
    return random.randint(min_val, max_val)

def random_choice(options):
    """Memilih satu elemen secara random dari list."""
    return random.choice(options)

def random_choice_weighted(options, weights):
    """Memilih dengan bobot (weighted random)."""
    return random.choices(options, weights=weights, k=1)[0]

def random_bool():
    return random.choice([0, 1])

def classify_risk(p_stay):
    if p_stay >= RISK_THRESH["rendah"]:
        return "rendah"
    elif p_stay >= RISK_THRESH["sedang"]:
        return "sedang"
    else:
        return "tinggi"

def compute_trajectory(prev_p_dropout, new_p_dropout):
    if prev_p_dropout is None:
        return "baru"
    diff = new_p_dropout - prev_p_dropout
    if diff > TRAJECTORY_THRESHOLD:
        return "memburuk"
    elif diff < -TRAJECTORY_THRESHOLD:
        return "membaik"
    else:
        return "stagnan"

def generate_random_features(risk_bias=None):
    """Menghasilkan raw_features dengan bias tertentu."""
    if risk_bias == "tinggi":
        mot = random_int(1, 4)
        attendance = random_float(30, 65, 1)
        scores = random_float(40, 60, 1)
        backlogs = random_int(3, 8)
        stress = random_int(2, 3)
        support = random_int(1, 2)
        teaching = random_int(1, 4)
    elif risk_bias == "sedang":
        mot = random_int(4, 7)
        attendance = random_float(60, 80, 1)
        scores = random_float(55, 75, 1)
        backlogs = random_int(1, 3)
        stress = random_int(1, 2)
        support = random_int(2, 3)
        teaching = random_int(4, 7)
    else:  # rendah atau random
        mot = random_int(7, 10)
        attendance = random_float(80, 98, 1)
        scores = random_float(75, 95, 1)
        backlogs = random_int(0, 1)
        stress = random_int(1, 2)
        support = random_int(2, 3)
        teaching = random_int(7, 10)
    
    loc = random_choice(["Urban", "Semi-urban", "Rural"])
    income = random_int(2000, 50000)
    aid = random_int(0, 2)
    distance = round(random.uniform(0.5, 50.0), 1)
    internet = random_int(0, 2)
    career = random_int(1, 3)

    return {
        "location_type": loc,
        "family_income": income,
        "financial_aid_status": aid,
        "distance_to_institute": distance,
        "internet_connectivity_issues": internet,
        "motivation_score": mot,
        "career_alignment": career,
        "stress_levels": stress,
        "family_support": support,
        "attendance_rate": attendance,
        "test_scores_avg": scores,
        "backlogs": backlogs,
        "teaching_quality_rating": teaching,
    }

def simulate_p_dropout(features):
    """Hitung p_dropout dari features secara heuristic."""
    p_stay = 0.5
    p_stay += features["motivation_score"] * 0.025
    p_stay += features["attendance_rate"] / 100 * 0.2
    p_stay += features["test_scores_avg"] / 100 * 0.2
    p_stay -= features["backlogs"] * 0.06
    p_stay -= features["stress_levels"] * 0.08
    p_stay -= (features["distance_to_institute"] / 100) * 0.1
    if features["internet_connectivity_issues"] == 2:
        p_stay -= 0.08
    if features["family_support"] == 1:
        p_stay -= 0.05
    if features["teaching_quality_rating"] <= 3:
        p_stay -= 0.07
    p_stay = max(0.01, min(0.99, p_stay))
    return 1 - p_stay

# ==============================================
# MEMBUAT KONEKSI DATABASE
# ==============================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn

# ==============================================
# GENERATE DATA
# ==============================================
def main():
    print("=" * 50)
    print("GENERATE DATA DUMMY STAYLEARN")
    print("=" * 50)
    
    conn = init_db()
    cur = conn.cursor()
    
    # Hapus data lama dengan urutan yang benar
    print("\n1. Menghapus data lama...")
    cur.execute("DELETE FROM mentor_sessions")
    cur.execute("DELETE FROM mentors")
    cur.execute("DELETE FROM infrastructure_reports")
    cur.execute("DELETE FROM counseling_records")
    cur.execute("DELETE FROM predictions")
    cur.execute("DELETE FROM students")
    cur.execute("DELETE FROM users")
    cur.execute("DELETE FROM sqlite_sequence")
    conn.commit()
    print("   - Data lama berhasil dihapus")
    
    # Buat user konselor
    print("\n2. Membuat user konselor...")
    from werkzeug.security import generate_password_hash
    hashed = generate_password_hash("Admin123!")
    cur.execute("INSERT INTO users (username, password) VALUES (?, ?)", ("admin", hashed))
    print("   - User admin (password: Admin123!)")
    
    # ==========================================
    # 1. STUDENTS
    # ==========================================
    print(f"\n3. Menambahkan {NUM_STUDENTS} mahasiswa...")
    student_ids = []
    used_nims = set()
    start_date = datetime(2023, 6, 1)
    end_date = datetime.now()
    
    for i in range(NUM_STUDENTS):
        while True:
            year = random_choice([2022, 2023, 2024])
            nim = f"{year}{random_int(1000, 9999)}"
            if nim not in used_nims:
                used_nims.add(nim)
                break
        
        nama = f"Mahasiswa {nim[-4:]}"
        birth_date = f"{random_int(1995, 2005)}-{random_int(1,12):02d}-{random_int(1,28):02d}"
        created_at = random_date(start_date, end_date).strftime("%Y-%m-%d %H:%M:%S")
        actual_dropout = 1 if random.random() < 0.08 else 0
        
        gender = random_choice(["L", "P", None])
        disability = 1 if random.random() < 0.05 else 0
        indigenous = 1 if random.random() < 0.03 else 0
        conflict = 1 if random.random() < 0.04 else 0
        
        cur.execute("""
            INSERT INTO students 
            (nim, nama, created_at, actual_dropout, birth_date,
             gender, disability_status, indigenous, conflict_affected)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (nim, nama, created_at, actual_dropout, birth_date,
              gender, disability, indigenous, conflict))
        student_ids.append(cur.lastrowid)
    
    print(f"   - {len(student_ids)} mahasiswa berhasil ditambahkan")
    
    # ==========================================
    # 2. PREDICTIONS
    # ==========================================
    print("\n4. Menambahkan prediksi (riwayat perkembangan)...")
    all_predictions = []
    total_pred = 0
    
    base_date = datetime.now() - timedelta(days=240)
    
    for sid in student_ids:
        num_pred = random.randint(*NUM_PREDICTIONS_PER_STUDENT)
        trend = random_choice(["improving", "worsening", "volatile", "stable"])
        pred_times = sorted([random_date(base_date, datetime.now()) for _ in range(num_pred)])
        
        prev_p_dropout = None
        for idx, t in enumerate(pred_times):
            if trend == "improving":
                risk_bias = "tinggi" if idx < num_pred//3 else ("sedang" if idx < 2*num_pred//3 else "rendah")
            elif trend == "worsening":
                risk_bias = "rendah" if idx < num_pred//3 else ("sedang" if idx < 2*num_pred//3 else "tinggi")
            elif trend == "volatile":
                risk_bias = random_choice(["rendah", "sedang", "tinggi"])
            else:
                risk_bias = "sedang"
            
            features = generate_random_features(risk_bias)
            p_dropout = simulate_p_dropout(features)
            p_stay = 1 - p_dropout
            risk_level = classify_risk(p_stay)
            source = random_choice(["individual", "batch"])
            trajectory = compute_trajectory(prev_p_dropout, p_dropout)
            raw_json = json.dumps(features)
            
            cur.execute("""
                INSERT INTO predictions
                (student_id, risk_level, p_dropout, raw_features, source, trajectory, predicted_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (sid, risk_level, p_dropout, raw_json, source, trajectory, t.strftime("%Y-%m-%d %H:%M:%S")))
            pred_id = cur.lastrowid
            all_predictions.append((sid, pred_id, p_dropout, t, risk_level, features))
            prev_p_dropout = p_dropout
            total_pred += 1
    
    print(f"   - {total_pred} prediksi berhasil ditambahkan")
    
    # ==========================================
    # 3. COUNSELING RECORDS (diperbaiki)
    # ==========================================
    print("\n5. Menambahkan catatan konseling...")
    eligible = [p for p in all_predictions if p[4] in ("tinggi", "sedang")]
    
    # Pilih secara random tanpa weights
    if len(eligible) > NUM_COUNSELING:
        selected = random.sample(eligible, NUM_COUNSELING)
    else:
        selected = eligible
    
    counseling_added = 0
    for (sid, pred_id, p_dropout, pred_time, risk_level, features) in selected:
        action_type = "konseling" if risk_level == "tinggi" else "monitoring"
        
        # Pilih status dengan distribusi manual
        rand = random.random()
        if rand < 0.2:      # 20% pending
            status = "pending"
            scheduled_at = None
            completed_at = None
        elif rand < 0.5:    # 30% scheduled
            status = "scheduled"
            scheduled_at = random_date(pred_time, pred_time + timedelta(days=14)).strftime("%Y-%m-%d %H:%M:%S")
            completed_at = None
        elif rand < 0.9:    # 40% done
            status = "done"
            scheduled_at = random_date(pred_time, pred_time + timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
            completed_at = random_date(pred_time, pred_time + timedelta(days=21)).strftime("%Y-%m-%d %H:%M:%S")
        else:               # 10% skipped
            status = "skipped"
            scheduled_at = None
            completed_at = None
        
        notes_options = [
            f"Mahasiswa memiliki motivasi {features['motivation_score']}/10, perlu pendampingan",
            f"Kendala utama: jarak {features['distance_to_institute']}km dan kehadiran {features['attendance_rate']}%",
            f"Disarankan mengikuti program mentoring dan bimbingan akademik",
            f"Telah diberikan motivasi dan strategi belajar",
            f"Rencana tindak lanjut: konsultasi rutin setiap 2 minggu",
            f"Orang tua dihubungi, siap mendukung penuh",
            f"Mahasiswa mulai menunjukkan perbaikan dalam 2 minggu terakhir"
        ]
        notes = random_choice(notes_options)
        intervention_type = random_choice(["motivasi", "bimbingan_akademik", "bantuan_finansial", 
                                           "pengembangan_karir", "kesehatan_mental", "umum"])
        
        cur.execute("""
            INSERT INTO counseling_records
            (student_id, prediction_id, action_type, status, scheduled_at, completed_at, notes, intervention_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (sid, pred_id, action_type, status, scheduled_at, completed_at, notes, intervention_type))
        counseling_added += 1
    
    print(f"   - {counseling_added} catatan konseling berhasil ditambahkan")
    
    # ==========================================
    # 4. INFRASTRUCTURE REPORTS
    # ==========================================
    print("\n6. Menambahkan laporan infrastruktur...")
    campuses = ["Kampus Utama - Purwokerto", "Kampus B - Banyumas", "Gedung A - Informatika", 
                "Gedung B - Bisnis", "Kampus C - Purbalingga"]
    
    infra_added = 0
    for _ in range(NUM_INFRA_REPORTS):
        campus = random_choice(campuses)
        faculty = random_choice(["Fakultas Informatika", "Fakultas Ekonomi", "Fakultas Teknik", None])
        reported_by = random_choice(student_ids) if random.random() < 0.3 else None
        
        electricity = random_bool() or random.random() < 0.9
        clean_water = random_bool() or random.random() < 0.85
        sanitation = random_bool() or random.random() < 0.8
        separate_wc = random_bool() or random.random() < 0.5
        computers = random_bool() or random.random() < 0.7
        internet = random_bool() or random.random() < 0.75
        disability_access = random_bool() or random.random() < 0.4
        safe_environment = random_bool() or random.random() < 0.85
        
        notes = random_choice([
            "Fasilitas dalam kondisi baik", 
            "Perlu perbaikan AC dan penerangan",
            "Toilet perlu renovasi", 
            "Jaringan internet kadang terputus",
            "Ruang kuliah nyaman dan bersih"
        ])
        
        cur.execute("""
            INSERT INTO infrastructure_reports
            (campus, faculty, reported_by, electricity, clean_water, sanitation,
             separate_wc, computers, internet, disability_access, safe_environment, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (campus, faculty, reported_by, electricity, clean_water, sanitation,
              separate_wc, computers, internet, disability_access, safe_environment, notes))
        infra_added += 1
    
    print(f"   - {infra_added} laporan infrastruktur berhasil ditambahkan")
    
    # ==========================================
    # 5. MENTORS
    # ==========================================
    print("\n7. Menambahkan mentor...")
    mentor_names = ["Dr. Budi Santoso", "Ibu Siti Aminah", "Bapak Agus Wijaya", 
                    "Ibu Dewi Lestari", "Bapak Eko Prasetyo", "Ibu Ratna Sari",
                    "Bapak Hendra Gunawan", "Ibu Mulyani"]
    
    mentor_ids = []
    for i in range(min(NUM_MENTORS, len(mentor_names))):
        name = mentor_names[i]
        nim = f"MNT{2020 + i}{random_int(100, 999)}"
        major = random_choice(["Informatika", "Sistem Informasi", "Manajemen", "Teknik Elektro", "Hukum"])
        availability = json.dumps(random.sample(["Senin 09-11", "Selasa 13-15", "Rabu 10-12", 
                                                  "Kamis 14-16", "Jumat 08-10"], k=random_int(2, 4)))
        max_mentees = random_int(2, 5)
        is_active = random_bool() or random.random() < 0.8
        
        cur.execute("""
            INSERT INTO mentors (name, nim, major, availability_json, max_mentees, is_active)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (name, nim, major, availability, max_mentees, is_active))
        mentor_ids.append(cur.lastrowid)
    
    print(f"   - {len(mentor_ids)} mentor berhasil ditambahkan")
    
    # ==========================================
    # 6. MENTOR SESSIONS
    # ==========================================
    print("\n8. Menambahkan sesi mentoring...")
    session_added = 0
    session_statuses = ["scheduled", "done", "cancelled"]
    
    for _ in range(NUM_MENTOR_SESSIONS):
        mentor_id = random_choice(mentor_ids)
        mentee_id = random_choice(student_ids)
        session_date = random_date(datetime(2024, 1, 1), datetime.now())
        
        # Distribusi status dengan probabilitas manual
        rand = random.random()
        if rand < 0.3:
            status = "scheduled"
            rating = None
        elif rand < 0.9:
            status = "done"
            rating = random_int(3, 5) if random.random() < 0.8 else None
        else:
            status = "cancelled"
            rating = None
        
        notes = random_choice([
            "Membahas kesulitan akademik dan strategi belajar",
            "Konsultasi karir dan persiapan magang",
            "Motivasi dan pengembangan soft skill",
            "Pembahasan tugas akhir dan penelitian"
        ])
        
        cur.execute("""
            INSERT INTO mentor_sessions
            (mentor_id, mentee_id, session_date, notes, rating, status)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (mentor_id, mentee_id, session_date.strftime("%Y-%m-%d %H:%M:%S"), 
              notes, rating, status))
        session_added += 1
    
    print(f"   - {session_added} sesi mentoring berhasil ditambahkan")
    
    # ==========================================
    # COMMIT & CLOSE
    # ==========================================
    conn.commit()
    conn.close()
    
    print("\n" + "=" * 50)
    print("✅ DATA DUMMY BERHASIL DITAMBAHKAN!")
    print("=" * 50)
    print(f"\n📊 STATISTIK:")
    print(f"   • Mahasiswa        : {NUM_STUDENTS} orang")
    print(f"   • Prediksi         : {total_pred} data")
    print(f"   • Konseling        : {counseling_added} sesi")
    print(f"   • Mentor           : {len(mentor_ids)} orang")
    print(f"   • Sesi mentoring   : {session_added} sesi")
    print(f"   • Infrastruktur    : {infra_added} laporan")
    print(f"\n🔐 LOGIN KONSELOR:")
    print(f"   Username: admin")
    print(f"   Password: Admin123!")
    print("\n✨ Sekarang buka browser dan login ke dashboard konselor!")
    print("=" * 50)

if __name__ == "__main__":
    main()