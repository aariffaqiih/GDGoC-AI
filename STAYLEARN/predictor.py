"""
predictor.py — Module for dropout risk prediction using logistic regression.

Konsep kunci:
- Menggunakan pipeline yang sudah dilatih (staylearn_model.joblib)
- Menerima input individu atau batch CSV
- Menghasilkan probabilitas bertahan/dropout, tingkat risiko, dan analisis faktor
- Tidak lagi menggunakan cache prediksi individu untuk menghindari memory leak
"""

import os
import random
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
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
]

RANDOM_RANGES = {
    "location_type": ["Urban", "Rural", "Semi-urban"],
    "family_income": (2000, 50000, False),
    "financial_aid_status": (0, 2, True),
    "distance_to_institute": (0.5, 50.0, False),
    "internet_connectivity_issues": (0, 2, True),
    "motivation_score": (1, 10, True),
    "career_alignment": (1, 3, True),
    "stress_levels": (1, 3, True),
    "family_support": (1, 3, True),
    "attendance_rate": (10.0, 99.7, False),
    "test_scores_avg": (35.0, 100.0, False),
    "backlogs": (0, 9, True),
    "teaching_quality_rating": (1, 10, True),
}

ALLOWED_LOCATION_TYPES = {"Urban", "Rural", "Semi-urban"}

NUMERIC_RANGES = {
    "family_income": (2000, 50000),
    "financial_aid_status": (0, 2),
    "distance_to_institute": (0.5, 50),
    "internet_connectivity_issues": (0, 2),
    "motivation_score": (1, 10),
    "career_alignment": (1, 3),
    "stress_levels": (1, 3),
    "family_support": (1, 3),
    "attendance_rate": (10, 100),
    "test_scores_avg": (35, 100),
    "backlogs": (0, 9),
    "teaching_quality_rating": (1, 10),
}

RISK_THRESHOLDS = {
    "rendah": 0.60,   # p_stay >= 60% -> rendah
    "sedang": 0.30,   # 30% <= p_stay < 60% -> sedang
    # di bawah 30% -> tinggi
}

def _classify_risk(p_stay: float) -> tuple:
    if p_stay >= RISK_THRESHOLDS["rendah"]:
        return "rendah", "Risiko Rendah"
    if p_stay >= RISK_THRESHOLDS["sedang"]:
        return "sedang", "Risiko Sedang"
    return "tinggi", "Risiko Tinggi"

def _factor_analysis(data: dict) -> dict:
    concerns: list[str] = []
    strengths: list[str] = []

    attendance = float(data.get("attendance_rate", 100))
    if attendance < 60:
        concerns.append(f"Kehadiran sangat rendah ({attendance:.1f}%)")
    elif attendance < 80:
        concerns.append(f"Kehadiran perlu ditingkatkan ({attendance:.1f}%)")
    else:
        strengths.append(f"Kehadiran baik ({attendance:.1f}%)")

    scores = float(data.get("test_scores_avg", 100))
    if scores < 50:
        concerns.append(f"Nilai rata-rata di bawah standar ({scores:.1f})")
    elif scores < 70:
        concerns.append(f"Nilai rata-rata perlu ditingkatkan ({scores:.1f})")
    else:
        strengths.append(f"Nilai rata-rata memuaskan ({scores:.1f})")

    backlogs = int(data.get("backlogs", 0))
    if backlogs >= 3:
        concerns.append(f"Banyak mata kuliah tertunggak ({backlogs} MK)")
    elif backlogs > 0:
        concerns.append(f"Ada mata kuliah tertunggak ({backlogs} MK)")
    else:
        strengths.append("Tidak ada mata kuliah tertunggak")

    motivation = int(data.get("motivation_score", 10))
    if motivation <= 3:
        concerns.append(f"Motivasi belajar sangat rendah ({motivation}/10)")
    elif motivation <= 5:
        concerns.append(f"Motivasi belajar rendah ({motivation}/10)")
    elif motivation >= 8:
        strengths.append(f"Motivasi belajar tinggi ({motivation}/10)")

    stress = int(data.get("stress_levels", 1))
    if stress == 3:
        concerns.append("Tingkat stres sangat tinggi")
    elif stress == 2:
        concerns.append("Tingkat stres sedang, perlu dipantau")
    else:
        strengths.append("Tingkat stres terkendali")

    support = int(data.get("family_support", 3))
    if support == 1:
        concerns.append("Dukungan keluarga sangat kurang")
    elif support == 2:
        concerns.append("Dukungan keluarga cukup, dapat ditingkatkan")
    else:
        strengths.append("Dukungan keluarga kuat")

    income = float(data.get("family_income", 50000))
    if income < 5000:
        concerns.append(f"Pendapatan keluarga sangat terbatas (Rp {int(income):,})")
    elif income < 10000:
        concerns.append(f"Pendapatan keluarga terbatas (Rp {int(income):,})")

    aid = int(data.get("financial_aid_status", 0))
    if aid == 2:
        strengths.append("Mendapat beasiswa / bantuan keuangan penuh")
    elif aid == 1:
        strengths.append("Mendapat bantuan keuangan sebagian")
    else:
        if income < 10000:
            concerns.append("Tidak ada bantuan keuangan meski pendapatan terbatas")

    internet = int(data.get("internet_connectivity_issues", 0))
    if internet == 2:
        concerns.append("Sering mengalami masalah koneksi internet")
    elif internet == 1:
        concerns.append("Kadang mengalami masalah koneksi internet")

    alignment = int(data.get("career_alignment", 3))
    if alignment == 1:
        concerns.append("Kesesuaian karir yang dipilih sangat rendah")
    elif alignment == 2:
        concerns.append("Kesesuaian karir cukup, perlu eksplorasi lebih lanjut")
    else:
        strengths.append("Pilihan karir sesuai dengan bidang studi")

    distance = float(data.get("distance_to_institute", 0))
    if distance > 30:
        concerns.append(f"Jarak ke kampus sangat jauh ({distance:.1f} km)")
    elif distance > 15:
        concerns.append(f"Jarak ke kampus cukup jauh ({distance:.1f} km)")

    teaching = int(data.get("teaching_quality_rating", 10))
    if teaching <= 3:
        concerns.append(f"Kualitas pengajaran dinilai sangat rendah ({teaching}/10)")
    elif teaching <= 5:
        concerns.append(f"Kualitas pengajaran dinilai kurang memuaskan ({teaching}/10)")
    elif teaching >= 8:
        strengths.append(f"Kualitas pengajaran dinilai baik ({teaching}/10)")

    if data.get("location_type") == "Urban" and distance > 30:
        concerns.append("Lokasi perkotaan namun jarak ke kampus sangat jauh — periksa kembali data.")
    if aid == 2 and income > 30000:
        concerns.append("Pendapatan tinggi namun menerima beasiswa penuh — periksa kembali data.")

    return {"concerns": concerns, "strengths": strengths}

def _get_recommendation(risk_level: str) -> str:
    if risk_level == "tinggi":
        return (
            "Mahasiswa ini memerlukan perhatian segera. Segera jadwalkan sesi "
            "konseling personal dan koordinasikan dengan dosen wali untuk merancang "
            "rencana pendampingan yang terstruktur dan berkelanjutan."
        )
    if risk_level == "sedang":
        return (
            "Pantau perkembangan mahasiswa ini secara berkala. Pertimbangkan untuk "
            "mengadakan pertemuan konsultasi rutin dan memberikan dukungan motivasi "
            "tambahan agar potensi risiko tidak meningkat."
        )
    return (
        "Profil mahasiswa ini menunjukkan stabilitas yang baik. Pertahankan kondisi "
        "ini dengan memastikan lingkungan belajar yang mendukung dan komunikasi "
        "terbuka antara mahasiswa, dosen, dan keluarga."
    )


class Predictor:
    def __init__(self, model_path: str, max_batch_rows: int = 10000):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model tidak ditemukan di: {model_path}. "
                f"Pastikan file staylearn_model.joblib ada di folder models/"
            )
        self._model_path = model_path
        self._model = None  # dimuat secara lazy saat prediksi pertama
        self.max_batch_rows = max_batch_rows
        logger.info("Predictor siap (model akan dimuat saat prediksi pertama)")

    @property
    def model(self):
        """Muat model dari disk hanya saat pertama kali dibutuhkan."""
        if self._model is None:
            logger.info("Memuat model dari %s ...", self._model_path)
            try:
                self._model = joblib.load(self._model_path)
                logger.info("Model berhasil dimuat")
            except Exception as exc:
                logger.error("Gagal memuat model: %s", exc)
                raise RuntimeError(
                    f"Gagal memuat model. Pastikan scikit-learn versi "
                    f"{sklearn_version} kompatibel."
                ) from exc
        return self._model

    def reload_model(self):
        """
        Force reload model from disk and clear any internal caches.
        Digunakan setelah model diperbarui (misal retraining).
        """
        self._model = None
        # Muat ulang model
        _ = self.model
        logger.info("Model reloaded")

    def _validate_location(self, location: str) -> None:
        if location not in ALLOWED_LOCATION_TYPES:
            raise ValueError(
                f"location_type tidak valid: '{location}'. "
                f"Harus salah satu dari {sorted(ALLOWED_LOCATION_TYPES)}"
            )

    def _validate_numeric_range(self, key: str, value) -> None:
        if key in NUMERIC_RANGES:
            lo, hi = NUMERIC_RANGES[key]
            try:
                num_val = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{key} harus berupa angka, namun mendapat {value}")
            if not (lo <= num_val <= hi):
                raise ValueError(
                    f"{key} harus antara {lo} dan {hi}, namun mendapat {num_val}"
                )

    def _validate_cross_field(self, data: dict) -> None:
        loc = data.get("location_type")
        dist = data.get("distance_to_institute", 0)
        if loc == "Urban" and dist > 30:
            logger.warning("Kombinasi tidak wajar: Urban dengan jarak %.1f km", dist)
        aid = data.get("financial_aid_status", 0)
        income = data.get("family_income", 0)
        if aid == 2 and income > 30000:
            logger.warning("Kombinasi tidak wajar: beasiswa penuh dengan pendapatan tinggi %.0f", income)

    def _validate_single(self, data: dict) -> None:
        self._validate_location(data["location_type"])
        for key, val in data.items():
            if key in NUMERIC_RANGES:
                self._validate_numeric_range(key, val)
        self._validate_cross_field(data)

    def _normalize_types(self, data: dict) -> dict:
        return {
            "location_type": str(data["location_type"]),
            "family_income": float(data["family_income"]),
            "financial_aid_status": int(data["financial_aid_status"]),
            "distance_to_institute": float(data["distance_to_institute"]),
            "internet_connectivity_issues": int(data["internet_connectivity_issues"]),
            "motivation_score": int(data["motivation_score"]),
            "career_alignment": int(data["career_alignment"]),
            "stress_levels": int(data["stress_levels"]),
            "family_support": int(data["family_support"]),
            "attendance_rate": float(data["attendance_rate"]),
            "test_scores_avg": float(data["test_scores_avg"]),
            "backlogs": int(data["backlogs"]),
            "teaching_quality_rating": int(data["teaching_quality_rating"]),
        }

    def _to_dataframe(self, data: dict) -> pd.DataFrame:
        self._validate_single(data)
        return pd.DataFrame([data])

    def predict_single(self, data: dict) -> dict:
        """
        Prediksi untuk satu mahasiswa (tanpa caching).

        Args:
            data: dictionary fitur (sama seperti form input)

        Returns:
            dict dengan keys: p_stay, p_dropout, risk_level, risk_label,
                              concerns, strengths, recommendation
        """
        normalised = self._normalize_types(data)
        feature_dict = {k: v for k, v in normalised.items() if k in FEATURE_NAMES}
        df = pd.DataFrame([feature_dict])
        proba = self.model.predict_proba(df)[0]
        p_stay = float(proba[0])
        p_dropout = float(proba[1])
        risk_level, risk_label = _classify_risk(p_stay)
        factors = _factor_analysis(data)
        recommendation = _get_recommendation(risk_level)
        return {
            "p_stay": round(p_stay * 100, 1),
            "p_dropout": round(p_dropout * 100, 1),
            "risk_level": risk_level,
            "risk_label": risk_label,
            "concerns": factors["concerns"],
            "strengths": factors["strengths"],
            "recommendation": recommendation,
        }

    def predict_batch(self, df: pd.DataFrame) -> list:
        results, _ = self.predict_batch_with_errors(df)
        return results

    def predict_batch_with_errors(self, df: pd.DataFrame):
        if len(df) > self.max_batch_rows:
            raise ValueError(
                f"File terlalu besar: {len(df)} baris. "
                f"Maksimal {self.max_batch_rows} baris."
            )

        missing = set(FEATURE_NAMES) - set(df.columns)
        if missing:
            raise ValueError(
                f"Kolom wajib tidak ditemukan: {', '.join(sorted(missing))}. "
                f"Unduh template CSV untuk mendapatkan format yang benar."
            )

        features = df[FEATURE_NAMES].copy()
        features["location_type"] = features["location_type"].astype(str)

        valid_indices: list = []
        error_details: list = []
        valid_rows: list = []

        for idx, row in features.iterrows():
            try:
                data = row.to_dict()
                for key in [
                    "family_income",
                    "distance_to_institute",
                    "attendance_rate",
                    "test_scores_avg",
                ]:
                    data[key] = float(data[key])
                for key in NUMERIC_RANGES:
                    if key not in (
                        "family_income",
                        "distance_to_institute",
                        "attendance_rate",
                        "test_scores_avg",
                    ):
                        data[key] = int(float(data[key]))
                    self._validate_numeric_range(key, data[key])
                self._validate_location(data["location_type"])
                self._validate_cross_field(data)
                valid_rows.append(data)
                valid_indices.append(idx)
            except (ValueError, TypeError) as exc:
                error_details.append(
                    {"row": int(idx), "error": str(exc), "data": row.to_dict()}
                )

        if not valid_rows:
            return [], error_details

        valid_df = pd.DataFrame(valid_rows)[FEATURE_NAMES]
        probas = self.model.predict_proba(valid_df)

        results: list = []
        for i, (orig_idx, row_data) in enumerate(zip(valid_indices, valid_rows)):
            p_stay = float(probas[i][0])
            p_dropout = float(probas[i][1])
            risk_level, risk_label = _classify_risk(p_stay)
            orig_row = df.loc[orig_idx].to_dict()
            result = {k: orig_row.get(k, "") for k in df.columns}
            result.update(
                {
                    "kemungkinan_bertahan_pct": round(p_stay * 100, 1),
                    "kemungkinan_dropout_pct": round(p_dropout * 100, 1),
                    "tingkat_risiko": risk_label,
                }
            )
            results.append(result)

        return results, error_details

    def generate_random(self) -> dict:
        result: dict = {}
        for key, val in RANDOM_RANGES.items():
            if isinstance(val, list):
                result[key] = random.choice(val)
            else:
                lo, hi, is_int = val
                if is_int:
                    result[key] = random.randint(int(lo), int(hi))
                else:
                    result[key] = round(random.uniform(lo, hi), 1)
        return result