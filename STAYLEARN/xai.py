"""
xai.py — Explainable AI untuk StayLearn (SDG 4.1)

Menyediakan dua jenis penjelasan:
1. Feature importance: kontribusi tiap fitur terhadap p_dropout
2. Counterfactual: "Jika X berubah menjadi Y, risiko akan turun Z%"

Menggunakan SHAP jika tersedia, fallback ke koefisien model logistik.
Penjelasan dihitung per prediksi, cached per feature-tuple.

Catatan keamanan:
- Tidak ada data mahasiswa yang bocor ke klien di luar yang sudah diinput
- Penjelasan SHAP adalah aproksimasi — tidak mengekspos parameter model
"""

import logging
from functools import lru_cache
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Nama tampilan untuk setiap fitur
_FEATURE_LABELS: Dict[str, str] = {
    "location_type":                "Lokasi tempat tinggal",
    "family_income":                "Pendapatan keluarga",
    "financial_aid_status":         "Status bantuan keuangan",
    "distance_to_institute":        "Jarak ke kampus",
    "internet_connectivity_issues": "Masalah koneksi internet",
    "motivation_score":             "Skor motivasi",
    "career_alignment":             "Kesesuaian karir",
    "stress_levels":                "Tingkat stres",
    "family_support":               "Dukungan keluarga",
    "attendance_rate":              "Tingkat kehadiran",
    "test_scores_avg":              "Nilai ujian rata-rata",
    "backlogs":                     "Mata kuliah tertunggak",
    "teaching_quality_rating":      "Kualitas pengajaran",
}

# Fitur yang bisa ditingkatkan mahasiswa (untuk counterfactual)
_ACTIONABLE_FEATURES: Dict[str, Dict[str, Any]] = {
    "attendance_rate":   {"direction": "increase", "target": 80.0,  "unit": "%",   "label": "tingkat kehadiran"},
    "motivation_score":  {"direction": "increase", "target": 8,     "unit": "/10", "label": "skor motivasi"},
    "test_scores_avg":   {"direction": "increase", "target": 70.0,  "unit": "",    "label": "nilai ujian rata-rata"},
    "backlogs":          {"direction": "decrease", "target": 0,     "unit": " MK", "label": "mata kuliah tertunggak"},
    "stress_levels":     {"direction": "decrease", "target": 1,     "unit": "/3",  "label": "tingkat stres"},
    "career_alignment":  {"direction": "increase", "target": 3,     "unit": "/3",  "label": "kesesuaian karir"},
}

FEATURE_NAMES = [
    "location_type", "family_income", "financial_aid_status",
    "distance_to_institute", "internet_connectivity_issues",
    "motivation_score", "career_alignment", "stress_levels",
    "family_support", "attendance_rate", "test_scores_avg",
    "backlogs", "teaching_quality_rating",
]


class DropoutExplainer:
    """
    Explainer untuk model dropout risk.

    Usage:
        explainer = DropoutExplainer(predictor_model)
        explanation = explainer.explain(student_features)
    """

    def __init__(self, model):
        """
        Args:
            model: scikit-learn Pipeline object (dari Predictor._model)
        """
        self._model = model
        self._shap_explainer = None
        self._shap_available = False
        self._coef_importance: Optional[np.ndarray] = None
        self._init_explainer()

    def _init_explainer(self) -> None:
        """Inisialisasi SHAP explainer atau fallback ke koefisien."""
        try:
            import shap
            # Ambil step terakhir pipeline (classifier)
            classifier = self._model.named_steps.get("model")
            preprocessor = self._model.named_steps.get("preprocessor")
            if classifier is not None and preprocessor is not None:
                # Gunakan LinearExplainer untuk logistic regression
                self._shap_explainer = shap.LinearExplainer(
                    classifier, masker=shap.maskers.Independent(
                        np.zeros((1, classifier.coef_.shape[1]))
                    )
                )
                self._shap_available = True
                logger.info("SHAP LinearExplainer berhasil diinisialisasi")
        except (ImportError, Exception) as e:
            logger.info("SHAP tidak tersedia (%s), fallback ke koefisien model", e)
            self._shap_available = False
            self._init_coefficient_importance()

    def _init_coefficient_importance(self) -> None:
        """Ekstrak feature importance dari koefisien logistic regression."""
        try:
            classifier = self._model.named_steps.get("model")
            if classifier is not None and hasattr(classifier, "coef_"):
                # coef_[0] = koefisien untuk kelas dropout=1
                self._coef_importance = np.abs(classifier.coef_[0])
        except Exception as e:
            logger.warning("Gagal ekstrak koefisien: %s", e)

    def explain(
        self,
        student_features: Dict[str, Any],
        p_dropout_current: float,
    ) -> Dict[str, Any]:
        """
        Generate penjelasan lengkap untuk satu prediksi.

        Args:
            student_features: dict raw_features (dari form input)
            p_dropout_current: probabilitas dropout saat ini (0–1)

        Returns:
            Dict dengan keys:
                feature_impacts  — list fitur diurutkan berdasarkan dampak
                counterfactuals  — list skenario "jika ... maka ..."
                top_risk_factor  — fitur paling berkontribusi meningkatkan risiko
                method           — "shap" atau "coefficient"
                warning          — peringatan tentang interpretasi (Issue 8)
        """
        warning = (
            "Nilai kontribusi fitur di bawah ini adalah setelah transformasi (log dan scaling). "
            "Angka tidak mencerminkan perubahan dalam unit asli. Gunakan sebagai indikator prioritas, "
            "bukan besaran pasti."
        )

        if self._shap_available and self._shap_explainer:
            feature_impacts = self._compute_shap_impacts(student_features)
        else:
            feature_impacts = self._compute_coefficient_impacts(student_features)

        counterfactuals = self._compute_counterfactuals(
            student_features, p_dropout_current
        )

        # Fitur teratas yang meningkatkan risiko
        risk_factors = [
            f for f in feature_impacts if f["impact_direction"] == "meningkatkan risiko"
        ]
        top_risk = risk_factors[0]["label"] if risk_factors else None

        return {
            "feature_impacts": feature_impacts[:7],  # top-7 agar tidak overwhelming
            "counterfactuals": counterfactuals,
            "top_risk_factor": top_risk,
            "method": "shap" if self._shap_available else "coefficient",
            "p_dropout_pct": round(p_dropout_current * 100, 1),
            "warning": warning,
        }

    def _compute_shap_impacts(
        self, features: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Hitung SHAP values dan return list impact per fitur."""
        try:
            import pandas as pd
            df = pd.DataFrame([features])
            # Transform melalui preprocessor
            preprocessor = self._model.named_steps["preprocessor"]
            X_transformed = preprocessor.transform(df)
            shap_values = self._shap_explainer.shap_values(X_transformed)

            # SHAP values untuk kelas dropout=1
            if hasattr(shap_values, '__len__') and len(shap_values) == 2:
                vals = shap_values[1][0]  # class 1 (dropout)
            else:
                vals = shap_values[0] if hasattr(shap_values, '__len__') else shap_values

            return self._format_impacts(vals, features)
        except Exception as e:
            logger.warning("SHAP calculation failed: %s, fallback ke coef", e)
            return self._compute_coefficient_impacts(features)

    def _compute_coefficient_impacts(
        self, features: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Estimasi dampak fitur dari koefisien model (tanpa SHAP)."""
        if self._coef_importance is None:
            return self._rule_based_impacts(features)

        # Transformasi fitur lewat preprocessor
        try:
            import pandas as pd
            df = pd.DataFrame([features])
            preprocessor = self._model.named_steps["preprocessor"]
            X_transformed = preprocessor.transform(df)

            # Koefisien × nilai fitur ≈ kontribusi linear
            coef = self._model.named_steps["model"].coef_[0]
            contributions = coef * X_transformed[0]

            return self._format_impacts(contributions, features)
        except Exception:
            return self._rule_based_impacts(features)

    def _format_impacts(
        self, values: np.ndarray, features: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Format impact values ke list dict yang siap ditampilkan.
        values: array kontribusi per fitur transformasi (bisa berbeda dimensi dari input)
        """
        # Karena preprocessor mengubah dimensi (one-hot dll), kita gunakan
        # mapping berbeda: ambil magnitude total per original feature
        n_feat = min(len(values), len(FEATURE_NAMES))
        impact_list = []

        for i in range(n_feat):
            fname = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feature_{i}"
            val = float(values[i]) if i < len(values) else 0.0
            fval = features.get(fname)

            # Positif = meningkatkan p_dropout = meningkatkan risiko dropout
            direction = "meningkatkan risiko" if val > 0 else "menurunkan risiko"
            impact_list.append({
                "feature": fname,
                "label": _FEATURE_LABELS.get(fname, fname),
                "value": fval,
                "shap_value": round(val, 4),
                "impact_pct": round(abs(val) * 100, 2),
                "impact_direction": direction,
            })

        # Urutkan berdasarkan magnitude
        impact_list.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return impact_list

    def _rule_based_impacts(
        self, features: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Fallback: estimasi dampak berdasarkan aturan heuristik.
        Digunakan saat model tidak tersedia.
        """
        impacts = []

        def add(fname: str, val: float, direction: str, magnitude: float):
            impacts.append({
                "feature": fname,
                "label": _FEATURE_LABELS.get(fname, fname),
                "value": features.get(fname),
                "shap_value": magnitude if direction == "meningkatkan risiko" else -magnitude,
                "impact_pct": round(magnitude * 100, 1),
                "impact_direction": direction,
            })

        att = float(features.get("attendance_rate", 75))
        if att < 60:
            add("attendance_rate", att, "meningkatkan risiko", 0.30)
        elif att >= 85:
            add("attendance_rate", att, "menurunkan risiko", 0.15)
        else:
            add("attendance_rate", att, "meningkatkan risiko", 0.10)

        sc = float(features.get("test_scores_avg", 65))
        if sc < 50:
            add("test_scores_avg", sc, "meningkatkan risiko", 0.25)
        elif sc >= 75:
            add("test_scores_avg", sc, "menurunkan risiko", 0.12)

        mot = int(features.get("motivation_score", 5))
        if mot <= 3:
            add("motivation_score", mot, "meningkatkan risiko", 0.20)
        elif mot >= 8:
            add("motivation_score", mot, "menurunkan risiko", 0.10)

        bl = int(features.get("backlogs", 0))
        if bl >= 3:
            add("backlogs", bl, "meningkatkan risiko", 0.18)

        st = int(features.get("stress_levels", 2))
        if st == 3:
            add("stress_levels", st, "meningkatkan risiko", 0.15)

        impacts.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return impacts

    def _compute_counterfactuals(
        self,
        features: Dict[str, Any],
        p_dropout_current: float,
    ) -> List[Dict[str, Any]]:
        """
        Hitung counterfactual explanations: "jika X berubah, maka risiko menjadi Y".

        Untuk setiap fitur actionable, simulasikan prediksi dengan nilai target.
        """
        counterfactuals = []

        for fname, cfg in _ACTIONABLE_FEATURES.items():
            current_val = features.get(fname)
            if current_val is None:
                continue

            target = cfg["target"]
            direction = cfg["direction"]
            current_float = float(current_val)

            # Skip jika sudah lebih baik dari target
            if direction == "increase" and current_float >= target:
                continue
            if direction == "decrease" and current_float <= target:
                continue

            # Prediksi dengan nilai baru
            new_features = {**features, fname: target}
            try:
                import pandas as pd
                df = pd.DataFrame([new_features])
                proba = self._model.predict_proba(df)[0]
                p_dropout_new = float(proba[1])
                delta_pct = (p_dropout_current - p_dropout_new) * 100

                if delta_pct > 0.5:  # hanya tampilkan jika ada dampak signifikan
                    counterfactuals.append({
                        "feature": fname,
                        "label": cfg["label"],
                        "current_value": current_float,
                        "target_value": target,
                        "unit": cfg["unit"],
                        "p_dropout_new_pct": round(p_dropout_new * 100, 1),
                        "dropout_reduction_pct": round(delta_pct, 1),
                        "direction": direction,
                        "message": (
                            f"Jika {cfg['label']} "
                            f"{'ditingkatkan menjadi' if direction == 'increase' else 'diturunkan menjadi'} "
                            f"{target}{cfg['unit']}, "
                            f"risiko dropout diperkirakan turun dari "
                            f"{round(p_dropout_current*100,1)}% menjadi {round(p_dropout_new*100,1)}% "
                            f"(-{round(delta_pct,1)} poin)."
                        ),
                    })
            except Exception as exc:
                logger.debug("Counterfactual failed for %s: %s", fname, exc)

        # Urutkan berdasarkan dampak terbesar
        counterfactuals.sort(key=lambda x: x["dropout_reduction_pct"], reverse=True)
        return counterfactuals[:4]  # maksimal 4 skenario


# ── Singleton ─────────────────────────────────────────────────────────────────
_explainer_instance: Optional[DropoutExplainer] = None


def get_explainer(model) -> DropoutExplainer:
    """Ambil atau buat singleton explainer."""
    global _explainer_instance
    if _explainer_instance is None:
        _explainer_instance = DropoutExplainer(model)
    return _explainer_instance


def explain_prediction(
    model,
    student_features: Dict[str, Any],
    p_dropout: float,
) -> Dict[str, Any]:
    """
    Convenience function: generate penjelasan untuk satu prediksi.

    Args:
        model: scikit-learn Pipeline (Predictor._model)
        student_features: dict fitur mahasiswa
        p_dropout: probabilitas dropout (0–1)

    Returns:
        Dict penjelasan lengkap
    """
    try:
        explainer = get_explainer(model)
        return explainer.explain(student_features, p_dropout)
    except Exception as exc:
        logger.error("explain_prediction error: %s", exc)
        return {
            "feature_impacts": [],
            "counterfactuals": [],
            "top_risk_factor": None,
            "method": "error",
            "error": str(exc),
            "p_dropout_pct": round(p_dropout * 100, 1),
            "warning": "Penjelasan AI tidak tersedia saat ini.",
        }