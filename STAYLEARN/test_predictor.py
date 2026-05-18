import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from predictor import (
    Predictor,
    FEATURE_NAMES,
    _classify_risk,
    _factor_analysis,
    _get_recommendation,
)
VALID_SAMPLE: dict = {
    "location_type": "Urban",
    "family_income": 8000.0,
    "financial_aid_status": 1,
    "distance_to_institute": 5.0,
    "internet_connectivity_issues": 0,
    "motivation_score": 7,
    "career_alignment": 3,
    "stress_levels": 2,
    "family_support": 3,
    "attendance_rate": 85.0,
    "test_scores_avg": 75.0,
    "backlogs": 0,
    "teaching_quality_rating": 7,
}
HIGH_RISK_SAMPLE: dict = {
    "location_type": "Rural",
    "family_income": 3000.0,
    "financial_aid_status": 0,
    "distance_to_institute": 40.0,
    "internet_connectivity_issues": 2,
    "motivation_score": 2,
    "career_alignment": 1,
    "stress_levels": 3,
    "family_support": 1,
    "attendance_rate": 45.0,
    "test_scores_avg": 40.0,
    "backlogs": 6,
    "teaching_quality_rating": 2,
}
@pytest.fixture
def mock_model():
    model = MagicMock()
    def _predict_proba(df):
        n = len(df)
        return np.tile([0.75, 0.25], (n, 1))
    model.predict_proba.side_effect = _predict_proba
    return model
@pytest.fixture
def predictor(mock_model, tmp_path):
    model_path = str(tmp_path / "test_model.joblib")
    with patch("predictor.os.path.exists", return_value=True), \
         patch("predictor.joblib.load", return_value=mock_model):
        p = Predictor(model_path)
    return p
class TestClassifyRisk:
    def test_low_risk_boundary(self):
        assert _classify_risk(0.8) == ("rendah", "Risiko Rendah")
    def test_low_risk_exact_threshold(self):
        assert _classify_risk(0.7) == ("rendah", "Risiko Rendah")
    def test_medium_risk(self):
        assert _classify_risk(0.5) == ("sedang", "Risiko Sedang")
    def test_medium_risk_exact_lower_boundary(self):
        assert _classify_risk(0.4) == ("sedang", "Risiko Sedang")
    def test_high_risk_just_below_medium(self):
        assert _classify_risk(0.39) == ("tinggi", "Risiko Tinggi")
    def test_high_risk_zero(self):
        assert _classify_risk(0.0) == ("tinggi", "Risiko Tinggi")
class TestFactorAnalysis:
    def test_all_bad_produces_concerns(self):
        result = _factor_analysis(HIGH_RISK_SAMPLE)
        assert len(result["concerns"]) > 0
        assert len(result["strengths"]) == 0
    def test_all_good_produces_strengths(self):
        good = {
            "attendance_rate": 90,
            "test_scores_avg": 85,
            "backlogs": 0,
            "motivation_score": 9,
            "stress_levels": 1,
            "family_support": 3,
            "family_income": 20000,
            "internet_connectivity_issues": 0,
            "career_alignment": 3,
            "distance_to_institute": 3.0,
            "financial_aid_status": 2,
            "teaching_quality_rating": 9,
        }
        result = _factor_analysis(good)
        assert len(result["strengths"]) > 0
        assert len(result["concerns"]) == 0
    def test_attendance_dead_zone_60_to_79(self):
        data = {**HIGH_RISK_SAMPLE, "attendance_rate": 70.0}
        result = _factor_analysis(data)
        attendance_concerns = [c for c in result["concerns"] if "Kehadiran" in c]
        assert len(attendance_concerns) > 0, (
            "attendance 70% should produce a concern (dead-zone fix missing)"
        )
    def test_scores_dead_zone_50_to_69(self):
        data = {**HIGH_RISK_SAMPLE, "test_scores_avg": 60.0}
        result = _factor_analysis(data)
        score_concerns = [c for c in result["concerns"] if "Nilai" in c]
        assert len(score_concerns) > 0, (
            "test_scores_avg 60 should produce a concern (dead-zone fix missing)"
        )
    def test_low_teaching_quality_produces_concern(self):
        data = {**VALID_SAMPLE, "teaching_quality_rating": 2}
        result = _factor_analysis(data)
        teaching_concerns = [c for c in result["concerns"] if "pengajaran" in c.lower()]
        assert len(teaching_concerns) > 0
    def test_high_teaching_quality_produces_strength(self):
        data = {**VALID_SAMPLE, "teaching_quality_rating": 9}
        result = _factor_analysis(data)
        teaching_strengths = [s for s in result["strengths"] if "pengajaran" in s.lower()]
        assert len(teaching_strengths) > 0
    def test_far_distance_produces_concern(self):
        data = {**VALID_SAMPLE, "distance_to_institute": 45.0}
        result = _factor_analysis(data)
        dist_concerns = [c for c in result["concerns"] if "km" in c]
        assert len(dist_concerns) > 0
    def test_full_scholarship_produces_strength(self):
        data = {**VALID_SAMPLE, "financial_aid_status": 2}
        result = _factor_analysis(data)
        aid_strengths = [s for s in result["strengths"] if "beasiswa" in s.lower() or "penuh" in s.lower()]
        assert len(aid_strengths) > 0
    def test_no_aid_low_income_produces_concern(self):
        data = {**VALID_SAMPLE, "financial_aid_status": 0, "family_income": 4000.0}
        result = _factor_analysis(data)
        aid_concerns = [c for c in result["concerns"] if "bantuan" in c.lower()]
        assert len(aid_concerns) > 0
class TestGetRecommendation:
    def test_high_risk(self):
        assert "perhatian segera" in _get_recommendation("tinggi")
    def test_medium_risk(self):
        assert "Pantau perkembangan" in _get_recommendation("sedang")
    def test_low_risk(self):
        assert "stabilitas yang baik" in _get_recommendation("rendah")
class TestPredictorInit:
    def test_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Predictor(str(tmp_path / "does_not_exist.joblib"))
    def test_raises_runtime_error_on_load_failure(self, tmp_path):
        model_path = str(tmp_path / "bad_model.joblib")
        with patch("predictor.os.path.exists", return_value=True), \
             patch("predictor.joblib.load", side_effect=Exception("corrupt")):
            with pytest.raises(RuntimeError):
                Predictor(model_path)
    def test_model_loaded_successfully(self, predictor, mock_model):
        assert predictor.model is mock_model
class TestPredictSingle:
    def test_returns_expected_keys(self, predictor):
        result = predictor.predict_single(VALID_SAMPLE)
        for key in ("p_stay", "p_dropout", "risk_level", "risk_label",
                    "concerns", "strengths", "recommendation"):
            assert key in result, f"Missing key: {key}"
    def test_probabilities_sum_to_100(self, predictor):
        result = predictor.predict_single(VALID_SAMPLE)
        assert abs(result["p_stay"] + result["p_dropout"] - 100.0) < 0.2
    def test_risk_level_matches_p_stay(self, predictor, mock_model):
        mock_model.predict_proba.side_effect = None
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])
        predictor._cached_predict_single.cache_clear()
        result = predictor.predict_single(VALID_SAMPLE)
        assert result["risk_level"] == "tinggi"
    def test_invalid_location_raises_value_error(self, predictor):
        bad = {**VALID_SAMPLE, "location_type": "Mars"}
        with pytest.raises(ValueError, match="location_type"):
            predictor.predict_single(bad)
    def test_out_of_range_attendance_raises_value_error(self, predictor):
        bad = {**VALID_SAMPLE, "attendance_rate": 105.0}
        with pytest.raises(ValueError, match="attendance_rate"):
            predictor.predict_single(bad)
    def test_cache_hit_int_vs_float_income(self, predictor, mock_model):
        predictor._cached_predict_single.cache_clear()
        mock_model.predict_proba.side_effect = lambda df: np.tile([0.75, 0.25], (len(df), 1))
        sample_int = {**VALID_SAMPLE, "family_income": 8000}
        sample_float = {**VALID_SAMPLE, "family_income": 8000.0}
        predictor.predict_single(sample_int)
        predictor.predict_single(sample_float)
        assert mock_model.predict_proba.call_count == 1, (
            "Cache miss for int vs float income, type normalisation is broken"
        )
    def test_cache_hit_same_data_twice(self, predictor, mock_model):
        predictor._cached_predict_single.cache_clear()
        mock_model.predict_proba.side_effect = lambda df: np.tile([0.75, 0.25], (len(df), 1))
        predictor.predict_single(VALID_SAMPLE)
        predictor.predict_single(VALID_SAMPLE)
        assert mock_model.predict_proba.call_count == 1
class TestPredictBatchWithErrors:
    def _make_df(self, rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows)
    def test_valid_batch_returns_results(self, predictor):
        df = self._make_df([VALID_SAMPLE, {**VALID_SAMPLE, "attendance_rate": 55.0}])
        results, errors = predictor.predict_batch_with_errors(df)
        assert len(results) == 2
        assert len(errors) == 0
    def test_result_contains_prediction_columns(self, predictor):
        df = self._make_df([VALID_SAMPLE])
        results, _ = predictor.predict_batch_with_errors(df)
        assert "kemungkinan_bertahan_pct" in results[0]
        assert "kemungkinan_dropout_pct" in results[0]
        assert "tingkat_risiko" in results[0]
    def test_invalid_row_isolated_as_error(self, predictor):
        bad = {**VALID_SAMPLE, "location_type": "Nowhere", "attendance_rate": 80.0}
        df = self._make_df([VALID_SAMPLE, bad])
        results, errors = predictor.predict_batch_with_errors(df)
        assert len(results) == 1
        assert len(errors) == 1
        assert "row" in errors[0]
        assert "error" in errors[0]
    def test_all_invalid_returns_empty_results(self, predictor):
        bad1 = {**VALID_SAMPLE, "location_type": "X"}
        bad2 = {**VALID_SAMPLE, "attendance_rate": 999}
        df = self._make_df([bad1, bad2])
        results, errors = predictor.predict_batch_with_errors(df)
        assert results == []
        assert len(errors) == 2
    def test_missing_column_raises_value_error(self, predictor):
        df = pd.DataFrame([{"location_type": "Urban", "family_income": 8000}])
        with pytest.raises(ValueError, match="Kolom wajib"):
            predictor.predict_batch_with_errors(df)
    def test_exceeds_max_rows_raises_value_error(self, predictor):
        predictor.max_batch_rows = 2
        df = self._make_df([VALID_SAMPLE] * 3)
        with pytest.raises(ValueError, match="terlalu besar"):
            predictor.predict_batch_with_errors(df)
    def test_identity_columns_preserved_in_results(self, predictor):
        row = {**VALID_SAMPLE, "nim": "123", "nama": "Budi"}
        df = self._make_df([row])
        results, _ = predictor.predict_batch_with_errors(df)
        assert results[0]["nim"] == "123"
        assert results[0]["nama"] == "Budi"
class TestGenerateRandom:
    def test_returns_all_feature_keys(self, predictor):
        data = predictor.generate_random()
        for key in FEATURE_NAMES:
            assert key in data, f"Missing feature key: {key}"
    def test_location_type_is_valid(self, predictor):
        from predictor import ALLOWED_LOCATION_TYPES
        for _ in range(10):
            data = predictor.generate_random()
            assert data["location_type"] in ALLOWED_LOCATION_TYPES
    def test_numeric_values_within_bounds(self, predictor):
        from predictor import NUMERIC_RANGES
        for _ in range(20):
            data = predictor.generate_random()
            for key, (lo, hi) in NUMERIC_RANGES.items():
                val = data[key]
                assert lo <= val <= hi, (
                    f"{key}={val} is outside [{lo}, {hi}]"
                )
    def test_predict_single_accepts_random_data(self, predictor):
        for _ in range(10):
            data = predictor.generate_random()
            result = predictor.predict_single(data)
            assert "risk_level" in result