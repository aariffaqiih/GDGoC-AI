"""
recommender.py — Intervention Recommendation Engine (SDG 4.1 & 4.4)

Menggunakan KMeans clustering untuk mengelompokkan profil mahasiswa,
lalu merekomendasikan jenis intervensi berdasarkan apa yang berhasil
pada mahasiswa dengan profil serupa.

Arsitektur:
  InterventionRecommender
    .build()           — fit model dari data historis di DB
    .recommend(features) — rekomendasikan intervensi untuk profil baru
    .extract_type(notes) — ekstrak jenis intervensi dari catatan konselor

Keterbatasan yang disadari:
  - Cold-start: tanpa data historis, fallback ke rule-based
  - Cluster semantics: KMeans tidak selalu interpretable
  - Causal assumption: korelasi intervensi–hasil ≠ kausalitas
"""

import json
import logging
import re
from functools import lru_cache
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Mapping kata kunci → tipe intervensi ─────────────────────────────────────
_KEYWORD_MAP: List[Tuple[str, str]] = [
    # (pola regex, label intervensi)
    (r"motivasi|semangat|dorongan|percaya diri", "motivasi"),
    (r"akademik|belajar|nilai|ipk|matkul|tunggakan|remedial|tutor", "bimbingan_akademik"),
    (r"beasiswa|finansial|keuangan|ukt|biaya|bantuan", "bantuan_finansial"),
    (r"karir|magang|kerja|industri|profesi|kompetensi", "pengembangan_karir"),
    (r"keluarga|orang tua|rumah|dukungan", "dukungan_keluarga"),
    (r"stres|psikologi|mental|cemas|depresi|emosi|konseling psikolog", "kesehatan_mental"),
    (r"internet|koneksi|akses|online|teknologi", "akses_teknologi"),
    (r"jadwal|kehadiran|absen|hadir", "manajemen_waktu"),
]

# Fitur numerik yang digunakan untuk clustering
_CLUSTER_FEATURES = [
    "motivation_score",
    "stress_levels",
    "family_support",
    "attendance_rate",
    "test_scores_avg",
    "backlogs",
    "teaching_quality_rating",
    "financial_aid_status",
]

# Fallback rekomendasi berdasarkan risk level saja (cold-start)
_FALLBACK_RECOMMENDATIONS: Dict[str, List[Dict]] = {
    "tinggi": [
        {
            "intervention_type": "bimbingan_akademik",
            "label": "Bimbingan Akademik Intensif",
            "description": (
                "Sesi bimbingan 1-on-1 dengan dosen wali, fokus pada perbaikan "
                "nilai dan penyelesaian mata kuliah tertunggak."
            ),
            "evidence_base": "rule_based",
        },
        {
            "intervention_type": "kesehatan_mental",
            "label": "Dukungan Psikologis",
            "description": (
                "Sesi konseling dengan psikolog kampus untuk mengatasi stres "
                "dan meningkatkan resiliensi akademik."
            ),
            "evidence_base": "rule_based",
        },
    ],
    "sedang": [
        {
            "intervention_type": "motivasi",
            "label": "Program Penguatan Motivasi",
            "description": (
                "Workshop goal-setting dan mindset, menghubungkan mahasiswa "
                "dengan komunitas belajar yang suportif."
            ),
            "evidence_base": "rule_based",
        },
        {
            "intervention_type": "pengembangan_karir",
            "label": "Career Mentoring",
            "description": (
                "Sesi eksplorasi karir dengan alumni atau praktisi industri "
                "untuk memperkuat relevansi studi."
            ),
            "evidence_base": "rule_based",
        },
    ],
    "rendah": [
        {
            "intervention_type": "pengembangan_karir",
            "label": "Career Development Program",
            "description": (
                "Program peningkatan keterampilan kerja dan TIK (SDG 4.4) "
                "untuk mempersiapkan lulusan yang kompetitif."
            ),
            "evidence_base": "rule_based",
        },
    ],
}


def extract_intervention_type(notes: str) -> str:
    """
    Ekstrak jenis intervensi dari catatan konselor menggunakan keyword matching.

    Args:
        notes: Teks catatan konselor (bebas format)

    Returns:
        Label intervensi (string). Default 'umum' jika tidak ada kata kunci.
    """
    if not notes or not notes.strip():
        return "umum"

    notes_lower = notes.lower()
    for pattern, label in _KEYWORD_MAP:
        if re.search(pattern, notes_lower):
            return label
    return "umum"


class InterventionRecommender:
    """
    Recommendation engine berbasis KMeans clustering dan evidence dari historis.

    Lifecycle:
        1. recommender = InterventionRecommender()
        2. recommender.build(evaluated_interventions)  # dari DB
        3. recs = recommender.recommend(student_features, risk_level)

    Thread safety: setelah build() selesai, recommend() bersifat read-only
    dan aman untuk concurrent calls.
    """

    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self._kmeans = None
        self._knowledge_base: List[Dict[str, Any]] = []
        self._is_built = False
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None

    def build(self, evaluated_interventions: List[Dict[str, Any]]) -> None:
        """
        Fit KMeans dari data historis intervensi yang sudah terevaluasi.

        Args:
            evaluated_interventions: Output dari db.get_evaluated_interventions()
        """
        if len(evaluated_interventions) < max(self.n_clusters, 5):
            logger.info(
                "Data historis tidak cukup (%d kasus, butuh minimal %d). "
                "Fallback ke rule-based recommendations.",
                len(evaluated_interventions),
                max(self.n_clusters, 5),
            )
            self._is_built = False
            return

        try:
            from sklearn.cluster import KMeans

            X = self._extract_feature_matrix(evaluated_interventions)
            if X is None or X.shape[0] < self.n_clusters:
                self._is_built = False
                return

            # Normalisasi fitur
            self._feature_means = X.mean(axis=0)
            self._feature_stds  = X.std(axis=0)
            # Hindari pembagian dengan nol
            self._feature_stds[self._feature_stds == 0] = 1.0
            X_norm = (X - self._feature_means) / self._feature_stds

            # Sesuaikan n_clusters jika data terlalu sedikit
            n_clusters = min(self.n_clusters, X.shape[0])
            self._kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10,
                max_iter=300,
            )
            cluster_labels = self._kmeans.fit_predict(X_norm)

            # Bangun knowledge base: per cluster, per intervention_type
            self._knowledge_base = []
            for i, row in enumerate(evaluated_interventions):
                self._knowledge_base.append({
                    **row,
                    "cluster": int(cluster_labels[i]),
                })

            self._is_built = True
            logger.info(
                "Recommender built: %d cases, %d clusters",
                len(evaluated_interventions),
                n_clusters,
            )

        except Exception as exc:
            logger.error("Gagal build recommender: %s", exc)
            self._is_built = False

    def recommend(
        self,
        student_features: Dict[str, Any],
        risk_level: str = "sedang",
        top_n: int = 3,
    ) -> Dict[str, Any]:
        """
        Rekomendasikan intervensi untuk mahasiswa dengan profil given.

        Args:
            student_features: dict raw_features dari prediksi terbaru mahasiswa
            risk_level:       'tinggi' | 'sedang' | 'rendah'
            top_n:            jumlah rekomendasi yang dikembalikan

        Returns:
            Dict dengan keys: recommendations, method, cluster_id, similar_cases_count
        """
        if not self._is_built or self._kmeans is None:
            return self._fallback(risk_level, top_n, reason="insufficient_data")

        try:
            features_vec = self._features_to_vector(student_features)
            if features_vec is None:
                return self._fallback(risk_level, top_n, reason="feature_extraction_failed")

            # Cari cluster terdekat
            features_norm = (features_vec - self._feature_means) / self._feature_stds
            cluster_id = int(self._kmeans.predict(features_norm.reshape(1, -1))[0])

            # Filter knowledge base untuk cluster ini, dan ambil yang berhasil (delta > 0)
            cluster_cases = [
                case for case in self._knowledge_base
                if case["cluster"] == cluster_id and case["delta"] > 0.01
            ]

            if len(cluster_cases) < 2:
                # Tidak cukup evidence untuk cluster ini
                return self._fallback(risk_level, top_n, reason="no_cluster_evidence")

            # Hitung effectiveness per intervention_type
            type_stats: Dict[str, Dict] = {}
            for case in cluster_cases:
                itype = case["intervention_type"]
                if itype not in type_stats:
                    type_stats[itype] = {"deltas": [], "count": 0}
                type_stats[itype]["deltas"].append(case["delta"])
                type_stats[itype]["count"] += 1

            # Rank berdasarkan rata-rata delta (penurunan p_dropout)
            ranked = sorted(
                type_stats.items(),
                key=lambda x: sum(x[1]["deltas"]) / len(x[1]["deltas"]),
                reverse=True,
            )[:top_n]

            recommendations = []
            for itype, stats in ranked:
                avg_delta = sum(stats["deltas"]) / len(stats["deltas"])
                success_rate = len([d for d in stats["deltas"] if d > 0.02]) / len(stats["deltas"])
                recommendations.append({
                    "intervention_type": itype,
                    "label": self._type_label(itype),
                    "description": self._type_description(itype),
                    "avg_dropout_reduction_pct": round(avg_delta * 100, 1),
                    "success_rate_pct": round(success_rate * 100, 1),
                    "evidence_cases": stats["count"],
                    "evidence_base": "historical",
                })

            return {
                "recommendations": recommendations,
                "method": "kmeans_cluster",
                "cluster_id": cluster_id,
                "similar_cases_count": len(cluster_cases),
            }

        except Exception as exc:
            logger.error("recommend() error: %s", exc)
            return self._fallback(risk_level, top_n, reason="runtime_error")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_feature_matrix(
        self, cases: List[Dict[str, Any]]
    ) -> Optional[np.ndarray]:
        rows = []
        for case in cases:
            vec = self._features_to_vector(case.get("features", {}))
            if vec is not None:
                rows.append(vec)
        if not rows:
            return None
        return np.array(rows, dtype=float)

    def _features_to_vector(self, features: Dict[str, Any]) -> Optional[np.ndarray]:
        try:
            vec = []
            for key in _CLUSTER_FEATURES:
                val = features.get(key)
                if val is None:
                    return None
                vec.append(float(val))
            return np.array(vec, dtype=float)
        except (TypeError, ValueError):
            return None

    def _fallback(
        self, risk_level: str, top_n: int, reason: str
    ) -> Dict[str, Any]:
        recs = _FALLBACK_RECOMMENDATIONS.get(risk_level, _FALLBACK_RECOMMENDATIONS["sedang"])
        return {
            "recommendations": recs[:top_n],
            "method": "rule_based_fallback",
            "fallback_reason": reason,
            "cluster_id": None,
            "similar_cases_count": 0,
        }

    def _type_label(self, itype: str) -> str:
        labels = {
            "motivasi":             "Penguatan Motivasi",
            "bimbingan_akademik":   "Bimbingan Akademik",
            "bantuan_finansial":    "Bantuan Finansial / Beasiswa",
            "pengembangan_karir":   "Pengembangan Karir & TIK",
            "dukungan_keluarga":    "Konseling Keluarga",
            "kesehatan_mental":     "Dukungan Psikologis",
            "akses_teknologi":      "Akses Teknologi & Internet",
            "manajemen_waktu":      "Manajemen Waktu & Kehadiran",
            "umum":                 "Konseling Umum",
        }
        return labels.get(itype, itype.replace("_", " ").title())

    def _type_description(self, itype: str) -> str:
        descriptions = {
            "motivasi": (
                "Sesi motivasi, goal-setting, dan koneksi dengan komunitas belajar "
                "yang suportif untuk membangun kembali semangat studi."
            ),
            "bimbingan_akademik": (
                "Bimbingan akademik intensif: penyelesaian tunggakan, strategi belajar, "
                "dan program peer tutoring."
            ),
            "bantuan_finansial": (
                "Fasilitasi akses ke beasiswa, keringanan UKT, atau program magang "
                "berbayar sesuai kondisi finansial mahasiswa."
            ),
            "pengembangan_karir": (
                "Program Career Development: eksplorasi karir, workshop keterampilan TIK "
                "relevan (SDG 4.4), dan mentoring industri."
            ),
            "dukungan_keluarga": (
                "Mediasi dan konseling melibatkan keluarga untuk memperkuat dukungan "
                "dari lingkungan rumah."
            ),
            "kesehatan_mental": (
                "Sesi dengan psikolog kampus: manajemen stres, mindfulness, dan "
                "penanganan kecemasan akademik."
            ),
            "akses_teknologi": (
                "Solusi konektivitas: akses kampus, subsidi paket data, "
                "atau perangkat pinjam untuk mendukung pembelajaran."
            ),
            "manajemen_waktu": (
                "Coaching manajemen waktu, pembuatan jadwal belajar terstruktur, "
                "dan monitoring kehadiran rutin."
            ),
            "umum": (
                "Sesi konseling umum untuk mengidentifikasi kebutuhan spesifik "
                "sebelum menentukan jenis intervensi yang tepat."
            ),
        }
        return descriptions.get(itype, "Intervensi sesuai kebutuhan mahasiswa.")


# ── Singleton module-level ────────────────────────────────────────────────────
_recommender_instance: Optional[InterventionRecommender] = None


def get_recommender() -> InterventionRecommender:
    """Ambil atau buat singleton recommender. Build dilakukan saat pertama kali."""
    global _recommender_instance
    if _recommender_instance is None:
        _recommender_instance = InterventionRecommender()
        _rebuild_recommender()
    return _recommender_instance


def _rebuild_recommender() -> None:
    """Rebuild model dari data DB terkini. Dipanggil saat cache invalidated."""
    global _recommender_instance
    if _recommender_instance is None:
        _recommender_instance = InterventionRecommender()
    try:
        import database as db
        data = db.get_evaluated_interventions()
        _recommender_instance.build(data)
    except Exception as exc:
        logger.error("Gagal rebuild recommender: %s", exc)


def invalidate_recommender_cache() -> None:
    """Paksa rebuild recommender saat ada data baru."""
    global _recommender_instance
    _recommender_instance = None
    logger.info("Recommender cache invalidated")
