"""
ai_engine.py — Unified AI Engine untuk StayLearn
=================================================
Menggunakan LLM nyata melalui API eksternal (tanpa GPU lokal):

Prioritas backend:
  1. Groq API  — gratis, cepat, model Llama3-8B/70B & Mixtral
     → set GROQ_API_KEY di .env
  2. Hugging Face Inference API — gratis (rate-limited), banyak pilihan model
     → opsional: set HF_API_TOKEN di .env untuk limit lebih tinggi
  3. Fallback rule-based — selalu tersedia jika kedua API tidak bisa diakses

Tiga fungsi utama:
  - generate_recommendation(features, risk_level, concerns, strengths)
    → Rekomendasi tindak lanjut yang personal & actionable (bukan rule-based)

  - generate_career_advice(features)
    → Saran karir konkret sesuai kondisi mahasiswa (SDG 4.4)

  - chat(message, history)
    → Chatbot conversational untuk pendampingan mahasiswa (SDG 4.7)

Catatan keamanan:
  - Tidak mengirim PII (nama, NIM) ke API eksternal
  - Semua response di-sanitize sebelum dikirim ke client
  - Timeout 15 detik per request; fallback otomatis jika gagal
"""

import json
import logging
import os
import re
import time
from typing import Dict, List, Any, Optional, Tuple

import requests

# Auto-load .env jika dotenv tersedia (safety net jika diimport sebelum config.py)
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv()
except ImportError:
    pass  # dotenv tidak wajib, env vars bisa diset manual

logger = logging.getLogger(__name__)

# ─── Konfigurasi API ──────────────────────────────────────────────────────────

# ─── API keys dibaca secara lazy (saat pertama kali digunakan) ───────────────
# Ini memastikan dotenv sudah di-load sebelum kita membaca env vars.
def _get_groq_key() -> str:
    """Baca GROQ_API_KEY dari environment secara lazy."""
    return os.environ.get("GROQ_API_KEY", "").strip()

def _get_hf_token() -> str:
    """Baca HF_API_TOKEN dari environment secara lazy."""
    return os.environ.get("HF_API_TOKEN", "").strip()

# Groq: model aktif per 2025
# Ref: https://console.groq.com/docs/models
GROQ_MODEL      = "llama-3.1-8b-instant"      # cepat, gratis, recommended
GROQ_MODEL_CHAT = "llama-3.3-70b-versatile"   # pintar, gratis, untuk chatbot
GROQ_BASE_URL   = "https://api.groq.com/openai/v1/chat/completions"

# HuggingFace Inference API: model yang stabil dan gratis
HF_MODELS = {
    "small": "HuggingFaceH4/zephyr-7b-beta",          # stabil, cepat
    "chat":  "HuggingFaceH4/zephyr-7b-beta",           # conversational
}
HF_BASE_URL = "https://api-inference.huggingface.co/models"

REQUEST_TIMEOUT = 20  # detik

# ─── Konteks kampus untuk sistem prompt ───────────────────────────────────────

_SYSTEM_CONTEXT = """Kamu adalah AI konselor akademik dari sistem StayLearn, 
platform deteksi dini mahasiswa berisiko dropout di Telkom University Purwokerto.
Sistem ini mendukung SDG 4 (Pendidikan Berkualitas).

Prinsip utama:
- Jawab dalam Bahasa Indonesia yang hangat, empatik, dan mendorong
- Berikan saran KONKRET dan ACTIONABLE, bukan deskriptif
- Fokus pada langkah nyata yang bisa dilakukan mahasiswa SEKARANG
- Hindari jawaban generik; sesuaikan dengan data spesifik yang diberikan
- Maksimal 3-4 kalimat per rekomendasi, padat dan bermakna"""

_FEATURE_LABELS = {
    "location_type": "Lokasi",
    "family_income": "Pendapatan keluarga",
    "financial_aid_status": "Status beasiswa",
    "distance_to_institute": "Jarak ke kampus",
    "internet_connectivity_issues": "Masalah internet",
    "motivation_score": "Skor motivasi",
    "career_alignment": "Kesesuaian karir",
    "stress_levels": "Tingkat stres",
    "family_support": "Dukungan keluarga",
    "attendance_rate": "Kehadiran",
    "test_scores_avg": "Rata-rata nilai",
    "backlogs": "MK tertunggak",
    "teaching_quality_rating": "Rating pengajaran",
}

_AID_LABELS = {0: "tidak ada beasiswa", 1: "beasiswa sebagian", 2: "beasiswa penuh"}
_STRESS_LABELS = {1: "rendah", 2: "sedang", 3: "tinggi"}
_SUPPORT_LABELS = {1: "kurang", 2: "cukup", 3: "baik"}
_ALIGNMENT_LABELS = {1: "tidak sesuai", 2: "cukup sesuai", 3: "sangat sesuai"}
_INTERNET_LABELS = {0: "tidak ada masalah", 1: "kadang bermasalah", 2: "sering bermasalah"}


def _build_student_summary(features: Dict[str, Any]) -> str:
    """Buat ringkasan profil mahasiswa yang mudah dibaca LLM."""
    f = features
    lines = [
        f"- Kehadiran: {f.get('attendance_rate', '?')}%",
        f"- Nilai rata-rata: {f.get('test_scores_avg', '?')}",
        f"- MK tertunggak: {f.get('backlogs', '?')} mata kuliah",
        f"- Motivasi belajar: {f.get('motivation_score', '?')}/10",
        f"- Tingkat stres: {_STRESS_LABELS.get(int(f.get('stress_levels', 2)), '?')}",
        f"- Dukungan keluarga: {_SUPPORT_LABELS.get(int(f.get('family_support', 2)), '?')}",
        f"- Kesesuaian karir: {_ALIGNMENT_LABELS.get(int(f.get('career_alignment', 2)), '?')}",
        f"- Status beasiswa: {_AID_LABELS.get(int(f.get('financial_aid_status', 0)), '?')}",
        f"- Pendapatan keluarga: Rp {int(f.get('family_income', 0)):,}/bulan",
        f"- Jarak ke kampus: {f.get('distance_to_institute', '?')} km",
        f"- Masalah internet: {_INTERNET_LABELS.get(int(f.get('internet_connectivity_issues', 0)), '?')}",
        f"- Rating kualitas pengajaran: {f.get('teaching_quality_rating', '?')}/10",
        f"- Lokasi: {f.get('location_type', '?')}",
    ]
    return "\n".join(lines)


# ─── Groq API ─────────────────────────────────────────────────────────────────

def _call_groq(
    messages: List[Dict[str, str]],
    model: str = GROQ_MODEL,
    max_tokens: int = 300,
    temperature: float = 0.7,
) -> Optional[str]:
    """Panggil Groq API. Returns None jika gagal."""
    api_key = _get_groq_key()
    if not api_key:
        logger.warning("GROQ_API_KEY tidak ditemukan di environment. Pastikan sudah diset di .env")
        return None
    try:
        resp = requests.post(
            GROQ_BASE_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            },
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code == 200:
            data = resp.json()
            text = data["choices"][0]["message"]["content"].strip()
            logger.info("Groq OK: model=%s, tokens=%s", model,
                        data.get("usage", {}).get("completion_tokens", "?"))
            return text
        elif resp.status_code == 429:
            logger.warning("Groq rate limit hit — coba lagi nanti")
            return None
        elif resp.status_code == 401:
            logger.error("Groq API key tidak valid (401). Cek GROQ_API_KEY di .env")
            return None
        elif resp.status_code == 404:
            logger.error("Groq model '%s' tidak ditemukan (404). Cek nama model.", model)
            return None
        else:
            logger.warning("Groq API error %d: %s", resp.status_code, resp.text[:300])
            return None
    except requests.exceptions.Timeout:
        logger.warning("Groq API timeout setelah %ds", REQUEST_TIMEOUT)
        return None
    except requests.exceptions.ConnectionError as e:
        logger.warning("Groq API connection error: %s", e)
        return None
    except Exception as e:
        logger.error("Groq API exception tidak terduga: %s", e)
        return None


# ─── HuggingFace Inference API ────────────────────────────────────────────────

def _call_hf_inference(
    prompt: str,
    model_key: str = "small",
    max_new_tokens: int = 300,
) -> Optional[str]:
    """Panggil HuggingFace Inference API. Returns None jika gagal."""
    hf_token = _get_hf_token()
    model_id = HF_MODELS.get(model_key, HF_MODELS["small"])
    headers = {"Content-Type": "application/json"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    else:
        logger.info("HF_API_TOKEN tidak diset — menggunakan public rate limit")

    try:
        resp = requests.post(
            f"{HF_BASE_URL}/{model_id}",
            headers=headers,
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": 0.7,
                    "do_sample": True,
                    "return_full_text": False,
                },
            },
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and data:
                text = data[0].get("generated_text", "")
            elif isinstance(data, dict):
                text = data.get("generated_text", "")
            else:
                return None
            result = text.strip() if text.strip() else None
            if result:
                logger.info("HF Inference OK: model=%s", model_id)
            return result
        elif resp.status_code == 503:
            logger.warning("HF model '%s' sedang loading (503) — coba lagi nanti", model_id)
            return None
        elif resp.status_code == 401:
            logger.error("HF token tidak valid (401)")
            return None
        else:
            logger.warning("HF API error %d: %s", resp.status_code, resp.text[:200])
            return None
    except requests.exceptions.Timeout:
        logger.warning("HF API timeout setelah %ds", REQUEST_TIMEOUT)
        return None
    except requests.exceptions.ConnectionError as e:
        logger.warning("HF API connection error: %s", e)
        return None
    except Exception as e:
        logger.error("HF API exception: %s", e)
        return None


# ─── Sanitize output ──────────────────────────────────────────────────────────

def _sanitize(text: str, max_chars: int = 1000) -> str:
    """Bersihkan output LLM dari karakter berbahaya."""
    if not text:
        return ""
    # Hapus HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Hapus karakter kontrol kecuali newline
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)
    # Potong jika terlalu panjang
    if len(text) > max_chars:
        # Potong di akhir kalimat
        cut = text[:max_chars]
        last_period = max(cut.rfind(". "), cut.rfind(".\n"), cut.rfind("! "), cut.rfind("? "))
        if last_period > max_chars * 0.7:
            text = text[:last_period + 1]
        else:
            text = cut + "..."
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. REKOMENDASI TINDAK LANJUT (LLM-powered)
# ═══════════════════════════════════════════════════════════════════════════════

_REC_FALLBACK = {
    "tinggi": (
        "🚨 Segera jadwalkan sesi konseling 1-on-1 minggu ini. "
        "Hubungi dosen wali untuk membuat rencana studi darurat. "
        "Prioritaskan penyelesaian 1-2 mata kuliah tertunggak yang paling kritis."
    ),
    "sedang": (
        "📅 Jadwalkan check-in bulanan dengan dosen wali atau konselor. "
        "Bergabunglah dengan kelompok belajar untuk meningkatkan engagement. "
        "Evaluasi beban studi dan pertimbangkan minta perpanjangan tugas jika perlu."
    ),
    "rendah": (
        "✅ Pertahankan momentum positif ini. "
        "Manfaatkan posisi baik ini untuk mentoring mahasiswa lain atau ikut riset dosen. "
        "Mulai eksplorasi peluang magang untuk persiapan karir."
    ),
}


def generate_recommendation(
    features: Dict[str, Any],
    risk_level: str,
    concerns: List[str],
    strengths: List[str],
    p_dropout: float,
) -> str:
    """
    Generate rekomendasi tindak lanjut personal menggunakan LLM.

    Args:
        features:   raw_features mahasiswa
        risk_level: 'tinggi' | 'sedang' | 'rendah'
        concerns:   list faktor risiko dari predictor
        strengths:  list faktor protektif dari predictor
        p_dropout:  probabilitas dropout 0-100

    Returns:
        String rekomendasi yang actionable
    """
    risk_labels = {"tinggi": "TINGGI (perlu intervensi segera)", "sedang": "SEDANG (perlu dipantau)", "rendah": "RENDAH (kondisi baik)"}
    risk_label = risk_labels.get(risk_level, risk_level)

    student_summary = _build_student_summary(features)
    concerns_text = "\n".join(f"  • {c}" for c in concerns[:5]) if concerns else "  (tidak ada)"
    strengths_text = "\n".join(f"  • {s}" for s in strengths[:3]) if strengths else "  (tidak ada)"

    prompt_content = f"""Data mahasiswa berikut menunjukkan risiko dropout {risk_label} ({p_dropout:.1f}%):

PROFIL MAHASISWA:
{student_summary}

FAKTOR RISIKO TERDETEKSI:
{concerns_text}

FAKTOR KEKUATAN:
{strengths_text}

Berikan rekomendasi tindak lanjut yang KONKRET dan ACTIONABLE dalam 3-4 kalimat.
Sebutkan langkah spesifik yang bisa dilakukan MINGGU INI (bukan saran umum).
Gunakan data yang ada untuk membuat rekomendasi yang personal.
Mulai dengan tindakan paling mendesak."""

    messages = [
        {"role": "system", "content": _SYSTEM_CONTEXT},
        {"role": "user", "content": prompt_content},
    ]

    # Coba Groq dulu
    result = _call_groq(messages, model=GROQ_MODEL, max_tokens=250, temperature=0.6)
    if result:
        return _sanitize(result, max_chars=600)

    # Fallback HuggingFace
    hf_prompt = f"<s>[INST] {_SYSTEM_CONTEXT}\n\n{prompt_content} [/INST]"
    result = _call_hf_inference(hf_prompt, model_key="small", max_new_tokens=200)
    if result:
        return _sanitize(result, max_chars=600)

    # Fallback rule-based
    return _REC_FALLBACK.get(risk_level, _REC_FALLBACK["sedang"])


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CAREER ADVICE — SDG 4.4 (LLM-powered)
# ═══════════════════════════════════════════════════════════════════════════════

_CAREER_FALLBACK = {
    "tinggi": (
        "Fokus dulu stabilkan kondisi akademik. Setelah stabil, coba daftar Google Digital Garage "
        "(gratis, online) untuk keterampilan digital dasar yang langsung meningkatkan daya saing kerja."
    ),
    "sedang": (
        "Ikuti 1 kursus online gratis di Dicoding atau Google Career Certificates bulan ini. "
        "Bangun profil LinkedIn dan mulai terhubung dengan alumni di bidang yang diminati."
    ),
    "rendah": (
        "Manfaatkan posisi akademik yang baik: daftarkan diri ke program magang perusahaan teknologi "
        "atau riset dosen. Kejar sertifikasi profesi (AWS, Google, Microsoft) sebelum lulus."
    ),
}


def generate_career_advice(
    features: Dict[str, Any],
    risk_level: str = "sedang",
) -> str:
    """
    Generate saran karir konkret dan personal menggunakan LLM (SDG 4.4).

    Returns:
        String saran karir yang spesifik dan actionable
    """
    student_summary = _build_student_summary(features)

    # Identifikasi kelemahan spesifik untuk saran yang lebih personal
    mot = int(features.get("motivation_score", 5))
    att = float(features.get("attendance_rate", 75))
    scores = float(features.get("test_scores_avg", 65))
    alignment = int(features.get("career_alignment", 2))
    backlogs = int(features.get("backlogs", 0))
    teaching = int(features.get("teaching_quality_rating", 5))
    internet = int(features.get("internet_connectivity_issues", 0))

    context_hints = []
    if mot <= 4:
        context_hints.append("motivasi rendah, perlu koneksi ke tujuan karir")
    if att < 70:
        context_hints.append("kehadiran rendah, saran harus bisa dilakukan dari rumah")
    if alignment == 1:
        context_hints.append("karir belum jelas, perlu eksplorasi jurusan/minat")
    if backlogs >= 3:
        context_hints.append("banyak tunggakan, saran karir jangka menengah")
    if teaching <= 4:
        context_hints.append("pengajaran dinilai buruk, dorong belajar mandiri")
    if internet == 2:
        context_hints.append("internet sering bermasalah, sarankan resource offline/downloadable")

    hints_text = "; ".join(context_hints) if context_hints else "kondisi cukup baik"

    prompt_content = f"""Mahasiswa dengan tingkat risiko dropout {risk_level.upper()} membutuhkan saran karir.

PROFIL:
{student_summary}

KONTEKS KHUSUS: {hints_text}

Berikan saran pengembangan karir yang SPESIFIK dan ACTIONABLE dalam 3-4 kalimat.
Sebutkan:
1. Satu platform/kursus spesifik yang relevan (nama nyata, gratis/terjangkau)
2. Satu keterampilan konkret yang harus dikembangkan bulan ini
3. Satu langkah karir yang bisa dimulai minggu ini

Pastikan saranmu realistis sesuai kondisi mahasiswa (nilai, motivasi, kehadiran).
Sesuaikan dengan SDG 4.4 (keterampilan kerja, TIK, kewirausahaan).
Gunakan Bahasa Indonesia yang memotivasi."""

    messages = [
        {"role": "system", "content": _SYSTEM_CONTEXT},
        {"role": "user", "content": prompt_content},
    ]

    # Coba Groq
    result = _call_groq(messages, model=GROQ_MODEL, max_tokens=300, temperature=0.75)
    if result:
        return _sanitize(result, max_chars=700)

    # Fallback HuggingFace
    hf_prompt = f"<s>[INST] {_SYSTEM_CONTEXT}\n\n{prompt_content} [/INST]"
    result = _call_hf_inference(hf_prompt, model_key="small", max_new_tokens=250)
    if result:
        return _sanitize(result, max_chars=700)

    # Fallback rule-based
    return _CAREER_FALLBACK.get(risk_level, _CAREER_FALLBACK["sedang"])


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CHATBOT PENDAMPING MAHASISWA — SDG 4.7 (LLM-powered)
# ═══════════════════════════════════════════════════════════════════════════════

_CHATBOT_SYSTEM = """Kamu adalah StayLearn Assistant, chatbot pendamping mahasiswa cerdas dari Telkom University Purwokerto.
Peranmu: membantu mahasiswa dengan pertanyaan seputar akademik, karir, beasiswa, kesehatan mental, dan SDG.

Aturan:
- Selalu jawab dalam Bahasa Indonesia yang hangat dan suportif
- Berikan informasi yang AKURAT dan BERGUNA, bukan generik
- Untuk pertanyaan beasiswa: sebutkan nama program nyata (KIP-Kuliah, LPDP, Bidikmisi, dll.)
- Untuk pertanyaan karir: sebutkan platform nyata (Dicoding, LinkedIn, Glints, dll.)
- Untuk kesehatan mental: empati dulu, lalu berikan langkah konkret
- Untuk pertanyaan di luar konteks: akui keterbatasan dan arahkan ke sumber yang tepat
- Jangan pernah menyarankan hal berbahaya
- Respons maksimal 4-5 kalimat, padat dan bermakna
- Kamu bisa membahas semua topik yang relevan untuk mahasiswa Indonesia"""

# Simpan history per session (key = session ID)
_chat_histories: Dict[str, List[Dict[str, str]]] = {}
_MAX_HISTORY = 10  # max turns per session


def chat(
    message: str,
    session_id: str = "default",
) -> Dict[str, str]:
    """
    Generate respons chatbot menggunakan LLM.

    Args:
        message:    Pesan dari mahasiswa
        session_id: ID sesi untuk menyimpan history percakapan

    Returns:
        Dict dengan keys: response (str), model (str)
    """
    if not message or not message.strip():
        return {"response": "Silakan ketik pertanyaanmu!", "model": "none"}

    # Sanitize input
    clean_msg = re.sub(r"<[^>]+>", "", message).strip()[:500]

    # Init history jika belum ada
    if session_id not in _chat_histories:
        _chat_histories[session_id] = []

    history = _chat_histories[session_id]

    # Bangun messages array
    messages = [{"role": "system", "content": _CHATBOT_SYSTEM}]
    messages.extend(history[-_MAX_HISTORY:])  # ambil history terbaru
    messages.append({"role": "user", "content": clean_msg})

    # Coba Groq dengan model lebih besar untuk chatbot
    result = _call_groq(
        messages,
        model=GROQ_MODEL_CHAT if _get_groq_key() else GROQ_MODEL,
        max_tokens=400,
        temperature=0.8,
    )

    if result:
        clean_result = _sanitize(result, max_chars=800)
        # Update history
        history.append({"role": "user", "content": clean_msg})
        history.append({"role": "assistant", "content": clean_result})
        # Trim history
        if len(history) > _MAX_HISTORY * 2:
            _chat_histories[session_id] = history[-(  _MAX_HISTORY * 2):]
        return {"response": clean_result, "model": "groq/" + (GROQ_MODEL_CHAT if _get_groq_key() else GROQ_MODEL)}

    # Fallback HuggingFace — bangun prompt conversation
    conv_prompt = f"<s>[INST] {_CHATBOT_SYSTEM} [/INST]\n"
    for turn in history[-4:]:  # ambil 4 turn terakhir untuk HF
        role = "Mahasiswa" if turn["role"] == "user" else "Assistant"
        conv_prompt += f"{role}: {turn['content']}\n"
    conv_prompt += f"[INST] {clean_msg} [/INST]\n"

    result = _call_hf_inference(conv_prompt, model_key="chat", max_new_tokens=300)
    if result:
        clean_result = _sanitize(result, max_chars=800)
        history.append({"role": "user", "content": clean_msg})
        history.append({"role": "assistant", "content": clean_result})
        return {"response": clean_result, "model": "hf/" + HF_MODELS["chat"]}

    # Final fallback — minimal rule-based hanya untuk tidak crash
    fallback_responses = {
        "beasiswa": "Untuk beasiswa, cek: KIP-Kuliah (kip-kuliah.kemdikbud.go.id), LPDP (lpdp.kemenkeu.go.id), dan beasiswa internal kampusmu. Hubungi bagian kemahasiswaan untuk info deadline.",
        "stres": "Saya dengar kamu sedang stres. Coba teknik 4-7-8: tarik napas 4 detik, tahan 7 detik, lepas 8 detik. Jika berlanjut, jangan ragu konsultasi ke psikolog kampus — itu tanda kekuatan, bukan kelemahan.",
        "karir": "Mulai dari Dicoding (dicoding.com) untuk keterampilan teknis, atau LinkedIn Learning untuk soft skills. Buat profil LinkedIn sekarang dan aktif terhubung dengan alumni.",
        "default": "Maaf, koneksi AI sedang terbatas. Untuk bantuan segera, hubungi konselor kampus atau cek portal mahasiswa. Coba lagi dalam beberapa menit!",
    }

    msg_lower = clean_msg.lower()
    for key in fallback_responses:
        if key in msg_lower:
            return {"response": fallback_responses[key], "model": "fallback"}
    return {"response": fallback_responses["default"], "model": "fallback"}


def reset_chat_session(session_id: str = "default") -> None:
    """Hapus history percakapan sesi tertentu."""
    _chat_histories.pop(session_id, None)


# ─── Singleton-style init (dipanggil dari app.py) ─────────────────────────────

def get_ai_status() -> Dict[str, Any]:
    """
    Cek status koneksi ke AI backends.
    Digunakan oleh /api/health untuk monitoring.
    """
    api_key = _get_groq_key()
    hf_token = _get_hf_token()
    status = {
        "groq_configured": bool(api_key),
        "hf_configured": bool(hf_token),
        "groq_model": GROQ_MODEL,
        "groq_chat_model": GROQ_MODEL_CHAT,
    }

    # Quick ping ke Groq jika ada key
    if api_key:
        try:
            resp = requests.post(
                GROQ_BASE_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": GROQ_MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5},
                timeout=8,
            )
            status["groq_online"] = resp.status_code == 200
            if resp.status_code != 200:
                status["groq_error"] = resp.text[:100]
        except Exception as e:
            status["groq_online"] = False
            status["groq_error"] = str(e)[:100]
    else:
        status["groq_online"] = False
        status["groq_error"] = "GROQ_API_KEY tidak diset"

    return status
