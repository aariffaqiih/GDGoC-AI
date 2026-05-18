"""
career_advice.py — Thin wrapper untuk backward compatibility.
Fungsi utama sekarang ada di ai_engine.py.
"""
from ai_engine import generate_career_advice

def init_career_advice(model_name: str = None):
    """No-op: AI engine tidak memerlukan inisialisasi."""
    pass

def get_advice(student_data: dict) -> str:
    """Wrapper untuk backward compatibility."""
    return generate_career_advice(
        features=student_data,
        risk_level=student_data.get("risk_level", "sedang"),
    )
