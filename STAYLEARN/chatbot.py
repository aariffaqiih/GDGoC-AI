"""
chatbot.py — Thin wrapper untuk backward compatibility.
Implementasi LLM nyata sekarang ada di ai_engine.py.
"""
from ai_engine import chat as ai_chat, reset_chat_session

class StudentChatbot:
    def respond(self, message: str) -> dict:
        result = ai_chat(message, session_id="legacy")
        return {
            "response": result["response"],
            "intent": "ai_generated",
            "method": result.get("model", "ai"),
        }
    def reset_conversation(self):
        reset_chat_session("legacy")

_instance = None
def get_chatbot() -> StudentChatbot:
    global _instance
    if _instance is None:
        _instance = StudentChatbot()
    return _instance
