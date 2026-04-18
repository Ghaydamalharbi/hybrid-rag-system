import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 🔥 لا توقف النظام
if not GEMINI_API_KEY:
    print("[WARNING] GEMINI_API_KEY not set — Gemini disabled")