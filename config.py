import os
from pathlib import Path

from dotenv import load_dotenv
import google.generativeai as genai


# =====================================================
# Paths
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


# =====================================================
# Load environment variables
# =====================================================

load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set.\n"
        "Copy .env.example to .env in the project root and add your key:\n"
        "  GEMINI_API_KEY=your_key_here"
    )

genai.configure(api_key=GEMINI_API_KEY)

GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME","gemini-3.6-flash")


# =====================================================
# Shared model factory
# =====================================================

def get_model():
    """Return a configured Gemini model instance. Each agent module
    calls this once at import time and reuses the returned object."""
    return genai.GenerativeModel(GEMINI_MODEL_NAME)
