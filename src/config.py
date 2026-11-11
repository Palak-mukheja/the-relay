import os
from pathlib import Path
from dotenv import load_dotenv

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env
load_dotenv(BASE_DIR / ".env")

# Groq configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

DEFAULT_MODEL = "openai/gpt-oss-20b"

# Request settings
REQUEST_TIMEOUT_SECONDS = 60
MAX_COMPLETION_TOKENS = 64
REASONING_EFFORT = "low"

# Results directory
RESULTS_DIR = BASE_DIR / "results"
RESULTS_FILE = RESULTS_DIR / "results.csv"
