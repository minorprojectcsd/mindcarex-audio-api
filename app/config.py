import os
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

# 🔥 THIS is the key fix
load_dotenv(dotenv_path=ENV_PATH, override=True)

DATABASE_URL: str = os.getenv("DATABASE_URL", "")

HF_API_TOKEN: str = os.getenv("HF_API_TOKEN", "")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

print("USING DB:", DATABASE_URL)  # debug

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set.\n"
        "Add your Neon connection string in svc2/.env"
    )