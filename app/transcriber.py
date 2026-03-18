"""
transcriber.py — Groq Whisper STT
Free tier: ~28,800 sec/day.  Get key: https://console.groq.com
Returns "" if key not set — app continues without transcript.
"""
from __future__ import annotations
import logging, os, tempfile
import requests
from app.config import GROQ_API_KEY

log = logging.getLogger("transcriber")
GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"


def transcribe(audio_bytes: bytes, content_type: str = "audio/wav") -> str:
    if not GROQ_API_KEY:
        return ""
    ext = ("webm" if "webm" in content_type else
           "ogg"  if "ogg"  in content_type else
           "mp4"  if "mp4"  in content_type else "wav")
    try:
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
            tmp.write(audio_bytes)
            path = tmp.name
        with open(path, "rb") as f:
            r = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                files={"file": (f"audio.{ext}", f, content_type)},
                data={"model": "whisper-large-v3", "language": "en", "response_format": "text"},
                timeout=20,
            )
        os.unlink(path)
        if r.status_code == 200:
            return r.text.strip()
        log.warning(f"Groq STT {r.status_code}: {r.text[:80]}")
        return ""
    except Exception as e:
        log.warning(f"Transcription error: {e}")
        return ""
