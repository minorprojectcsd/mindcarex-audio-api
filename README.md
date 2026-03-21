# 🎙️ MindCareX — Voice Analysis Service (SVC1)

> Real-time voice stress analysis for mental health consultations.
> Analyses audio every 7 seconds during a doctor-patient video session.

---

## What This Service Does

During a live video consultation, the frontend captures the patient's microphone audio every 7 seconds and sends it here. This service:

1. Extracts 13 acoustic features using `librosa` (pitch, entropy, silence, MFCC etc.)
2. Converts speech to text using Groq Whisper
3. Computes a stress score 0–100
4. Saves everything to Neon PostgreSQL
5. Broadcasts the result instantly to the doctor's screen via WebSocket

The doctor sees a live stress overlay updating throughout the session.

---

## Stress Score Formula

Pure acoustic analysis (no HF dependency):

```
stress = pitch_variability × 0.20
       + silence_pattern   × 0.15
       + spectral_entropy  × 0.15
       + pitch_mean        × 0.15
       + zero_crossing     × 0.15
       + rms_energy        × 0.10
       + spectral_centroid × 0.10
```

| Score | State | Color |
|-------|-------|-------|
| 72–100 | High Stress | 🔴 Red |
| 50–71 | Moderate Stress | 🟠 Orange |
| 30–49 | Mild Stress | 🟡 Yellow |
| 0–29 | Calm / Relaxed | 🟢 Green |

---

## File Structure

```
svc1/
├── main.py                  FastAPI app, creates DB tables on startup
├── requirements.txt
├── Dockerfile               python:3.11-slim + ffmpeg + libsndfile1
├── .env.example
└── app/
    ├── config.py            env var loading
    ├── database.py          Neon SQLAlchemy engine, pool_size=5
    ├── models.py            VoiceSession + VoiceChunk ORM tables
    ├── schemas.py           Pydantic request/response models
    ├── audio_analyzer.py    librosa features + stress score computation
    ├── transcriber.py       Groq Whisper speech-to-text
    ├── session_manager.py   all DB reads/writes + session summary
    └── router.py            all REST endpoints + WebSocket broadcast
```

---

## Database Tables

Creates and owns these two tables in the shared Neon DB:

**`voice_sessions`**
```
id, patient_id, label, status, started_at, ended_at,
full_transcript, summary_json
```

**`voice_chunks`**
```
id, session_id, chunk_index, timestamp_sec,
stress_score, mental_state, mental_state_label, color, risk_level,
acoustic_json, top_emotions_json, chunk_transcript,
emotion_stress, acoustic_stress, mode, processed_at
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service health check |
| `GET` | `/api/voice/status` | Shows loaded API keys (groq_stt) |
| `POST` | `/api/voice/session/start` | Start session. Body: `{patient_id, label}` → returns `session_id` |
| `POST` | `/api/voice/{id}/chunk` | Upload audio chunk (multipart: file). Returns stress score, transcript, acoustic data |
| `WS` | `/api/voice/{id}/live` | WebSocket — doctor screen receives each chunk result ~100ms after processing |
| `POST` | `/api/voice/session/stop` | End session, compute full summary, save to Neon |
| `GET` | `/api/voice/{id}` | Full session + all chunks |
| `GET` | `/api/voice/{id}/timeline` | Stress + pitch + entropy per chunk (for chart) |
| `GET` | `/api/voice/{id}/summary` | Aggregate: avg/peak stress, trend, state distribution, pitch summary |
| `GET` | `/api/voice/{id}/transcript` | Full transcript + per-chunk breakdown with timestamps |
| `GET` | `/api/voice/patient/{id}/history` | All sessions for a patient, newest first |

### WebSocket

Connect to `ws://host/api/voice/{session_id}/live` to receive real-time chunk results:

```json
{
  "event": "chunk_result",
  "session_id": "uuid",
  "stress_score": 57.5,
  "mental_state": "moderate_stress",
  "mental_state_label": "Moderate Stress",
  "color": "orange",
  "risk_level": "low",
  "chunk_transcript": "I'm feeling really anxious today",
  "mode": "acoustic_only",
  "total_chunks": 4
}
```

Send `"ping"` to keep connection alive — server replies `"pong"`.

### Chunk response shape

```json
{
  "success": true,
  "data": {
    "chunk_index": 3,
    "timestamp_sec": 21.0,
    "stress_score": 57.5,
    "mental_state": "moderate_stress",
    "mental_state_label": "Moderate Stress",
    "color": "orange",
    "risk_level": "low",
    "top_emotions": [],
    "acoustic": {
      "rms_energy": 0.0149,
      "zcr": 0.1586,
      "pitch_mean_hz": 213.5,
      "pitch_std_hz": 666.9,
      "spectral_entropy": 0.4834,
      "spectral_centroid": 0.3816,
      "speaking_rate": 7.23,
      "silence_ratio": 0.274,
      "voiced_fraction": 0.726,
      "duration_sec": 7.47,
      "mfcc_mean": [...]
    },
    "chunk_transcript": "Hi, I'm feeling low today.",
    "mode": "acoustic_only",
    "total_chunks": 4
  }
}
```

### Session stop / summary shape

```json
{
  "success": true,
  "data": {
    "session_id": "uuid",
    "status": "completed",
    "full_transcript": "full text of everything said",
    "summary": {
      "avg_stress_score": 57.5,
      "peak_stress_score": 72.1,
      "min_stress_score": 34.2,
      "trend": "worsening",
      "overall_risk_level": "medium",
      "dominant_label": "Moderate Stress",
      "total_chunks": 8,
      "total_duration_sec": 56.0,
      "state_distribution": {"moderate_stress": 5, "mild_stress": 3},
      "pitch_summary": {"mean_hz": 213.5, "std_hz": 28.1},
      "entropy_summary": {"mean": 0.483, "trend": "stable"},
      "transcript_word_count": 312
    }
  }
}
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | ✅ Yes | Neon PostgreSQL connection string |
| `GROQ_API_KEY` | ✅ Yes | Groq API key for Whisper STT. Free at console.groq.com |
| `ALLOWED_ORIGINS` | Optional | CORS origins. Default: `http://localhost:5173` |

### `.env.example`

```env
DATABASE_URL=postgresql://user:pass@ep-xxx.neon.tech/neondb?sslmode=require
GROQ_API_KEY=gsk_xxxxxxxxxxxx
ALLOWED_ORIGINS=https://mindcarex.vercel.app,http://localhost:5173
```

---

## Running Locally

```bash
cp .env.example .env        # fill in your keys
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Verify:
```bash
curl http://localhost:8000/health
# {"status":"ok","service":"svc1_voice_analysis"}

curl http://localhost:8000/api/voice/status
# {"groq_stt": true}
```

Test full flow:
```bash
# 1. Start session
curl -X POST http://localhost:8000/api/voice/session/start \
  -H "Content-Type: application/json" \
  -d '{"patient_id":"P001","label":"Test"}'

# 2. Send audio chunk (replace with real .wav file)
curl -X POST http://localhost:8000/api/voice/SESSION_ID/chunk \
  -F "file=@test.wav"

# 3. Stop session
curl -X POST http://localhost:8000/api/voice/session/stop \
  -H "Content-Type: application/json" \
  -d '{"session_id":"SESSION_ID"}'
```

---

## Docker

```bash
docker build -t mindcarex-svc1 .
docker run -p 8000:8000 --env-file .env mindcarex-svc1
```

The Dockerfile installs `ffmpeg` and `libsndfile1` for audio decoding.

---

## Notes

- **Audio emotion (HF)** was removed in March 2026 — HuggingFace deprecated `api-inference.huggingface.co`. Acoustic stress scoring is unaffected and still fully functional.
- **Supported audio formats**: WebM, WAV, MP3, OGG, M4A (ffmpeg handles conversion)
- **Chunk size**: 7 seconds recommended — long enough for meaningful acoustic features, short enough for real-time feedback
- **Render free tier**: ~150 MB RAM usage. Fits comfortably on free tier.
