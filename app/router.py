"""
router.py — All voice analysis endpoints

REST
  GET  /health
  GET  /status
  POST /session/start
  POST /session/stop
  POST /{session_id}/chunk
  GET  /{session_id}
  GET  /{session_id}/timeline
  GET  /{session_id}/summary
  GET  /{session_id}/transcript
  GET  /patient/{patient_id}/history

WebSocket
  WS   /{session_id}/live   ← pushes each chunk result to doctor screen
"""
from __future__ import annotations
import asyncio
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.database       import get_db
from app.models         import VoiceSession
from app.schemas        import StartSessionRequest, StopSessionRequest
from app                import audio_analyzer as ana
from app                import transcriber    as stt
from app                import session_manager as mgr

router = APIRouter()
log    = logging.getLogger("router")


# ── WebSocket broadcast manager ───────────────────────────────────────────────

class _WS:
    def __init__(self):
        self._c: dict[str, list[WebSocket]] = {}

    async def connect(self, sid: str, ws: WebSocket):
        await ws.accept()
        self._c.setdefault(sid, []).append(ws)

    def drop(self, sid: str, ws: WebSocket):
        if ws in self._c.get(sid, []):
            self._c[sid].remove(ws)

    async def push(self, sid: str, data: dict):
        for ws in list(self._c.get(sid, [])):
            try:
                await ws.send_json(data)
            except Exception:
                self.drop(sid, ws)


_ws = _WS()


# ── Helpers ───────────────────────────────────────────────────────────────────

def ok(data: dict, code: int = 200) -> JSONResponse:
    return JSONResponse({"success": True, "data": data}, status_code=code)

def err(msg: str, code: int = 400) -> JSONResponse:
    return JSONResponse({"success": False, "error": msg}, status_code=code)

def get_or_404(db: Session, sid: str) -> VoiceSession:
    s = mgr.get_session(db, sid)
    if not s:
        raise HTTPException(404, f"Session {sid} not found")
    return s


# ══════════════════════════════════════════════════════════════════════════════
# GET /health  |  GET /status
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/health")
def health():
    return {"status": "ok"}

@router.get("/status")
def status():
    return ok({
        "service":    "svc1_voice_analysis",
        "groq_stt":   bool(stt.GROQ_API_KEY),
        "hf_emotion": bool(ana.HF_API_TOKEN),
        "note":       "acoustic-only mode works without API keys",
    })


# ══════════════════════════════════════════════════════════════════════════════
# POST /session/start
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/session/start", status_code=201)
def start_session(body: StartSessionRequest, db: Annotated[Session, Depends(get_db)]):
    s = mgr.create_session(db, body.patient_id, body.label)
    return ok({
        "session_id": s.id,
        "patient_id": body.patient_id,
        "label":      body.label,
        "status":     "recording",
        "hint":       f"POST audio every 5-10s to /{s.id}/chunk",
    }, 201)


# ══════════════════════════════════════════════════════════════════════════════
# POST /{session_id}/chunk
# form-data: file=<audio>  chunk_index=-1  timestamp_sec=-1
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/{session_id}/chunk")
async def upload_chunk(
    session_id:    str,
    db:            Annotated[Session, Depends(get_db)],
    file:          UploadFile = File(...),
    chunk_index:   int   = Form(default=-1),
    timestamp_sec: float = Form(default=-1.0),
):
    s            = get_or_404(db, session_id)
    audio_bytes  = await file.read()
    content_type = file.content_type or "audio/wav"

    if s.status == "completed":
        return err("Session already completed")
    if len(audio_bytes) < 200:
        return err("Audio chunk too small (< 200 bytes)")

    existing = len(s.chunks)
    if chunk_index < 0:
        chunk_index = existing
    if timestamp_sec < 0:
        timestamp_sec = sum((c.acoustic_json or {}).get("duration_sec", 0) for c in s.chunks)

    # Run both tasks concurrently in a thread pool (non-blocking)
    loop = asyncio.get_event_loop()
    try:
        chunk_data, transcript = await asyncio.gather(
            loop.run_in_executor(None, ana.process_chunk, audio_bytes, chunk_index, timestamp_sec),
            loop.run_in_executor(None, stt.transcribe,    audio_bytes, content_type),
        )
    except Exception as e:
        log.error(f"Pipeline error: {e}")
        return err(f"Analysis failed: {e}", 500)

    chunk = mgr.append_chunk(db, s, chunk_data, transcript)

    result = {
        "session_id":        session_id,
        "chunk_index":       chunk.chunk_index,
        "timestamp_sec":     chunk.timestamp_sec,
        "stress_score":      chunk.stress_score,
        "mental_state":      chunk.mental_state,
        "mental_state_label": chunk.mental_state_label,
        "color":             chunk.color,
        "risk_level":        chunk.risk_level,
        "top_emotions":      chunk.top_emotions_json or [],
        "acoustic":          chunk.acoustic_json or {},
        "chunk_transcript":  chunk.chunk_transcript,
        "mode":              chunk.mode,
        "total_chunks":      existing + 1,
    }

    # Broadcast to live WebSocket clients (doctor screen)
    await _ws.push(session_id, {"event": "chunk_result", **result})

    return ok(result)


# ══════════════════════════════════════════════════════════════════════════════
# POST /session/stop
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/session/stop")
def stop_session(body: StopSessionRequest, db: Annotated[Session, Depends(get_db)]):
    s       = get_or_404(db, body.session_id)
    summary = mgr.finalise_session(db, s)
    return ok({
        "session_id":      body.session_id,
        "status":          "completed",
        "full_transcript": s.full_transcript,
        "summary":         summary,
        "next_step":       "POST to svc2 /report/generate with this session_id",
    })


# ══════════════════════════════════════════════════════════════════════════════
# WS /{session_id}/live  — real-time chunk push to doctor UI
# ══════════════════════════════════════════════════════════════════════════════

@router.websocket("/{session_id}/live")
async def ws_live(session_id: str, ws: WebSocket):
    """
    Receives each chunk result within ~100ms of processing.
    Message shape:
      { event, session_id, chunk_index, stress_score,
        mental_state, color, risk_level, chunk_transcript, ... }
    """
    await _ws.connect(session_id, ws)
    try:
        while True:
            msg = await ws.receive_text()
            if msg == "ping":
                await ws.send_text("pong")
    except WebSocketDisconnect:
        _ws.drop(session_id, ws)


# ══════════════════════════════════════════════════════════════════════════════
# GET /{session_id}
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/{session_id}")
def get_session(session_id: str, db: Annotated[Session, Depends(get_db)]):
    s = get_or_404(db, session_id)
    return ok({**s.to_dict(), "chunks": [c.to_dict() for c in s.chunks]})


# ══════════════════════════════════════════════════════════════════════════════
# GET /{session_id}/timeline
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/{session_id}/timeline")
def get_timeline(session_id: str, db: Annotated[Session, Depends(get_db)]):
    s = get_or_404(db, session_id)
    return ok({
        "session_id": session_id,
        "count":      len(s.chunks),
        "timeline": [
            {
                "chunk_index":   c.chunk_index,
                "timestamp_sec": c.timestamp_sec,
                "stress_score":  c.stress_score,
                "mental_state":  c.mental_state,
                "label":         c.mental_state_label,
                "color":         c.color,
                "risk_level":    c.risk_level,
                "pitch_hz":      (c.acoustic_json or {}).get("pitch_mean_hz", 0),
                "entropy":       (c.acoustic_json or {}).get("spectral_entropy", 0),
                "transcript":    c.chunk_transcript or "",
            }
            for c in s.chunks
        ],
    })


# ══════════════════════════════════════════════════════════════════════════════
# GET /{session_id}/summary
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/{session_id}/summary")
def get_summary(session_id: str, db: Annotated[Session, Depends(get_db)]):
    s = get_or_404(db, session_id)
    return ok({
        "session_id": session_id,
        "status":     s.status,
        "summary":    s.summary_json or {"note": "Session still recording — stop it first"},
    })


# ══════════════════════════════════════════════════════════════════════════════
# GET /{session_id}/transcript
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/{session_id}/transcript")
def get_transcript(session_id: str, db: Annotated[Session, Depends(get_db)]):
    s = get_or_404(db, session_id)
    return ok({
        "session_id":        session_id,
        "full_transcript":   s.full_transcript or "",
        "word_count":        len((s.full_transcript or "").split()),
        "chunk_transcripts": [
            {"chunk_index": c.chunk_index, "timestamp_sec": c.timestamp_sec, "text": c.chunk_transcript}
            for c in s.chunks if c.chunk_transcript
        ],
    })


# ══════════════════════════════════════════════════════════════════════════════
# GET /patient/{patient_id}/history
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/patient/{patient_id}/history")
def get_history(patient_id: str, db: Annotated[Session, Depends(get_db)]):
    sessions = mgr.get_patient_sessions(db, patient_id)
    return ok({
        "patient_id":     patient_id,
        "total_sessions": len(sessions),
        "sessions": [
            {
                "session_id":  s.id,
                "label":       s.label,
                "status":      s.status,
                "started_at":  s.started_at.isoformat() if s.started_at else None,
                "ended_at":    s.ended_at.isoformat()   if s.ended_at   else None,
                "chunk_count": len(s.chunks),
                "summary":     s.summary_json,
            }
            for s in sessions
        ],
    })
