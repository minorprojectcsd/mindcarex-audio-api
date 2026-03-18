"""
session_manager.py — All Neon DB read/write for voice sessions.
"""
from __future__ import annotations
import uuid
from collections import defaultdict
from datetime import datetime

import numpy as np
from sqlalchemy.orm import Session

from app.models import VoiceSession, VoiceChunk
from app.audio_analyzer import STATE_LABEL


def create_session(db: Session, patient_id: str, label: str) -> VoiceSession:
    s = VoiceSession(
        id=str(uuid.uuid4()),
        patient_id=patient_id,
        label=label,
        status="recording",
        started_at=datetime.utcnow(),
        full_transcript="",
    )
    db.add(s)
    db.commit()
    db.refresh(s)
    return s


def get_session(db: Session, session_id: str) -> VoiceSession | None:
    return db.query(VoiceSession).filter(VoiceSession.id == session_id).first()


def get_patient_sessions(db: Session, patient_id: str) -> list[VoiceSession]:
    return (
        db.query(VoiceSession)
        .filter(VoiceSession.patient_id == patient_id)
        .order_by(VoiceSession.started_at.desc())
        .all()
    )


def append_chunk(db: Session, session: VoiceSession, data: dict, transcript: str) -> VoiceChunk:
    chunk = VoiceChunk(
        session_id=session.id,
        chunk_index=data["chunk_index"],
        timestamp_sec=data["timestamp_sec"],
        stress_score=data["stress_score"],
        mental_state=data["mental_state"],
        mental_state_label=data["mental_state_label"],
        color=data["color"],
        risk_level=data["risk_level"],
        mode=data.get("mode", "acoustic_only"),
        acoustic_json=data["acoustic"],
        top_emotions_json=data["top_emotions"],
        emotion_stress=data["emotion_stress"],
        acoustic_stress=data["acoustic_stress"],
        chunk_transcript=transcript,
        processed_at=datetime.utcnow(),
    )
    db.add(chunk)
    if transcript:
        sep = " " if session.full_transcript else ""
        session.full_transcript = (session.full_transcript or "") + sep + transcript
    db.commit()
    db.refresh(chunk)
    return chunk


def finalise_session(db: Session, session: VoiceSession) -> dict:
    """Compute aggregate summary, write to Neon, mark completed."""
    chunks: list[VoiceChunk] = (
        db.query(VoiceChunk)
        .filter(VoiceChunk.session_id == session.id)
        .order_by(VoiceChunk.chunk_index)
        .all()
    )

    if not chunks:
        summary = {"error": "No chunks recorded"}
        session.status = "completed"
        session.ended_at = datetime.utcnow()
        session.summary_json = summary
        db.commit()
        return summary

    scores = [c.stress_score for c in chunks]
    avg, peak, low = round(sum(scores)/len(scores), 1), round(max(scores), 1), round(min(scores), 1)

    slope, trend = 0.0, "insufficient_data"
    if len(scores) >= 3:
        slope = float(np.polyfit(np.arange(len(scores), dtype=float), scores, 1)[0])
        trend = "worsening" if slope > 1.5 else ("improving" if slope < -1.5 else "stable")

    state_counts: dict[str, int] = defaultdict(int)
    for c in chunks:
        state_counts[c.mental_state] += 1
    dominant = max(state_counts, key=state_counts.get)

    risks = [c.risk_level for c in chunks]
    overall_risk = "high" if "high" in risks else ("medium" if "medium" in risks else "low")

    emo_t: dict[str, float] = defaultdict(float)
    emo_c: dict[str, int]   = defaultdict(int)
    for c in chunks:
        for e in (c.top_emotions_json or []):
            emo_t[e["label"]] += e["score"]
            emo_c[e["label"]] += 1
    top_emotions = sorted(
        [{"label": k, "avg_score": round(emo_t[k]/emo_c[k], 4)} for k in emo_t],
        key=lambda x: x["avg_score"], reverse=True
    )[:8]

    pitches = [(c.acoustic_json or {}).get("pitch_mean_hz", 0) for c in chunks if (c.acoustic_json or {}).get("pitch_mean_hz", 0) > 0]
    entropies = [(c.acoustic_json or {}).get("spectral_entropy", 0) for c in chunks]

    summary = {
        "avg_stress_score":      avg,
        "peak_stress_score":     peak,
        "min_stress_score":      low,
        "trend":                 trend,
        "trend_slope":           round(slope, 3),
        "dominant_mental_state": dominant,
        "dominant_label":        STATE_LABEL[dominant],
        "overall_risk_level":    overall_risk,
        "total_chunks":          len(chunks),
        "total_duration_sec":    round(sum((c.acoustic_json or {}).get("duration_sec", 0) for c in chunks), 1),
        "state_distribution":    dict(state_counts),
        "top_emotions":          top_emotions,
        "pitch_summary": {
            "mean_hz":  round(float(np.mean(pitches)), 1)  if pitches else 0.0,
            "std_hz":   round(float(np.std(pitches)),  1)  if pitches else 0.0,
            "min_hz":   round(float(np.min(pitches)),  1)  if pitches else 0.0,
            "max_hz":   round(float(np.max(pitches)),  1)  if pitches else 0.0,
            "contour":  [round(p, 1) for p in pitches],
        },
        "entropy_summary": {
            "mean":  round(float(np.mean(entropies)), 3) if entropies else 0.0,
            "max":   round(float(np.max(entropies)),  3) if entropies else 0.0,
            "trend": ("rising"  if len(entropies) >= 3 and entropies[-1] > entropies[0] else
                      "falling" if len(entropies) >= 3 and entropies[-1] < entropies[0] else "stable"),
        },
        "transcript_word_count": len((session.full_transcript or "").split()),
    }

    session.status       = "completed"
    session.ended_at     = datetime.utcnow()
    session.summary_json = summary
    db.commit()
    return summary
