from __future__ import annotations
from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from app.database import Base


class VoiceSession(Base):
    __tablename__ = "voice_sessions"

    id              = Column(String,   primary_key=True)
    patient_id      = Column(String,   nullable=False, index=True)
    label           = Column(String,   default="Voice Session")
    status          = Column(String,   default="recording")   # recording | completed
    started_at      = Column(DateTime, default=datetime.utcnow)
    ended_at        = Column(DateTime, nullable=True)
    full_transcript = Column(Text,     default="")
    summary_json    = Column(JSON,     nullable=True)

    chunks = relationship(
        "VoiceChunk",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="VoiceChunk.chunk_index",
        lazy="select",
    )

    def to_dict(self) -> dict:
        return {
            "session_id":      self.id,
            "patient_id":      self.patient_id,
            "label":           self.label,
            "status":          self.status,
            "started_at":      self.started_at.isoformat() if self.started_at else None,
            "ended_at":        self.ended_at.isoformat()   if self.ended_at   else None,
            "full_transcript": self.full_transcript or "",
            "summary":         self.summary_json,
            "chunk_count":     len(self.chunks) if self.chunks else 0,
        }


class VoiceChunk(Base):
    __tablename__ = "voice_chunks"

    id                 = Column(Integer, primary_key=True, autoincrement=True)
    session_id         = Column(String,  ForeignKey("voice_sessions.id"), nullable=False, index=True)
    chunk_index        = Column(Integer, nullable=False)
    timestamp_sec      = Column(Float,   default=0.0)
    processed_at       = Column(DateTime, default=datetime.utcnow)
    stress_score       = Column(Float,  default=0.0)
    mental_state       = Column(String, default="calm")
    mental_state_label = Column(String, default="Calm / Relaxed")
    color              = Column(String, default="green")
    risk_level         = Column(String, default="low")
    mode               = Column(String, default="acoustic_only")
    acoustic_json      = Column(JSON,   nullable=True)
    top_emotions_json  = Column(JSON,   nullable=True)
    emotion_stress     = Column(Float,  default=0.0)
    acoustic_stress    = Column(Float,  default=0.0)
    chunk_transcript   = Column(Text,   default="")

    session = relationship("VoiceSession", back_populates="chunks")

    def to_dict(self) -> dict:
        return {
            "chunk_index":        self.chunk_index,
            "timestamp_sec":      self.timestamp_sec,
            "stress_score":       self.stress_score,
            "mental_state":       self.mental_state,
            "mental_state_label": self.mental_state_label,
            "color":              self.color,
            "risk_level":         self.risk_level,
            "mode":               self.mode,
            "acoustic":           self.acoustic_json or {},
            "top_emotions":       self.top_emotions_json or [],
            "emotion_stress":     self.emotion_stress,
            "acoustic_stress":    self.acoustic_stress,
            "chunk_transcript":   self.chunk_transcript or "",
            "processed_at":       self.processed_at.isoformat() if self.processed_at else None,
        }
