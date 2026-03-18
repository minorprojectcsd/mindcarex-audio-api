"""
audio_analyzer.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Layer 1 — Acoustic features via librosa (runs on server, ~80ms/chunk)
  rms_energy, zcr, pitch mean+std (YIN),
  spectral_entropy (Shannon — erratic speech = high entropy),
  spectral_centroid, MFCC×13, silence_ratio, speaking_rate

Layer 2 — Emotion via HuggingFace Inference API (free, no GPU)
  Model: ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
  Returns: fearful · angry · disgusted · sad · surprised · neutral · happy

Stress score 0-100:
  HF token set  → 60% emotion + 40% acoustic
  No token      → 100% acoustic (still works)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from __future__ import annotations

import logging
import os
import tempfile
import time

import numpy as np
import requests
from scipy.stats import entropy as scipy_entropy

from app.config import HF_API_TOKEN

log = logging.getLogger("audio_analyzer")

HF_URL = (
    "https://api-inference.huggingface.co/models/"
    "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
)

EMOTION_WEIGHT: dict[str, float] = {
    "fearful":   1.00,
    "angry":     0.85,
    "disgusted": 0.70,
    "sad":       0.55,
    "surprised": 0.30,
    "neutral":   0.05,
    "happy":    -0.20,
}
HIGH_RISK   = {"fearful"}
MEDIUM_RISK = {"angry", "disgusted", "sad"}

STATE_LABEL = {
    "calm":            "Calm / Relaxed",
    "mild_stress":     "Mild Stress",
    "moderate_stress": "Moderate Stress",
    "high_stress":     "High Stress",
}
STATE_COLOR = {
    "calm": "green", "mild_stress": "yellow",
    "moderate_stress": "orange", "high_stress": "red",
}


# ── Audio loading ─────────────────────────────────────────────────────────────

def load_audio(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    import librosa
    with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
        tmp.write(audio_bytes)
        path = tmp.name
    try:
        y, sr = librosa.load(path, sr=16000, mono=True)
    finally:
        os.unlink(path)
    return y, sr


# ── Acoustic feature extraction ───────────────────────────────────────────────

def extract_features(y: np.ndarray, sr: int) -> dict:
    import librosa

    duration   = len(y) / sr
    rms_frames = librosa.feature.rms(y=y)[0]
    rms        = float(np.mean(rms_frames))
    zcr        = float(np.mean(librosa.feature.zero_crossing_rate(y=y)[0]))

    # Pitch + variability
    f0         = librosa.yin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
    voiced     = f0[f0 > 0]
    pitch_mean = float(np.median(voiced)) if len(voiced) > 0 else 0.0
    pitch_std  = float(np.std(voiced))    if len(voiced) > 1 else 0.0

    # Spectral entropy — Shannon entropy of normalised power spectrum
    S               = np.abs(librosa.stft(y)) ** 2
    power_per_frame = S.sum(axis=0) + 1e-10
    S_norm          = S / power_per_frame
    frame_ents      = [scipy_entropy(S_norm[:, i] + 1e-10) for i in range(S_norm.shape[1])]
    spec_entropy    = min(float(np.mean(frame_ents)) / float(np.log(S.shape[0])), 1.0)

    # Spectral centroid normalised to 500-4000 Hz
    centroid      = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_norm = min(max((float(np.mean(centroid)) - 500) / 3500, 0.0), 1.0)

    # MFCC 13
    mfcc_mean = [round(float(v), 3) for v in np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)]

    # Silence + speaking rate
    threshold     = max(rms * 0.10, 0.003)
    silence_ratio = float(np.sum(rms_frames < threshold) / max(len(rms_frames), 1))
    win           = int(sr * 0.1)
    peaks         = sum(
        1 for i in range(0, len(y) - win, win)
        if float(np.sqrt(np.mean(y[i:i + win] ** 2))) > threshold
    )

    return {
        "rms_energy":        round(rms, 4),
        "zcr":               round(zcr, 4),
        "pitch_mean_hz":     round(pitch_mean, 1),
        "pitch_std_hz":      round(pitch_std, 1),
        "spectral_entropy":  round(spec_entropy, 4),
        "spectral_centroid": round(centroid_norm, 4),
        "speaking_rate":     round(peaks / max(duration, 0.1), 2),
        "silence_ratio":     round(silence_ratio, 3),
        "voiced_fraction":   round(1.0 - silence_ratio, 3),
        "duration_sec":      round(duration, 2),
        "mfcc_mean":         mfcc_mean,
    }


# ── HuggingFace emotion API ───────────────────────────────────────────────────

def classify_emotions(audio_bytes: bytes) -> list[dict]:
    if not HF_API_TOKEN:
        return []
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    try:
        r = requests.post(HF_URL, headers=headers, data=audio_bytes, timeout=30)
        if r.status_code == 503:
            log.info("HF model cold-starting — retrying in 15s")
            time.sleep(15)
            r = requests.post(HF_URL, headers=headers, data=audio_bytes, timeout=30)
        if r.status_code != 200:
            log.warning(f"HF {r.status_code}: {r.text[:80]}")
            return []
        result = r.json()
        if isinstance(result, list):
            return [{"label": e["label"].lower().strip(), "score": round(float(e["score"]), 4)} for e in result]
        return []
    except Exception as e:
        log.warning(f"HF error: {e}")
        return []


# ── Stress score ──────────────────────────────────────────────────────────────

def compute_stress(features: dict, emotions: list[dict]) -> dict:
    p, ps   = features["pitch_mean_hz"], features["pitch_std_hz"]
    zcr     = features["zcr"]
    rms     = features["rms_energy"]
    sil     = features["silence_ratio"]
    entr    = features["spectral_entropy"]
    cent    = features["spectral_centroid"]

    acoustic = (
        min(max((p - 80) / 220, 0.0), 1.0) if p > 0 else 0.45
    ) * 0.15 + min(ps / 80, 1.0) * 0.20 + min(zcr / 0.12, 1.0) * 0.15 \
      + min(rms / 0.25, 1.0) * 0.10 + min(abs(sil - 0.35) / 0.65, 1.0) * 0.15 \
      + entr * 0.15 + cent * 0.10
    acoustic *= 100

    emotion_stress = 0.0
    if emotions:
        emo_map = {e["label"]: e["score"] for e in emotions}
        raw = sum(emo_map.get(e, 0.0) * w for e, w in EMOTION_WEIGHT.items())
        emotion_stress = max(0.0, min(raw * 100, 100.0))

    final = (0.60 * emotion_stress + 0.40 * acoustic) if emotions else acoustic
    final = round(min(max(final, 0.0), 100.0), 1)

    state = ("high_stress"     if final >= 72 else
             "moderate_stress" if final >= 50 else
             "mild_stress"     if final >= 30 else "calm")

    emo_labels = {e["label"] for e in emotions if e["score"] > 0.30}
    risk = ("high"   if emo_labels & HIGH_RISK   else
            "medium" if emo_labels & MEDIUM_RISK  else
            "medium" if final >= 65               else "low")

    return {
        "stress_score":       final,
        "mental_state":       state,
        "mental_state_label": STATE_LABEL[state],
        "color":              STATE_COLOR[state],
        "risk_level":         risk,
        "top_emotions":       sorted(emotions, key=lambda x: x["score"], reverse=True)[:5],
        "emotion_stress":     round(emotion_stress, 1),
        "acoustic_stress":    round(acoustic, 1),
    }


# ── Full pipeline ─────────────────────────────────────────────────────────────

def process_chunk(audio_bytes: bytes, chunk_index: int, timestamp_sec: float) -> dict:
    y, sr    = load_audio(audio_bytes)
    features = extract_features(y, sr)
    emotions = classify_emotions(audio_bytes)
    stress   = compute_stress(features, emotions)
    return {
        "chunk_index":        chunk_index,
        "timestamp_sec":      timestamp_sec,
        "stress_score":       stress["stress_score"],
        "mental_state":       stress["mental_state"],
        "mental_state_label": stress["mental_state_label"],
        "color":              stress["color"],
        "risk_level":         stress["risk_level"],
        "top_emotions":       stress["top_emotions"],
        "acoustic":           features,
        "emotion_stress":     stress["emotion_stress"],
        "acoustic_stress":    stress["acoustic_stress"],
        "mode":               "full" if emotions else "acoustic_only",
    }
