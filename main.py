"""
SVC 1 — Voice Analysis
Run locally:  uvicorn main:app --reload --port 8000
Docker:       see Dockerfile
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import init_db
from app.router   import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()   # creates voice_sessions + voice_chunks tables in Neon
    yield


app = FastAPI(
    title="Voice Analysis Service",
    version="1.0.0",
    description="Real-time voice stress analysis for doctor consultation sessions",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/voice")


# Convenience — health check at root too
@app.get("/health")
def health():
    return {"status": "ok", "service": "svc1_voice_analysis"}
