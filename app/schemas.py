from pydantic import BaseModel


class StartSessionRequest(BaseModel):
    patient_id: str = "unknown"
    label:      str = "Voice Session"


class StopSessionRequest(BaseModel):
    session_id: str
