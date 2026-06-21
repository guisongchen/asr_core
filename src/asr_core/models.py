from typing import Literal, Optional

from pydantic import BaseModel, Field


class ModelStatus(BaseModel):
    state: Literal["unloaded", "loading", "loaded", "error"] = "unloaded"
    model_name: Optional[str] = None
    error_message: Optional[str] = None
    gpu_memory_mb: Optional[float] = None
    requests_total: int = 0
    requests_failed: int = 0


class LoadRequest(BaseModel):
    model_name: str = Field(default="qwen3-asr-0.6b")


class TranscribeRequest(BaseModel):
    audio_path: str
    model_name: Optional[str] = None
    language: Optional[str] = None


class TranscribeResponse(BaseModel):
    text: str
    detected_language: Optional[str] = None
    duration_seconds: Optional[float] = None


class ServiceStatus(BaseModel):
    name: str
    active: bool
    status: str
    uptime: Optional[str] = None
