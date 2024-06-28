from enum import Enum
from typing import Optional

from pydantic import BaseModel


class PayloadType(str, Enum):
    Text = "text"
    Json = "json"
    PNG = "png"
    Error = "error"


class WorkerPhase(str, Enum):
    Prefetch = "prefetch"
    InitInference = "init_inference"
    Inference = "inference"


class WorkerPayloadMessage(BaseModel):
    worker_phase: WorkerPhase
    has_payload: bool
    has_next: bool
    payload_type: Optional[PayloadType] = None
