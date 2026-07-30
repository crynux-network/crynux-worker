from typing import Literal

from pydantic import BaseModel, Field


class SuccessResult(BaseModel):
    status: Literal["success"]


class ErrorResult(BaseModel):
    status: Literal["error"]
    traceback: str
    # GPUs actually used for this failed task. TP uses the final world size
    # (after reduce_gpus when applicable); classic / SD use the visible count.
    gpu_count: int = 0


class TaskResult(BaseModel):
    task_name: Literal["inference", "download"]
    task_id_commitment: str
    result: SuccessResult | ErrorResult = Field(discriminator="status")
