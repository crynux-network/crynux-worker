from typing import Literal

from pydantic import BaseModel, Field


class SuccessResult(BaseModel):
    status: Literal["success"]


class ErrorResult(BaseModel):
    status: Literal["error"]
    traceback: str


class TaskResult(BaseModel):
    task_name: Literal["inference", "download"]
    task_id_commitment: str
    result: SuccessResult | ErrorResult = Field(discriminator="status")
