from enum import IntEnum
from typing import Literal

from pydantic import BaseModel


class TaskType(IntEnum):
    SD = 0
    LLM = 1
    SD_FT_LORA = 2


class ModelConfig(BaseModel):
    id: str
    variant: str | None = None


class DownloadTaskInput(BaseModel):
    task_type: TaskType
    task_id_commitment: str
    model_type: Literal["base", "vae", "controlnet"]
    model: ModelConfig


class InferenceTaskInput(BaseModel):
    task_type: TaskType
    task_id_commitment: str
    model_id: str
    task_args: str
    output_dir: str


class TaskInput(BaseModel):
    task_name: Literal["inference", "download"]
    task: DownloadTaskInput | InferenceTaskInput


class TaskResult(BaseModel):
    task_name: Literal["inference", "download"]
    task_id_commitment: str
    status: Literal["success", "error"]
    traceback: str | None = None
