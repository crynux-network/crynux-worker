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
    model_type: Literal["base", "vae", "controlnet"]
    model: ModelConfig


class InferenceTaskInput(BaseModel):
    task_type: TaskType
    task_id_commitment: str
    task_args: str
    output_dir: str


class TaskInput(BaseModel):
    task_name: str
    input: DownloadTaskInput | InferenceTaskInput


class TaskResult(BaseModel):
    status: Literal["success", "error"]
    traceback: str | None = None


class DownloadTaskResult(TaskResult):
    task_input: DownloadTaskInput


class InferenceTaskResult(TaskResult):
    task_input: InferenceTaskInput
