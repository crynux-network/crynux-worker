from enum import IntEnum
from typing import Literal

from pydantic import BaseModel, Field


class TaskType(IntEnum):
    SD = 0
    LLM = 1
    SD_FT_LORA = 2


class ModelConfig(BaseModel):
    id: str
    variant: str | None = None


class DownloadTaskInput(BaseModel):
    task_name: Literal["download"]
    task_type: TaskType
    task_id_commitment: str
    model_type: Literal["base", "vae", "controlnet"]
    model: ModelConfig


class InferenceTaskInput(BaseModel):
    task_name: Literal["inference"]
    task_type: TaskType
    task_id_commitment: str
    model_id: str
    task_args: str
    output_dir: str


class TaskInput(BaseModel):
    task: DownloadTaskInput | InferenceTaskInput = Field(discriminator="task_name")
