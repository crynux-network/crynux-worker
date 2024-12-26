from enum import IntEnum
from typing import Literal, List

from pydantic import BaseModel, Field


class TaskType(IntEnum):
    SD = 0
    LLM = 1
    SD_FT_LORA = 2


class ModelConfig(BaseModel):
    id: str
    type: Literal["base", "lora", "controlnet"]
    variant: str | None = None

    def to_model_id(self) -> str:
        model_body = self.id
        if self.variant:
            model_body += f"+{self.variant}"
        return f"{self.type}:{model_body}"
    
    @classmethod
    def from_model_id(cls, model_id: str) -> "ModelConfig":
        model_type, model_body = model_id.split(":", maxsplit=1)
        if "+" in model_body:
            model_name, variant = model_body.split("+", maxsplit=1)
        else:
            model_name = model_body
            variant = None
        return cls.model_validate({
            "id": model_name,
            "type": model_type,
            "variant": variant
        })


class DownloadTaskInput(BaseModel):
    task_name: Literal["download"]
    task_type: TaskType
    task_id: str
    model: ModelConfig


class InferenceTaskInput(BaseModel):
    task_name: Literal["inference"]
    task_type: TaskType
    task_id: str
    models: List[ModelConfig]
    task_args: str
    output_dir: str


class TaskInput(BaseModel):
    task: DownloadTaskInput | InferenceTaskInput = Field(discriminator="task_name")
