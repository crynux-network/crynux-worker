from enum import IntEnum
from typing import Literal, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .model_file import ModelFileEntry


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
    model_config = ConfigDict(protected_namespaces=())

    task_name: Literal["download"]
    task_type: TaskType
    task_id: str
    # Task-scoped auxiliary downloads carry `model`; model file downloads carry
    # `model_files`.
    model: Optional[ModelConfig] = None
    model_files: List[ModelFileEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _require_one_target(self) -> "DownloadTaskInput":
        if (self.model is None) == (len(self.model_files) == 0):
            raise ValueError(
                "A download task requires exactly one of model and model_files"
            )
        return self


class InferenceTaskInput(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    task_name: Literal["inference"]
    task_type: TaskType
    task_id: str
    models: List[ModelConfig]
    task_args: str
    output_dir: str
    # Exact model files frozen by the Relay for this execution
    execution_model_files: List[ModelFileEntry] = Field(default_factory=list)
    # Relay-assigned identity of the model the task requests in memory
    requested_model_cache_hash: Optional[str] = None


class TaskInput(BaseModel):
    task: DownloadTaskInput | InferenceTaskInput = Field(discriminator="task_name")
