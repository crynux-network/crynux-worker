from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .model_file import ModelFileEntry


class SuccessResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: Literal["success"]
    execution_dtype: Optional[str] = None
    # Download results list the exact entry of every downloaded model file
    model_files: Optional[List[ModelFileEntry]] = None
    # Inference results return the requested model cache hash as the identity
    # of the model now held in memory
    model_cache_hash: Optional[str] = None


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
