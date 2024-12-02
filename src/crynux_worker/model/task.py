from enum import IntEnum

from pydantic import BaseModel


class TaskType(IntEnum):
    SD = 0
    LLM = 1
    SD_FT_LORA = 2


class TaskInput(BaseModel):
    task_id_commitment: str
    task_name: str
    task_type: TaskType
    task_args: str
