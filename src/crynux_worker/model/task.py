from enum import Enum

from pydantic import BaseModel


class TaskType(int, Enum):
    SD = 0
    LLM = 1
    SD_FT_LORA = 2


class TaskInput(BaseModel):
    task_id: int
    task_name: str
    task_type: TaskType
    task_args: str
