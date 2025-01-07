from .result import ErrorResult, SuccessResult, TaskResult
from .input import DownloadTaskInput, InferenceTaskInput, TaskInput, TaskType

__all__ = [
    "TaskInput",
    "TaskType",
    "InferenceTaskInput",
    "DownloadTaskInput",
    "TaskResult",
    "SuccessResult",
    "ErrorResult",
]
