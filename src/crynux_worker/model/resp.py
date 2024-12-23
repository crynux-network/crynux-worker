from pydantic import BaseModel, Field


class TaskSuccessResponse(BaseModel):
    status: str = Field(default="success", init=False)
    task_name: str
    task_id_commitment: str


class TaskErrorResponse(BaseModel):
    status: str = Field(default="error", init=False)
    task_name: str
    task_id_commitment: str
    traceback: str
