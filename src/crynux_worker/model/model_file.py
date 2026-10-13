from typing import Any, Dict

from pydantic import BaseModel, Field, field_validator

SOURCE_TYPE_HUGGINGFACE = "huggingface"
IMAGE_BASE_MODEL_FORMAT = "diffusers"
GGUF_COMPONENT_FORMAT = "gguf"


class ModelFileEntry(BaseModel):
    """One independently downloadable model source.

    An entry with a revision identifies exact files. An entry with an empty
    revision is a source of which one revision is downloaded."""

    source_type: str = SOURCE_TYPE_HUGGINGFACE
    repository: str
    revision: str = ""
    format: str = ""
    format_kwargs: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("format_kwargs", mode="before")
    @classmethod
    def _null_format_kwargs(cls, v: Any) -> Any:
        return {} if v is None else v

    @property
    def is_exact(self) -> bool:
        return self.revision != ""
