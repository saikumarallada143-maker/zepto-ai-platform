"""
src/schema.py — Task 4/5: Pydantic request/response models.
"""

from typing import List
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)  # chunk/doc IDs, empty for general_question
    confidence: float = Field(ge=0.0, le=1.0)
