from pydantic import BaseModel, Field
from typing import Optional

class RouterResponse(BaseModel):
    """Response from the conversation router determining if retrieval is needed."""

    can_answer_from_history: bool = Field(
        ...,
        description="True ONLY if the complete answer exists in conversation history. "
                    "If ANY part of the question cannot be fully answered, this must be False."
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Your confidence level (0.0-1.0) that the answer from history is "
                    "complete, accurate, and current. Values below 0.8 should trigger retrieval."
    )

    answer: Optional[str] = Field(
        None,
        description="The answer extracted from conversation history. "
                    "ONLY populate if can_answer_from_history=True AND confidence >= 0.8. "
                    "Must be a complete, standalone response."
    )
