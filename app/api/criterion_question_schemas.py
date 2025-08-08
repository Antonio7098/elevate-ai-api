"""
Criterion Question Schemas - Data Transfer Objects for criterion-specific question generation.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class CriterionQuestionDto(BaseModel):
    """Data transfer object for criterion-specific questions."""
    question_id: str = Field(..., description="Unique question identifier")
    criterion_id: str = Field(..., description="Associated mastery criterion ID")
    primitive_id: str = Field(..., description="Source primitive ID")
    question_text: str = Field(..., description="The question text")
    question_type: str = Field(..., description="Type of question (short_answer, multiple_choice, etc.)")
    correct_answer: str = Field(..., description="The correct answer")
    options: List[str] = Field(default_factory=list, description="Multiple choice options if applicable")
    explanation: str = Field(..., description="Explanation of the correct answer")
    total_marks: int = Field(default=10, description="Total marks for this question")
    similarity_score: float = Field(default=1.0, description="Similarity score to criterion")
    
    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "question_id": "q_001",
                "criterion_id": "crit_001", 
                "primitive_id": "prim_001",
                "question_text": "What is the main concept?",
                "question_type": "short_answer",
                "correct_answer": "The main concept is...",
                "options": [],
                "explanation": "This tests understanding of...",
                "total_marks": 10,
                "similarity_score": 0.85
            }
        }
