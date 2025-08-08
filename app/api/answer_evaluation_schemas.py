# Sprint 32 API Schemas for Core API Compatible Answer Evaluation

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator

class PrismaCriterionEvaluationRequest(BaseModel):
    """Request schema for Core API Prisma criterion evaluation."""
    criterionId: str = Field(..., description="Core API criterion ID")
    criterionTitle: str = Field(..., description="Criterion title")
    criterionDescription: Optional[str] = Field(None, description="Criterion description")
    primitiveId: str = Field(..., description="Core API primitive ID")
    primitiveTitle: str = Field(..., description="Primitive title")
    primitiveType: str = Field(default="knowledge", description="Primitive type")
    ueeLevel: str = Field(..., description="UEE level (UNDERSTAND|USE|EXPLORE)")
    criterionWeight: float = Field(..., description="Criterion weight (1.0-5.0)")
    isRequired: bool = Field(default=True, description="Whether criterion is required")
    questionText: str = Field(..., description="Question text")
    questionType: str = Field(..., description="Question type")
    correctAnswer: str = Field(..., description="Correct answer")
    userAnswer: str = Field(..., description="User's answer")
    totalMarks: int = Field(..., description="Total marks available")
    markingCriteria: Optional[str] = Field(None, description="Marking criteria")
    
    @field_validator('ueeLevel')
    @classmethod
    def validate_uee_level(cls, v):
        valid_levels = ['UNDERSTAND', 'USE', 'EXPLORE']
        if v not in valid_levels:
            raise ValueError(f'UEE level must be one of: {", ".join(valid_levels)}')
        return v
    
    @field_validator('criterionWeight')
    @classmethod
    def validate_criterion_weight(cls, v):
        if not (1.0 <= v <= 5.0):
            raise ValueError('Criterion weight must be between 1.0 and 5.0')
        return v


class PrismaCriterionEvaluationResponse(BaseModel):
    """Response schema for Core API Prisma criterion evaluation."""
    success: bool = Field(..., description="Whether evaluation was successful")
    criterionId: str = Field(..., description="Core API criterion ID")
    primitiveId: str = Field(..., description="Core API primitive ID")
    ueeLevel: str = Field(..., description="UEE level")
    masteryScore: float = Field(..., description="Mastery score (0.0-1.0)")
    masteryLevel: str = Field(..., description="Mastery level (novice|developing|mastered)")
    marksAchieved: int = Field(..., description="Marks achieved")
    totalMarks: int = Field(..., description="Total marks available")
    feedback: str = Field(..., description="UEE-specific feedback")
    correctedAnswer: str = Field(..., description="Corrected answer")
    criterionWeight: float = Field(..., description="Criterion weight")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Evaluation metadata")
    
    @field_validator('masteryScore')
    @classmethod
    def validate_mastery_score(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError('Mastery score must be between 0.0 and 1.0')
        return v
    
    @field_validator('masteryLevel')
    @classmethod
    def validate_mastery_level(cls, v):
        valid_levels = ['novice', 'developing', 'mastered']
        if v not in valid_levels:
            raise ValueError(f'Mastery level must be one of: {", ".join(valid_levels)}')
        return v


class BatchCriterionEvaluationRequest(BaseModel):
    """Request schema for batch criterion evaluation."""
    evaluationRequests: List[PrismaCriterionEvaluationRequest] = Field(..., description="List of evaluation requests")
    
    @field_validator('evaluationRequests')
    @classmethod
    def validate_evaluation_requests(cls, v):
        if len(v) == 0:
            raise ValueError('At least one evaluation request is required')
        if len(v) > 30:  # Reasonable batch limit
            raise ValueError('Maximum 30 evaluation requests per batch')
        return v


class BatchCriterionEvaluationResponse(BaseModel):
    """Response schema for batch criterion evaluation."""
    success: bool = Field(..., description="Whether batch evaluation was successful")
    results: Dict[str, PrismaCriterionEvaluationResponse] = Field(..., description="Results by criterion ID")
    totalEvaluated: int = Field(..., description="Total evaluations completed")
    failedCount: int = Field(..., description="Number of failed evaluations")
    overallMasteryScore: float = Field(..., description="Overall mastery score across all criteria")
    errors: List[str] = Field(default_factory=list, description="Any evaluation errors")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Batch metadata")


class MasteryAssessmentRequest(BaseModel):
    """Request schema for comprehensive mastery assessment."""
    primitiveId: str = Field(..., description="Core API primitive ID")
    criterionEvaluations: List[PrismaCriterionEvaluationRequest] = Field(..., description="Criterion evaluations")
    
    @field_validator('criterionEvaluations')
    @classmethod
    def validate_criterion_evaluations(cls, v):
        if len(v) == 0:
            raise ValueError('At least one criterion evaluation is required')
        return v


class MasteryAssessmentResponse(BaseModel):
    """Response schema for comprehensive mastery assessment."""
    success: bool = Field(..., description="Whether assessment was successful")
    primitiveId: str = Field(..., description="Core API primitive ID")
    overallMasteryScore: float = Field(..., description="Overall weighted mastery score")
    criterionAssessments: List[Dict[str, Any]] = Field(..., description="Individual criterion assessments")
    ueeProgression: Dict[str, Dict[str, Any]] = Field(..., description="UEE level progression analysis")
    comprehensiveFeedback: str = Field(..., description="Comprehensive feedback across all criteria")
    masteryLevel: str = Field(..., description="Overall primitive mastery level")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Assessment metadata")
    
    @field_validator('overallMasteryScore')
    @classmethod
    def validate_overall_mastery_score(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError('Overall mastery score must be between 0.0 and 1.0')
        return v
    
    @field_validator('masteryLevel')
    @classmethod
    def validate_mastery_level(cls, v):
        valid_levels = ['beginner', 'developing', 'intermediate', 'advanced']
        if v not in valid_levels:
            raise ValueError(f'Mastery level must be one of: {", ".join(valid_levels)}')
        return v
