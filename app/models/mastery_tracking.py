"""
Enhanced Mastery Tracking Models for AI API

This module defines models for mastery tracking that align with the Core API's
mastery system, including consecutive interval tracking, UUE stage progression,
and user-specific mastery configurations.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

from .blueprint_centric import UueStage, TrackingIntensity, DifficultyLevel


class MasteryThreshold(str, Enum):
    """Mastery threshold levels that match Core API."""
    SURVEY = "SURVEY"           # 60% - Basic understanding
    PROFICIENT = "PROFICIENT"   # 80% - Solid mastery
    EXPERT = "EXPERT"           # 95% - Expert level


class LearningStyle(str, Enum):
    """User learning style preferences."""
    CONSERVATIVE = "CONSERVATIVE"  # Slower, more thorough progression
    BALANCED = "BALANCED"         # Standard progression
    AGGRESSIVE = "AGGRESSIVE"     # Faster, more challenging progression


class ExperienceLevel(str, Enum):
    """User experience level."""
    BEGINNER = "BEGINNER"
    INTERMEDIATE = "INTERMEDIATE"
    ADVANCED = "ADVANCED"
    EXPERT = "EXPERT"


# User Mastery Configuration Models
class UserMasteryPreferences(BaseModel):
    """User-specific mastery configuration preferences."""
    user_id: int = Field(..., description="User ID")
    default_mastery_threshold: MasteryThreshold = Field(default=MasteryThreshold.PROFICIENT, description="Default mastery threshold")
    default_tracking_intensity: TrackingIntensity = Field(default=TrackingIntensity.NORMAL, description="Default tracking intensity")
    learning_style: LearningStyle = Field(default=LearningStyle.BALANCED, description="User's learning style")
    experience_level: ExperienceLevel = Field(default=ExperienceLevel.INTERMEDIATE, description="User's experience level")
    auto_adjustment: bool = Field(default=False, description="Whether to auto-adjust based on performance")
    daily_study_time: int = Field(default=60, description="Daily study time in minutes")
    preferred_uue_stages: List[UueStage] = Field(default_factory=lambda: [UueStage.UNDERSTAND, UueStage.USE, UueStage.EXPLORE], description="Preferred UUE stages")
    
    # Advanced settings
    consecutive_interval_requirement: int = Field(default=2, description="Number of consecutive intervals required for mastery")
    min_gap_days: int = Field(default=1, description="Minimum gap between attempts in days")
    max_review_frequency: int = Field(default=7, description="Maximum reviews per week")
    
    @field_validator('daily_study_time')
    @classmethod
    def validate_daily_study_time(cls, v):
        if v < 15 or v > 480:  # 15 minutes to 8 hours
            raise ValueError('Daily study time must be between 15 and 480 minutes')
        return v
    
    @field_validator('consecutive_interval_requirement')
    @classmethod
    def validate_consecutive_interval_requirement(cls, v):
        if v < 1 or v > 5:
            raise ValueError('Consecutive interval requirement must be between 1 and 5')
        return v
    
    @field_validator('min_gap_days')
    @classmethod
    def validate_min_gap_days(cls, v):
        if v < 0 or v > 7:
            raise ValueError('Minimum gap days must be between 0 and 7')
        return v


class SectionMasteryThreshold(BaseModel):
    """Section-specific mastery threshold configuration."""
    id: Optional[int] = Field(None, description="Database ID")
    user_id: int = Field(..., description="User ID")
    section_id: str = Field(..., description="Blueprint section ID")
    threshold: MasteryThreshold = Field(..., description="Mastery threshold for this section")
    threshold_value: float = Field(..., description="Numeric threshold value (0.6, 0.8, or 0.95)")
    description: str = Field(..., description="User's reason for choosing this threshold")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    
    @field_validator('threshold_value')
    @classmethod
    def validate_threshold_value(cls, v):
        valid_values = [0.6, 0.8, 0.95]
        if v not in valid_values:
            raise ValueError(f'Threshold value must be one of: {valid_values}')
        return v


class CriterionMasteryThreshold(BaseModel):
    """Criterion-specific mastery threshold configuration."""
    id: Optional[int] = Field(None, description="Database ID")
    user_id: int = Field(..., description="User ID")
    criterion_id: str = Field(..., description="Mastery criterion ID")
    threshold: MasteryThreshold = Field(..., description="Mastery threshold for this criterion")
    threshold_value: float = Field(..., description="Numeric threshold value")
    description: str = Field(..., description="User's reason for choosing this threshold")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    
    @field_validator('threshold_value')
    @classmethod
    def validate_threshold_value(cls, v):
        valid_values = [0.6, 0.8, 0.95]
        if v not in valid_values:
            raise ValueError(f'Threshold value must be one of: {valid_values}')
        return v


# Mastery Progress Tracking Models
class UserCriterionMastery(BaseModel):
    """User's mastery progress on a specific criterion."""
    id: Optional[str] = Field(None, description="Unique mastery record ID")
    user_id: int = Field(..., description="User ID")
    mastery_criterion_id: str = Field(..., description="Mastery criterion ID")
    blueprint_section_id: str = Field(..., description="Blueprint section ID")
    
    # Mastery tracking
    is_mastered: bool = Field(default=False, description="Whether the criterion is mastered")
    mastery_score: float = Field(default=0.0, description="Current mastery score (0.0-1.0)")
    uue_stage: UueStage = Field(default=UueStage.UNDERSTAND, description="Current UUE stage")
    
    # Consecutive interval tracking
    last_two_attempts: List[float] = Field(default_factory=list, description="Last 2 attempt scores")
    consecutive_intervals: int = Field(default=0, description="Count of consecutive intervals above threshold")
    last_threshold_check_date: Optional[datetime] = Field(None, description="Last date threshold was checked")
    
    # Spaced repetition tracking
    current_interval_step: int = Field(default=0, description="Current interval step")
    next_review_at: Optional[datetime] = Field(None, description="Next review date")
    last_reviewed_at: Optional[datetime] = Field(None, description="Last review date")
    review_count: int = Field(default=0, description="Total number of reviews")
    successful_reviews: int = Field(default=0, description="Number of successful reviews")
    consecutive_failures: int = Field(default=0, description="Number of consecutive failures")
    
    # Tracking intensity
    tracking_intensity: TrackingIntensity = Field(default=TrackingIntensity.NORMAL, description="Current tracking intensity")
    
    # Timestamps
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    
    @field_validator('mastery_score')
    @classmethod
    def validate_mastery_score(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError('Mastery score must be between 0.0 and 1.0')
        return v
    
    @field_validator('last_two_attempts')
    @classmethod
    def validate_last_two_attempts(cls, v):
        if len(v) > 2:
            raise ValueError('Last two attempts cannot have more than 2 scores')
        for score in v:
            if score < 0.0 or score > 1.0:
                raise ValueError('Attempt scores must be between 0.0 and 1.0')
        return v
    
    def add_attempt(self, score: float):
        """Add a new attempt score."""
        if score < 0.0 or score > 1.0:
            raise ValueError('Attempt score must be between 0.0 and 1.0')
        
        self.last_two_attempts.append(score)
        if len(self.last_two_attempts) > 2:
            self.last_two_attempts.pop(0)
        
        self.review_count += 1
        if score >= 0.8:  # Assuming 0.8 is the threshold for "successful"
            self.successful_reviews += 1
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
    
    def check_mastery(self, threshold: float) -> bool:
        """Check if the criterion should be marked as mastered."""
        if len(self.last_two_attempts) < 2:
            return False
        
        # Check if last 2 attempts are above threshold
        recent_attempts = self.last_two_attempts[-2:]
        if all(score >= threshold for score in recent_attempts):
            # Check if attempts are on different days
            if self.last_threshold_check_date:
                days_since_last_check = (datetime.now() - self.last_threshold_check_date).days
                if days_since_last_check >= 1:
                    self.consecutive_intervals += 1
                    self.last_threshold_check_date = datetime.now()
                    
                    # Mark as mastered if we have enough consecutive intervals
                    if self.consecutive_intervals >= 2:
                        self.is_mastered = True
                        return True
        
        return False


# Mastery Calculation Models
class MasteryCalculationRequest(BaseModel):
    """Request for mastery calculation."""
    user_id: int = Field(..., description="User ID")
    criterion_id: Optional[str] = Field(None, description="Specific criterion ID (optional)")
    section_id: Optional[str] = Field(None, description="Section ID for section-level calculation")
    blueprint_id: Optional[int] = Field(None, description="Blueprint ID for blueprint-level calculation")
    include_history: bool = Field(default=False, description="Include mastery history")
    include_recommendations: bool = Field(default=True, description="Include improvement recommendations")


class MasteryCalculationResult(BaseModel):
    """Result of mastery calculation."""
    user_id: int = Field(..., description="User ID")
    calculation_type: str = Field(..., description="Type of calculation performed")
    
    # Criterion-level results
    criterion_mastery: Optional[Dict[str, float]] = Field(None, description="Mastery scores by criterion ID")
    
    # Section-level results
    section_mastery: Optional[Dict[str, float]] = Field(None, description="Mastery scores by section ID")
    
    # Blueprint-level results
    blueprint_mastery: Optional[float] = Field(None, description="Overall blueprint mastery score")
    
    # UUE stage progression
    uue_stage_progress: Optional[Dict[UueStage, float]] = Field(None, description="Progress through UUE stages")
    
    # Statistics
    total_criteria: int = Field(..., description="Total number of criteria")
    mastered_criteria: int = Field(..., description="Number of mastered criteria")
    mastery_percentage: float = Field(..., description="Percentage of criteria mastered")
    
    # Recommendations
    recommendations: List[str] = Field(default_factory=list, description="Improvement recommendations")
    
    # Metadata
    calculation_timestamp: datetime = Field(default_factory=datetime.now, description="When calculation was performed")
    
    def add_recommendation(self, recommendation: str):
        """Add an improvement recommendation."""
        self.recommendations.append(recommendation)


# Performance Tracking Models
class MasteryPerformanceMetrics(BaseModel):
    """Performance metrics for mastery tracking."""
    user_id: int = Field(..., description="User ID")
    time_period: str = Field(..., description="Time period for metrics (daily, weekly, monthly)")
    
    # Review performance
    total_reviews: int = Field(default=0, description="Total reviews in period")
    successful_reviews: int = Field(default=0, description="Successful reviews in period")
    success_rate: float = Field(default=0.0, description="Review success rate")
    
    # Mastery progression
    criteria_mastered: int = Field(default=0, description="Criteria mastered in period")
    uue_stage_advancements: int = Field(default=0, description="UUE stage advancements in period")
    
    # Time efficiency
    average_review_time: float = Field(default=0.0, description="Average time per review in seconds")
    total_study_time: float = Field(default=0.0, description="Total study time in minutes")
    
    # Learning patterns
    preferred_study_times: List[str] = Field(default_factory=list, description="Preferred study time slots")
    difficulty_preferences: Dict[DifficultyLevel, int] = Field(default_factory=dict, description="Difficulty level preferences")
    
    # Timestamps
    period_start: datetime = Field(..., description="Period start timestamp")
    period_end: datetime = Field(..., description="Period end timestamp")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    
    @field_validator('success_rate')
    @classmethod
    def validate_success_rate(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError('Success rate must be between 0.0 and 1.0')
        return v
    
    def calculate_success_rate(self):
        """Calculate the review success rate."""
        if self.total_reviews > 0:
            self.success_rate = self.successful_reviews / self.total_reviews
        else:
            self.success_rate = 0.0


# Learning Path Models
class LearningPathNode(BaseModel):
    """Node in a learning path."""
    criterion_id: str = Field(..., description="Mastery criterion ID")
    uue_stage: UueStage = Field(..., description="UUE stage for this node")
    mastery_score: float = Field(default=0.0, description="Current mastery score")
    is_mastered: bool = Field(default=False, description="Whether this node is mastered")
    estimated_time: int = Field(default=0, description="Estimated time to master in minutes")
    prerequisites: List[str] = Field(default_factory=list, description="Prerequisite criterion IDs")


class LearningPath(BaseModel):
    """Complete learning path for a user."""
    id: str = Field(..., description="Learning path ID")
    user_id: int = Field(..., description="User ID")
    blueprint_id: int = Field(..., description="Blueprint ID")
    name: str = Field(..., description="Path name")
    description: str = Field(..., description="Path description")
    
    # Path structure
    nodes: List[LearningPathNode] = Field(default_factory=list, description="Path nodes")
    current_position: int = Field(default=0, description="Current position in path")
    
    # Progress tracking
    completed_nodes: int = Field(default=0, description="Number of completed nodes")
    total_nodes: int = Field(default=0, description="Total number of nodes")
    progress_percentage: float = Field(default=0.0, description="Path completion percentage")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    def calculate_progress(self):
        """Calculate path progress."""
        self.total_nodes = len(self.nodes)
        self.completed_nodes = sum(1 for node in self.nodes if node.is_mastered)
        if self.total_nodes > 0:
            self.progress_percentage = (self.completed_nodes / self.total_nodes) * 100
        else:
            self.progress_percentage = 0.0
    
    def get_next_node(self) -> Optional[LearningPathNode]:
        """Get the next unmastered node in the path."""
        for node in self.nodes:
            if not node.is_mastered:
                # Check if prerequisites are met
                if all(self._is_prerequisite_met(prereq_id) for prereq_id in node.prerequisites):
                    return node
        return None
    
    def _is_prerequisite_met(self, prereq_id: str) -> bool:
        """Check if a prerequisite criterion is mastered."""
        for node in self.nodes:
            if node.criterion_id == prereq_id:
                return node.is_mastered
        return False

