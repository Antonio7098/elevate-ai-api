"""
Comprehensive tests for Mastery Tracking Models

This test suite covers all the mastery tracking models including user preferences,
mastery thresholds, progress tracking, and learning paths.
"""

import pytest
from datetime import datetime, timedelta
from typing import List

from app.models.mastery_tracking import (
    MasteryThreshold, LearningStyle, ExperienceLevel,
    UserMasteryPreferences, SectionMasteryThreshold, CriterionMasteryThreshold,
    UserCriterionMastery, MasteryCalculationRequest, MasteryCalculationResult,
    MasteryPerformanceMetrics, LearningPathNode, LearningPath
)

from app.models.blueprint_centric import UueStage, DifficultyLevel


class TestEnums:
    """Test enum values and validation."""
    
    def test_mastery_threshold_enum(self):
        """Test mastery threshold enum values."""
        assert MasteryThreshold.SURVEY == "SURVEY"
        assert MasteryThreshold.PROFICIENT == "PROFICIENT"
        assert MasteryThreshold.EXPERT == "EXPERT"
        assert len(MasteryThreshold) == 3
    
    def test_learning_style_enum(self):
        """Test learning style enum values."""
        assert LearningStyle.CONSERVATIVE == "CONSERVATIVE"
        assert LearningStyle.BALANCED == "BALANCED"
        assert LearningStyle.AGGRESSIVE == "AGGRESSIVE"
        assert len(LearningStyle) == 3
    
    def test_experience_level_enum(self):
        """Test experience level enum values."""
        assert ExperienceLevel.BEGINNER == "BEGINNER"
        assert ExperienceLevel.INTERMEDIATE == "INTERMEDIATE"
        assert ExperienceLevel.ADVANCED == "ADVANCED"
        assert ExperienceLevel.EXPERT == "EXPERT"
        assert len(ExperienceLevel) == 4


class TestUserMasteryPreferences:
    """Test UserMasteryPreferences model."""
    
    def test_valid_user_mastery_preferences(self):
        """Test creating valid user mastery preferences."""
        preferences = UserMasteryPreferences(
            user_id=1,
            default_mastery_threshold=MasteryThreshold.PROFICIENT,
            default_tracking_intensity="NORMAL",
            learning_style=LearningStyle.BALANCED,
            experience_level=ExperienceLevel.INTERMEDIATE,
            auto_adjustment=True,
            daily_study_time=90,
            preferred_uue_stages=[UueStage.UNDERSTAND, UueStage.USE]
        )
        
        assert preferences.user_id == 1
        assert preferences.default_mastery_threshold == MasteryThreshold.PROFICIENT
        assert preferences.learning_style == LearningStyle.BALANCED
        assert preferences.experience_level == ExperienceLevel.INTERMEDIATE
        assert preferences.auto_adjustment is True
        assert preferences.daily_study_time == 90
        assert len(preferences.preferred_uue_stages) == 2
    
    def test_user_mastery_preferences_defaults(self):
        """Test user mastery preferences default values."""
        preferences = UserMasteryPreferences(user_id=1)
        
        assert preferences.default_mastery_threshold == MasteryThreshold.PROFICIENT
        assert preferences.default_tracking_intensity == "NORMAL"
        assert preferences.learning_style == LearningStyle.BALANCED
        assert preferences.experience_level == ExperienceLevel.INTERMEDIATE
        assert preferences.auto_adjustment is False
        assert preferences.daily_study_time == 60
        assert len(preferences.preferred_uue_stages) == 3
        assert preferences.consecutive_interval_requirement == 2
        assert preferences.min_gap_days == 1
        assert preferences.max_review_frequency == 7
    
    def test_user_mastery_preferences_validation(self):
        """Test user mastery preferences validation rules."""
        # Test daily study time validation
        with pytest.raises(ValueError, match="Daily study time must be between 15 and 480 minutes"):
            UserMasteryPreferences(
                user_id=1,
                daily_study_time=10
            )
        
        with pytest.raises(ValueError, match="Daily study time must be between 15 and 480 minutes"):
            UserMasteryPreferences(
                user_id=1,
                daily_study_time=500
            )
        
        # Test consecutive interval requirement validation
        with pytest.raises(ValueError, match="Consecutive interval requirement must be between 1 and 5"):
            UserMasteryPreferences(
                user_id=1,
                consecutive_interval_requirement=0
            )
        
        with pytest.raises(ValueError, match="Consecutive interval requirement must be between 1 and 5"):
            UserMasteryPreferences(
                user_id=1,
                consecutive_interval_requirement=6
            )
        
        # Test min gap days validation
        with pytest.raises(ValueError, match="Minimum gap days must be between 0 and 7"):
            UserMasteryPreferences(
                user_id=1,
                min_gap_days=-1
            )
        
        with pytest.raises(ValueError, match="Minimum gap days must be between 0 and 7"):
            UserMasteryPreferences(
                user_id=1,
                min_gap_days=8
            )


class TestSectionMasteryThreshold:
    """Test SectionMasteryThreshold model."""
    
    def test_valid_section_mastery_threshold(self):
        """Test creating valid section mastery threshold."""
        threshold = SectionMasteryThreshold(
            user_id=1,
            section_id="section_1",
            threshold=MasteryThreshold.EXPERT,
            threshold_value=0.95,
            description="User wants expert-level mastery for this section"
        )
        
        assert threshold.user_id == 1
        assert threshold.section_id == "section_1"
        assert threshold.threshold == MasteryThreshold.EXPERT
        assert threshold.threshold_value == 0.95
        assert threshold.description == "User wants expert-level mastery for this section"
    
    def test_section_mastery_threshold_validation(self):
        """Test section mastery threshold validation rules."""
        # Test invalid threshold value
        with pytest.raises(ValueError, match="Threshold value must be one of"):
            SectionMasteryThreshold(
                user_id=1,
                section_id="section_1",
                threshold=MasteryThreshold.PROFICIENT,
                threshold_value=0.75,
                description="Invalid threshold value"
            )


class TestCriterionMasteryThreshold:
    """Test CriterionMasteryThreshold model."""
    
    def test_valid_criterion_mastery_threshold(self):
        """Test creating valid criterion mastery threshold."""
        threshold = CriterionMasteryThreshold(
            user_id=1,
            criterion_id="criterion_1",
            threshold=MasteryThreshold.SURVEY,
            threshold_value=0.6,
            description="User wants basic understanding for this criterion"
        )
        
        assert threshold.user_id == 1
        assert threshold.criterion_id == "criterion_1"
        assert threshold.threshold == MasteryThreshold.SURVEY
        assert threshold.threshold_value == 0.6
        assert threshold.description == "User wants basic understanding for this criterion"
    
    def test_criterion_mastery_threshold_validation(self):
        """Test criterion mastery threshold validation rules."""
        # Test invalid threshold value
        with pytest.raises(ValueError, match="Threshold value must be one of"):
            CriterionMasteryThreshold(
                user_id=1,
                criterion_id="criterion_1",
                threshold=MasteryThreshold.PROFICIENT,
                threshold_value=0.7,
                description="Invalid threshold value"
            )


class TestUserCriterionMastery:
    """Test UserCriterionMastery model."""
    
    def test_valid_user_criterion_mastery(self):
        """Test creating valid user criterion mastery."""
        mastery = UserCriterionMastery(
            user_id=1,
            mastery_criterion_id="criterion_1",
            blueprint_section_id="section_1",
            is_mastered=False,
            mastery_score=0.7,
            uue_stage=UueStage.UNDERSTAND,
            last_two_attempts=[0.6, 0.8],
            consecutive_intervals=1,
            current_interval_step=2,
            tracking_intensity="NORMAL"
        )
        
        assert mastery.user_id == 1
        assert mastery.mastery_criterion_id == "criterion_1"
        assert mastery.mastery_score == 0.7
        assert mastery.uue_stage == UueStage.UNDERSTAND
        assert mastery.last_two_attempts == [0.6, 0.8]
        assert mastery.consecutive_intervals == 1
        assert mastery.current_interval_step == 2
        assert mastery.tracking_intensity == "NORMAL"
    
    def test_user_criterion_mastery_defaults(self):
        """Test user criterion mastery default values."""
        mastery = UserCriterionMastery(
            user_id=1,
            mastery_criterion_id="criterion_1",
            blueprint_section_id="section_1"
        )
        
        assert mastery.is_mastered is False
        assert mastery.mastery_score == 0.0
        assert mastery.uue_stage == UueStage.UNDERSTAND
        assert mastery.last_two_attempts == []
        assert mastery.consecutive_intervals == 0
        assert mastery.current_interval_step == 0
        assert mastery.review_count == 0
        assert mastery.successful_reviews == 0
        assert mastery.consecutive_failures == 0
        assert mastery.tracking_intensity == "NORMAL"
    
    def test_user_criterion_mastery_validation(self):
        """Test user criterion mastery validation rules."""
        # Test mastery score validation
        with pytest.raises(ValueError, match="Mastery score must be between 0.0 and 1.0"):
            UserCriterionMastery(
                user_id=1,
                mastery_criterion_id="criterion_1",
                blueprint_section_id="section_1",
                mastery_score=-0.1
            )
        
        with pytest.raises(ValueError, match="Mastery score must be between 0.0 and 1.0"):
            UserCriterionMastery(
                user_id=1,
                mastery_criterion_id="criterion_1",
                blueprint_section_id="section_1",
                mastery_score=1.1
            )
        
        # Test last two attempts validation
        with pytest.raises(ValueError, match="Last two attempts cannot have more than 2 scores"):
            UserCriterionMastery(
                user_id=1,
                mastery_criterion_id="criterion_1",
                blueprint_section_id="section_1",
                last_two_attempts=[0.6, 0.8, 0.9]
            )
        
        with pytest.raises(ValueError, match="Attempt scores must be between 0.0 and 1.0"):
            UserCriterionMastery(
                user_id=1,
                mastery_criterion_id="criterion_1",
                blueprint_section_id="section_1",
                last_two_attempts=[0.6, 1.2]
            )
    
    def test_user_criterion_mastery_methods(self):
        """Test user criterion mastery utility methods."""
        mastery = UserCriterionMastery(
            user_id=1,
            mastery_criterion_id="criterion_1",
            blueprint_section_id="section_1"
        )
        
        # Test add_attempt with successful score
        mastery.add_attempt(0.85)
        assert len(mastery.last_two_attempts) == 1
        assert mastery.last_two_attempts[0] == 0.85
        assert mastery.review_count == 1
        assert mastery.successful_reviews == 1
        assert mastery.consecutive_failures == 0
        
        # Test add_attempt with unsuccessful score
        mastery.add_attempt(0.65)
        assert len(mastery.last_two_attempts) == 2
        assert mastery.last_two_attempts == [0.85, 0.65]
        assert mastery.review_count == 2
        assert mastery.successful_reviews == 1
        assert mastery.consecutive_failures == 1
        
        # Test add_attempt with third score (should remove first)
        mastery.add_attempt(0.9)
        assert len(mastery.last_two_attempts) == 2
        assert mastery.last_two_attempts == [0.65, 0.9]
        assert mastery.review_count == 3
        assert mastery.successful_reviews == 2
        assert mastery.consecutive_failures == 0
    
    def test_user_criterion_mastery_check_mastery(self):
        """Test mastery checking logic."""
        mastery = UserCriterionMastery(
            user_id=1,
            mastery_criterion_id="criterion_1",
            blueprint_section_id="section_1"
        )
        
        # Test mastery check with insufficient attempts
        assert not mastery.check_mastery(0.8)
        
        # Test mastery check with successful attempts
        mastery.last_two_attempts = [0.85, 0.9]
        mastery.last_threshold_check_date = datetime.now() - timedelta(days=2)
        
        # Should not be mastered yet (need 2 consecutive intervals)
        assert not mastery.check_mastery(0.8)
        
        # Set consecutive intervals to 2
        mastery.consecutive_intervals = 2
        mastery.last_threshold_check_date = datetime.now() - timedelta(days=2)
        
        # Should now be mastered
        assert mastery.check_mastery(0.8)
        assert mastery.is_mastered is True


class TestMasteryCalculationRequest:
    """Test MasteryCalculationRequest model."""
    
    def test_valid_mastery_calculation_request(self):
        """Test creating valid mastery calculation request."""
        request = MasteryCalculationRequest(
            user_id=1,
            criterion_id="criterion_1",
            section_id="section_1",
            blueprint_id=1,
            include_history=True,
            include_recommendations=False
        )
        
        assert request.user_id == 1
        assert request.criterion_id == "criterion_1"
        assert request.section_id == "section_1"
        assert request.blueprint_id == 1
        assert request.include_history is True
        assert request.include_recommendations is False
    
    def test_mastery_calculation_request_defaults(self):
        """Test mastery calculation request default values."""
        request = MasteryCalculationRequest(user_id=1)
        
        assert request.criterion_id is None
        assert request.section_id is None
        assert request.blueprint_id is None
        assert request.include_history is False
        assert request.include_recommendations is True


class TestMasteryCalculationResult:
    """Test MasteryCalculationResult model."""
    
    def test_valid_mastery_calculation_result(self):
        """Test creating valid mastery calculation result."""
        result = MasteryCalculationResult(
            user_id=1,
            calculation_type="criterion_level",
            criterion_mastery={"criterion_1": 0.8, "criterion_2": 0.9},
            total_criteria=2,
            mastered_criteria=1,
            mastery_percentage=50.0
        )
        
        assert result.user_id == 1
        assert result.calculation_type == "criterion_level"
        assert result.criterion_mastery == {"criterion_1": 0.8, "criterion_2": 0.9}
        assert result.total_criteria == 2
        assert result.mastered_criteria == 1
        assert result.mastery_percentage == 50.0
        assert result.recommendations == []
    
    def test_mastery_calculation_result_methods(self):
        """Test mastery calculation result utility methods."""
        result = MasteryCalculationResult(
            user_id=1,
            calculation_type="criterion_level",
            total_criteria=5,
            mastered_criteria=2,
            mastery_percentage=40.0
        )
        
        # Test add_recommendation
        result.add_recommendation("Focus on understanding concepts")
        assert len(result.recommendations) == 1
        
        result.add_recommendation("Practice more application problems")
        assert len(result.recommendations) == 2


class TestMasteryPerformanceMetrics:
    """Test MasteryPerformanceMetrics model."""
    
    def test_valid_mastery_performance_metrics(self):
        """Test creating valid mastery performance metrics."""
        metrics = MasteryPerformanceMetrics(
            user_id=1,
            time_period="weekly",
            total_reviews=50,
            successful_reviews=40,
            period_start=datetime.now() - timedelta(days=7),
            period_end=datetime.now()
        )
        
        assert metrics.user_id == 1
        assert metrics.time_period == "weekly"
        assert metrics.total_reviews == 50
        assert metrics.successful_reviews == 40
        assert metrics.success_rate == 0.0  # Will be calculated
    
    def test_mastery_performance_metrics_validation(self):
        """Test mastery performance metrics validation rules."""
        # Test success rate validation
        with pytest.raises(ValueError, match="Success rate must be between 0.0 and 1.0"):
            MasteryPerformanceMetrics(
                user_id=1,
                time_period="weekly",
                success_rate=1.5,
                period_start=datetime.now() - timedelta(days=7),
                period_end=datetime.now()
            )
    
    def test_mastery_performance_metrics_methods(self):
        """Test mastery performance metrics utility methods."""
        metrics = MasteryPerformanceMetrics(
            user_id=1,
            time_period="weekly",
            total_reviews=50,
            successful_reviews=40,
            period_start=datetime.now() - timedelta(days=7),
            period_end=datetime.now()
        )
        
        # Test calculate_success_rate
        metrics.calculate_success_rate()
        assert metrics.success_rate == 0.8
        
        # Test with zero reviews
        metrics.total_reviews = 0
        metrics.calculate_success_rate()
        assert metrics.success_rate == 0.0


class TestLearningPathNode:
    """Test LearningPathNode model."""
    
    def test_valid_learning_path_node(self):
        """Test creating valid learning path node."""
        node = LearningPathNode(
            criterion_id="criterion_1",
            uue_stage=UueStage.UNDERSTAND,
            estimated_time=30,
            prerequisites=["prereq_1", "prereq_2"]
        )
        
        assert node.criterion_id == "criterion_1"
        assert node.uue_stage == UueStage.UNDERSTAND
        assert node.estimated_time == 30
        assert len(node.prerequisites) == 2
    
    def test_learning_path_node_defaults(self):
        """Test learning path node default values."""
        node = LearningPathNode(
            criterion_id="criterion_1",
            uue_stage=UueStage.UNDERSTAND
        )
        
        assert node.mastery_score == 0.0
        assert node.is_mastered is False
        assert node.estimated_time == 0
        assert node.prerequisites == []
    
    def test_learning_path_node_validation(self):
        """Test learning path node validation rules."""
        # Note: The model doesn't have validation for estimated_time range
        # so we just test that valid values work
        node = LearningPathNode(
            criterion_id="criterion_1",
            uue_stage=UueStage.UNDERSTAND,
            estimated_time=30
        )
        assert node.estimated_time == 30


class TestLearningPath:
    """Test LearningPath model."""
    
    def test_valid_learning_path(self):
        """Test creating valid learning path."""
        node1 = LearningPathNode(
            criterion_id="criterion_1",
            uue_stage=UueStage.UNDERSTAND
        )
        
        node2 = LearningPathNode(
            criterion_id="criterion_2",
            uue_stage=UueStage.USE
        )
        
        path = LearningPath(
            id="path_1",
            user_id=1,
            blueprint_id=1,
            name="Calculus Fundamentals",
            description="Learn calculus from basics to applications",
            nodes=[node1, node2]
        )
        
        assert path.id == "path_1"
        assert path.user_id == 1
        assert path.blueprint_id == 1
        assert path.name == "Calculus Fundamentals"
        assert len(path.nodes) == 2
        assert path.current_position == 0
        assert path.completed_nodes == 0
        assert path.total_nodes == 0  # Will be calculated
        assert path.progress_percentage == 0.0  # Will be calculated
    
    def test_learning_path_methods(self):
        """Test learning path utility methods."""
        node1 = LearningPathNode(
            criterion_id="criterion_1",
            uue_stage=UueStage.UNDERSTAND
        )
        
        node2 = LearningPathNode(
            criterion_id="criterion_2",
            uue_stage=UueStage.USE
        )
        
        path = LearningPath(
            id="path_1",
            user_id=1,
            blueprint_id=1,
            name="Test Path",
            description="Test description",
            nodes=[node1, node2]
        )
        
        # Test calculate_progress
        path.calculate_progress()
        assert path.total_nodes == 2
        assert path.completed_nodes == 0
        assert path.progress_percentage == 0.0
        
        # Test with mastered nodes
        node1.is_mastered = True
        path.calculate_progress()
        assert path.completed_nodes == 1
        assert path.progress_percentage == 50.0
        
        # Test get_next_node
        next_node = path.get_next_node()
        assert next_node == node2  # node1 is mastered, so next is node2
        
        # Test get_next_node when all are mastered
        node2.is_mastered = True
        next_node = path.get_next_node()
        assert next_node is None
    
    def test_learning_path_prerequisites(self):
        """Test learning path prerequisite checking."""
        node1 = LearningPathNode(
            criterion_id="criterion_1",
            uue_stage=UueStage.UNDERSTAND
        )
        
        node2 = LearningPathNode(
            criterion_id="criterion_2",
            uue_stage=UueStage.USE,
            prerequisites=["criterion_1"]
        )
        
        path = LearningPath(
            id="path_1",
            user_id=1,
            blueprint_id=1,
            name="Test Path",
            description="Test description",
            nodes=[node1, node2]
        )
        
        # Test prerequisite checking
        assert path._is_prerequisite_met("criterion_1") is False  # node1 not mastered
        
        node1.is_mastered = True
        assert path._is_prerequisite_met("criterion_1") is True
        
        # Test get_next_node with prerequisites
        next_node = path.get_next_node()
        assert next_node == node2  # node1 is mastered, prerequisites met
        
        # Test get_next_node with unmet prerequisites
        node1.is_mastered = False
        next_node = path.get_next_node()
        assert next_node == node1  # node1 has no prerequisites


class TestModelIntegration:
    """Test integration between different mastery tracking models."""
    
    def test_user_preferences_with_mastery_tracking(self):
        """Test user preferences integration with mastery tracking."""
        preferences = UserMasteryPreferences(
            user_id=1,
            default_mastery_threshold=MasteryThreshold.EXPERT,
            learning_style=LearningStyle.AGGRESSIVE,
            experience_level=ExperienceLevel.ADVANCED
        )
        
        mastery = UserCriterionMastery(
            user_id=1,
            mastery_criterion_id="criterion_1",
            blueprint_section_id="section_1"
        )
        
        # Test that preferences influence mastery tracking
        assert preferences.default_mastery_threshold == MasteryThreshold.EXPERT
        assert preferences.learning_style == LearningStyle.AGGRESSIVE
        assert preferences.experience_level == ExperienceLevel.ADVANCED
        
        # Test mastery tracking with preferences
        mastery.tracking_intensity = preferences.default_tracking_intensity
        assert mastery.tracking_intensity == "NORMAL"
    
    def test_mastery_calculation_integration(self):
        """Test mastery calculation integration."""
        request = MasteryCalculationRequest(
            user_id=1,
            blueprint_id=1,
            include_recommendations=True
        )
        
        result = MasteryCalculationResult(
            user_id=1,
            calculation_type="blueprint_level",
            total_criteria=10,
            mastered_criteria=6,
            mastery_percentage=60.0
        )
        
        # Test calculation result
        assert result.mastery_percentage == 60.0
        
        # Test adding recommendations
        result.add_recommendation("Focus on remaining criteria")
        assert len(result.recommendations) == 1
    
    def test_learning_path_integration(self):
        """Test learning path integration with mastery tracking."""
        # Create nodes with different UUE stages
        understand_node = LearningPathNode(
            criterion_id="criterion_1",
            uue_stage=UueStage.UNDERSTAND
        )
        
        use_node = LearningPathNode(
            criterion_id="criterion_2",
            uue_stage=UueStage.USE,
            prerequisites=["criterion_1"]
        )
        
        explore_node = LearningPathNode(
            criterion_id="criterion_3",
            uue_stage=UueStage.EXPLORE,
            prerequisites=["criterion_2"]
        )
        
        # Create learning path
        path = LearningPath(
            id="path_1",
            user_id=1,
            blueprint_id=1,
            name="UUE Progression Path",
            description="Progressive learning through UUE stages",
            nodes=[understand_node, use_node, explore_node]
        )
        
        # Test UUE stage progression
        assert path.nodes[0].uue_stage == UueStage.UNDERSTAND
        assert path.nodes[1].uue_stage == UueStage.USE
        assert path.nodes[2].uue_stage == UueStage.EXPLORE
        
        # Test prerequisite chain
        assert path.nodes[1].prerequisites == ["criterion_1"]
        assert path.nodes[2].prerequisites == ["criterion_2"]
        
        # Test path progression
        path.calculate_progress()
        assert path.total_nodes == 3
        assert path.completed_nodes == 0
        
        # Complete first node
        understand_node.is_mastered = True
        path.calculate_progress()
        assert path.completed_nodes == 1
        assert abs(path.progress_percentage - 33.33) < 0.01
        
        # Test next node selection
        next_node = path.get_next_node()
        assert next_node == use_node  # use_node prerequisites are met
        
        # Complete second node
        use_node.is_mastered = True
        path.calculate_progress()
        assert path.completed_nodes == 2
        assert abs(path.progress_percentage - 66.67) < 0.01
        
        # Test final node selection
        next_node = path.get_next_node()
        assert next_node == explore_node  # explore_node prerequisites are met


if __name__ == "__main__":
    pytest.main([__file__])

