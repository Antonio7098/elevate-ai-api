"""
Comprehensive tests for ContentAggregator

This test suite covers all service methods including content aggregation,
mastery progress calculation, and UUE stage progression tracking.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from app.services.content_aggregator import ContentAggregator
from app.models.blueprint_centric import UueStage


class TestContentAggregator:
    """Test ContentAggregator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = ContentAggregator()
        self.test_section_id = "test_section_123"
        self.test_user_id = 1
    
    def test_service_initialization(self):
        """Test service initialization."""
        assert self.service is not None
        assert hasattr(self.service, 'logger')
        assert self.service.logger is not None
    
    @pytest.mark.asyncio
    async def test_aggregate_section_content_success(self):
        """Test successful content aggregation."""
        result = await self.service.aggregate_section_content(self.test_section_id)
        
        assert isinstance(result, dict)
        assert result["section_id"] == self.test_section_id
        assert "aggregation_timestamp" in result
        assert "content_summary" in result
        assert "content_by_type" in result
        assert "mastery_overview" in result
        assert "uue_stage_distribution" in result
        assert "difficulty_distribution" in result
        assert "children_content" in result
        
        # Check content summary structure
        content_summary = result["content_summary"]
        assert "total_primitives" in content_summary
        assert "total_criteria" in content_summary
        assert "total_questions" in content_summary
        assert "total_sections" in content_summary
        assert "max_depth" in content_summary
        
        # Check content by type structure
        content_by_type = result["content_by_type"]
        assert "entities" in content_by_type
        assert "propositions" in content_by_type
        assert "processes" in content_by_type
        assert "relationships" in content_by_type
        
        # Check mastery overview structure
        mastery_overview = result["mastery_overview"]
        assert "total_mastery_criteria" in mastery_overview
        assert "mastered_criteria" in mastery_overview
        assert "in_progress_criteria" in mastery_overview
        assert "not_started_criteria" in mastery_overview
    
    @pytest.mark.asyncio
    async def test_calculate_mastery_progress_success(self):
        """Test successful mastery progress calculation."""
        result = await self.service.calculate_mastery_progress(self.test_section_id)
        
        assert isinstance(result, dict)
        assert result["section_id"] == self.test_section_id
        assert "calculation_timestamp" in result
        assert "overall_progress" in result
        assert "progress_by_uue_stage" in result
        assert "progress_by_difficulty" in result
        assert "recent_activity" in result
        assert "recommendations" in result
        
        # Check progress by UUE stage structure
        uue_progress = result["progress_by_uue_stage"]
        assert "understand" in uue_progress
        assert "use" in uue_progress
        assert "explore" in uue_progress
        
        for stage_data in uue_progress.values():
            assert "total" in stage_data
            assert "mastered" in stage_data
            assert "progress" in stage_data
        
        # Check progress by difficulty structure
        difficulty_progress = result["progress_by_difficulty"]
        assert "beginner" in difficulty_progress
        assert "intermediate" in difficulty_progress
        assert "advanced" in difficulty_progress
        
        for difficulty_data in difficulty_progress.values():
            assert "total" in difficulty_data
            assert "mastered" in difficulty_data
            assert "progress" in difficulty_data
        
        # Check recommendations
        assert isinstance(result["recommendations"], list)
    
    @pytest.mark.asyncio
    async def test_calculate_uue_stage_progress_success(self):
        """Test successful UUE stage progress calculation."""
        result = await self.service.calculate_uue_stage_progress(self.test_section_id, self.test_user_id)
        
        assert isinstance(result, dict)
        assert result["section_id"] == self.test_section_id
        assert result["user_id"] == self.test_user_id
        assert "calculation_timestamp" in result
        assert "current_stage" in result
        assert "stage_progression" in result
        assert "learning_path_recommendations" in result
        assert "next_milestones" in result
        assert "estimated_completion" in result
        
        # Check current stage
        assert result["current_stage"] in [UueStage.UNDERSTAND, UueStage.USE, UueStage.EXPLORE]
        
        # Check stage progression structure
        stage_progression = result["stage_progression"]
        assert "understand" in stage_progression
        assert "use" in stage_progression
        assert "explore" in stage_progression
        
        for stage_data in stage_progression.values():
            assert "status" in stage_data
            assert "completion_date" in stage_data
            assert "mastery_score" in stage_data
            assert "criteria_count" in stage_data
            assert "mastered_criteria" in stage_data
        
        # Check learning path recommendations
        assert isinstance(result["learning_path_recommendations"], list)
        
        # Check next milestones
        assert isinstance(result["next_milestones"], list)
    
    @pytest.mark.asyncio
    async def test_get_content_analytics_success(self):
        """Test successful content analytics generation."""
        result = await self.service.get_content_analytics(self.test_section_id, self.test_user_id)
        
        assert isinstance(result, dict)
        assert result["section_id"] == self.test_section_id
        assert result["user_id"] == self.test_user_id
        assert "analytics_timestamp" in result
        assert "content_overview" in result
        assert "mastery_progress" in result
        assert "uue_stage_progress" in result
        assert "difficulty_distribution" in result
        assert "learning_efficiency" in result
        assert "content_quality_metrics" in result
        assert "recommendations" in result
        
        # Check learning efficiency structure
        learning_efficiency = result["learning_efficiency"]
        assert "average_time_per_criterion" in learning_efficiency
        assert "mastery_retention_rate" in learning_efficiency
        assert "learning_curve" in learning_efficiency
        
        # Check content quality metrics structure
        content_quality = result["content_quality_metrics"]
        assert "completeness_score" in content_quality
        assert "difficulty_balance" in content_quality
        assert "uue_coverage" in content_quality
        
        # Check that quality metrics are floats between 0 and 1
        for metric in content_quality.values():
            assert isinstance(metric, float)
            assert 0.0 <= metric <= 1.0
    
    @pytest.mark.asyncio
    async def test_mastery_recommendations_generation(self):
        """Test mastery recommendations generation."""
        # Test with low progress
        low_progress = {
            "overall_progress": 0.2
        }
        recommendations = self.service._generate_mastery_recommendations(low_progress)
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any("foundational" in rec.lower() for rec in recommendations)
        
        # Test with medium progress
        medium_progress = {
            "overall_progress": 0.6
        }
        recommendations = self.service._generate_mastery_recommendations(medium_progress)
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any("practicing" in rec.lower() for rec in recommendations)
        
        # Test with high progress
        high_progress = {
            "overall_progress": 0.95
        }
        recommendations = self.service._generate_mastery_recommendations(high_progress)
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any("excellent" in rec.lower() for rec in recommendations)
    
    @pytest.mark.asyncio
    async def test_uue_recommendations_generation(self):
        """Test UUE stage recommendations generation."""
        # Test understand stage
        understand_progress = {
            "current_stage": UueStage.UNDERSTAND
        }
        recommendations = self.service._generate_uue_recommendations(understand_progress)
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any("understanding" in rec.lower() for rec in recommendations)
        
        # Test use stage
        use_progress = {
            "current_stage": UueStage.USE
        }
        recommendations = self.service._generate_uue_recommendations(use_progress)
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any("practice" in rec.lower() for rec in recommendations)
        
        # Test explore stage
        explore_progress = {
            "current_stage": UueStage.EXPLORE
        }
        recommendations = self.service._generate_uue_recommendations(explore_progress)
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any("explore" in rec.lower() for rec in recommendations)
    
    @pytest.mark.asyncio
    async def test_next_milestones_calculation(self):
        """Test next milestones calculation."""
        uue_progress = {
            "current_stage": UueStage.USE,
            "stage_progression": {
                "understand": {"status": "completed", "criteria_count": 5, "mastered_criteria": 5},
                "use": {"status": "in_progress", "criteria_count": 8, "mastered_criteria": 5},
                "explore": {"status": "not_started", "criteria_count": 3, "mastered_criteria": 0}
            }
        }
        
        milestones = self.service._calculate_next_milestones(uue_progress)
        assert isinstance(milestones, list)
        assert len(milestones) > 0
        
        # Check milestone structure
        for milestone in milestones:
            assert "stage" in milestone
            assert "type" in milestone
            assert "description" in milestone
            assert "estimated_effort" in milestone
            assert "priority" in milestone
    
    @pytest.mark.asyncio
    async def test_completion_time_estimation(self):
        """Test completion time estimation."""
        # Test with remaining criteria
        uue_progress = {
            "current_stage": UueStage.USE,
            "stage_progression": {
                "understand": {"status": "completed", "criteria_count": 5, "mastered_criteria": 5},
                "use": {"status": "in_progress", "criteria_count": 8, "mastered_criteria": 5},
                "explore": {"status": "not_started", "criteria_count": 3, "mastered_criteria": 0}
            }
        }
        
        completion_time = self.service._estimate_completion_time(uue_progress)
        assert completion_time is not None
        assert isinstance(completion_time, str)
        
        # Test with no remaining criteria
        completed_progress = {
            "stage_progression": {
                "understand": {"status": "completed", "criteria_count": 5, "mastered_criteria": 5},
                "use": {"status": "completed", "criteria_count": 8, "mastered_criteria": 8},
                "explore": {"status": "completed", "criteria_count": 3, "mastered_criteria": 3}
            }
        }
        
        completion_time = self.service._estimate_completion_time(completed_progress)
        assert completion_time is None
    
    @pytest.mark.asyncio
    async def test_content_quality_metrics_calculation(self):
        """Test content quality metrics calculation."""
        content = {
            "content_summary": {
                "total_primitives": 15,
                "total_criteria": 10
            }
        }
        
        progress = {
            "difficulty_distribution": {
                "beginner": 5,
                "intermediate": 3,
                "advanced": 2
            },
            "uue_stage_distribution": {
                "understand": 4,
                "use": 3,
                "explore": 3
            }
        }
        
        metrics = self.service._calculate_content_quality_metrics(content, progress)
        assert isinstance(metrics, dict)
        assert "completeness_score" in metrics
        assert "difficulty_balance" in metrics
        assert "uue_coverage" in metrics
        
        # Check that all metrics are floats between 0 and 1
        for metric in metrics.values():
            assert isinstance(metric, float)
            assert 0.0 <= metric <= 1.0
    
    @pytest.mark.asyncio
    async def test_service_error_handling(self):
        """Test service error handling."""
        # Test with invalid section ID - service should handle empty strings gracefully
        result = await self.service.aggregate_section_content("")
        assert isinstance(result, dict)
        assert result["section_id"] == ""
        
        # Test with invalid user ID - service should handle negative user IDs gracefully
        result = await self.service.calculate_uue_stage_progress("", -1)
        assert isinstance(result, dict)
        assert result["user_id"] == -1
    
    @pytest.mark.asyncio
    async def test_service_integration(self):
        """Test service integration and data flow."""
        # Test complete workflow
        content = await self.service.aggregate_section_content(self.test_section_id)
        progress = await self.service.calculate_mastery_progress(self.test_section_id)
        uue_progress = await self.service.calculate_uue_stage_progress(self.test_section_id, self.test_user_id)
        analytics = await self.service.get_content_analytics(self.test_section_id, self.test_user_id)
        
        # Verify data consistency
        assert content["section_id"] == progress["section_id"]
        assert progress["section_id"] == uue_progress["section_id"]
        assert uue_progress["section_id"] == analytics["section_id"]
        
        # Verify analytics includes data from other methods
        assert analytics["content_overview"] == content["content_summary"]
        assert analytics["mastery_progress"] == progress["overall_progress"]
        assert analytics["uue_stage_progress"] == progress["progress_by_uue_stage"]
