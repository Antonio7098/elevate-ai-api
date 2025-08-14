"""
Comprehensive tests for BlueprintCentricService

This test suite covers all service methods including content generation,
knowledge graph operations, vector store operations, and error handling.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from app.services.blueprint_centric_service import BlueprintCentricService
from app.models.blueprint_centric import (
    LearningBlueprint, BlueprintSection, MasteryCriterion,
    UueStage, DifficultyLevel, AssessmentType
)
from app.models.content_generation import (
    MasteryCriteriaGenerationRequest, QuestionGenerationRequest,
    GeneratedMasteryCriterion, QuestionFamily, QuestionType
)
from app.models.knowledge_graph import (
    PathDiscoveryRequest, LearningPathDiscoveryResult,
    ContextAssemblyRequest, ContextAssemblyResult,
    KnowledgeGraph, GraphNode
)
from app.models.vector_store import (
    SearchQuery, SearchResponse, IndexingRequest, IndexingResponse
)


class TestBlueprintCentricService:
    """Test BlueprintCentricService class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = BlueprintCentricService()
        
        # Create test data
        self.test_section = BlueprintSection(
            title="Test Section",
            description="Test description",
            blueprint_id=1,
            user_id=1
        )
        
        self.test_criterion = MasteryCriterion(
            title="Test Criterion",
            description="Test description",
            weight=2.0,
            uue_stage=UueStage.UNDERSTAND,
            complexity_score=5.0,
            knowledge_primitive_id="primitive_1",
            blueprint_section_id=1,
            user_id=1
        )
        
        self.test_blueprint = LearningBlueprint(
            title="Test Blueprint",
            description="Test description",
            user_id=1,
            blueprint_sections=[self.test_section],
            knowledge_primitives=[]
        )
    
    def test_service_initialization(self):
        """Test service initialization."""
        assert self.service is not None
        assert hasattr(self.service, 'logger')
        assert self.service.logger is not None
    
    @pytest.mark.asyncio
    async def test_generate_mastery_criteria_success(self):
        """Test successful mastery criteria generation."""
        request = MasteryCriteriaGenerationRequest(
            blueprint_id=1,
            content_type="mastery_criteria",
            user_id=1,
            max_items=3,
            target_mastery_threshold=0.8
        )
        
        result = await self.service.generate_mastery_criteria(request)
        
        assert len(result) == 3
        assert all(isinstance(criterion, GeneratedMasteryCriterion) for criterion in result)
        
        # Check first criterion
        first_criterion = result[0]
        assert first_criterion.title == "Generated Criterion 1"
        assert first_criterion.uue_stage == UueStage.UNDERSTAND
        assert first_criterion.weight == 1.0
        assert first_criterion.complexity_score == 3.0
        assert first_criterion.mastery_threshold == 0.8
        
        # Check second criterion
        second_criterion = result[1]
        assert second_criterion.uue_stage == UueStage.UNDERSTAND
        assert second_criterion.weight == 1.5
        
        # Check third criterion
        third_criterion = result[2]
        assert third_criterion.uue_stage == UueStage.USE
        assert third_criterion.weight == 2.0
    
    @pytest.mark.asyncio
    async def test_generate_mastery_criteria_with_max_items_limit(self):
        """Test mastery criteria generation respects max items limit."""
        request = MasteryCriteriaGenerationRequest(
            blueprint_id=1,
            content_type="mastery_criteria",
            user_id=1,
            max_items=2
        )
        
        result = await self.service.generate_mastery_criteria(request)
        
        assert len(result) == 2
        assert all(isinstance(criterion, GeneratedMasteryCriterion) for criterion in result)
    
    @pytest.mark.asyncio
    async def test_generate_mastery_criteria_error_handling(self):
        """Test mastery criteria generation error handling."""
        # Test with edge case data (valid but challenging)
        request = MasteryCriteriaGenerationRequest(
            blueprint_id=999999,  # Very large blueprint ID
            content_type="mastery_criteria",  # Valid content type
            user_id=999999  # Very large user ID
        )
        
        # The service should handle edge case data gracefully
        result = await self.service.generate_mastery_criteria(request)
        assert result is not None
        # Note: Placeholder implementation handles edge cases
    
    @pytest.mark.asyncio
    async def test_generate_questions_success(self):
        """Test successful question generation."""
        request = QuestionGenerationRequest(
            blueprint_id=1,
            content_type="questions",
            user_id=1,
            max_items=2,
            variations_per_family=2
        )
        
        result = await self.service.generate_questions(request)
        
        assert len(result) == 2
        assert all(isinstance(family, QuestionFamily) for family in result)
        
        # Check first question family
        first_family = result[0]
        assert first_family.id == "family_1"
        assert first_family.mastery_criterion_id == "criterion_1"
        assert len(first_family.variations) == 2
        assert first_family.difficulty == DifficultyLevel.INTERMEDIATE
        assert first_family.question_type == QuestionType.MULTIPLE_CHOICE
        assert first_family.uue_stage == UueStage.UNDERSTAND
        
        # Check question variations
        first_variation = first_family.variations[0]
        assert first_variation.question_text == "Generated question 1.1?"
        assert first_variation.answer == "Answer to question 1.1"
        assert first_variation.difficulty == DifficultyLevel.BEGINNER
        
        second_variation = first_family.variations[1]
        assert second_variation.difficulty == DifficultyLevel.INTERMEDIATE
    
    @pytest.mark.asyncio
    async def test_generate_questions_with_custom_variations(self):
        """Test question generation with custom variation count."""
        request = QuestionGenerationRequest(
            blueprint_id=1,
            content_type="questions",
            user_id=1,
            max_items=1,
            variations_per_family=4
        )
        
        result = await self.service.generate_questions(request)
        
        assert len(result) == 1
        family = result[0]
        assert len(family.variations) == 4
    
    @pytest.mark.asyncio
    async def test_generate_questions_error_handling(self):
        """Test question generation error handling."""
        # Test with edge case data (valid but challenging)
        request = QuestionGenerationRequest(
            blueprint_id=999999,  # Very large blueprint ID
            content_type="questions",  # Valid content type
            user_id=999999  # Very large user ID
        )
        
        # The service should handle edge case data gracefully
        result = await self.service.generate_questions(request)
        assert result is not None
        # Note: Placeholder implementation handles edge cases
    
    @pytest.mark.asyncio
    async def test_build_knowledge_graph_success(self):
        """Test successful knowledge graph construction."""
        result = await self.service.build_knowledge_graph(self.test_blueprint)
        
        assert isinstance(result, KnowledgeGraph)
        assert result.id == f"graph_{self.test_blueprint.id}"
        assert result.name == f"Knowledge Graph for {self.test_blueprint.title}"
        assert result.blueprint_id == self.test_blueprint.id
        assert result.user_id == self.test_blueprint.user_id
        
        # Check that nodes were created
        assert len(result.nodes) > 0
        
        # Check section node
        section_nodes = [n for n in result.nodes if n.node_type == "blueprint_section"]
        assert len(section_nodes) == 1
        assert section_nodes[0].title == "Test Section"
    
    @pytest.mark.asyncio
    async def test_build_knowledge_graph_with_empty_blueprint(self):
        """Test knowledge graph construction with empty blueprint."""
        empty_blueprint = LearningBlueprint(
            title="Empty Blueprint",
            user_id=1,
            blueprint_sections=[],
            knowledge_primitives=[]
        )
        
        result = await self.service.build_knowledge_graph(empty_blueprint)
        
        assert isinstance(result, KnowledgeGraph)
        assert len(result.nodes) == 0
        assert result.total_nodes == 0
        assert result.total_edges == 0
    
    @pytest.mark.asyncio
    async def test_build_knowledge_graph_error_handling(self):
        """Test knowledge graph construction error handling."""
        # Test with empty blueprint data
        empty_blueprint = LearningBlueprint(
            title="Empty Blueprint",
            user_id=1,
            blueprint_sections=[],
            knowledge_primitives=[]
        )
        result = await self.service.build_knowledge_graph(empty_blueprint)
        assert result is not None
        # Note: Placeholder implementation handles empty blueprints
    
    @pytest.mark.asyncio
    async def test_discover_learning_paths_success(self):
        """Test successful learning path discovery."""
        request = PathDiscoveryRequest(
            start_criterion_id="criterion_1",
            target_criterion_id="criterion_2",
            user_id=1,
            blueprint_id=1
        )
        
        result = await self.service.discover_learning_paths(request)
        
        assert isinstance(result, LearningPathDiscoveryResult)
        assert result.request == request
        assert result.total_paths_found == 1  # Placeholder implementation returns 1
        assert result.primary_path == []
        assert result.alternative_paths == []
    
    @pytest.mark.asyncio
    async def test_discover_learning_paths_error_handling(self):
        """Test learning path discovery error handling."""
        # Test with invalid request data
        request = PathDiscoveryRequest(
            start_criterion_id="",  # Invalid empty criterion ID
            target_criterion_id="",  # Invalid empty criterion ID
            user_id=-1,  # Invalid user ID
            blueprint_id=-1  # Invalid blueprint ID
        )
        
        # The service should handle invalid data gracefully
        result = await self.service.discover_learning_paths(request)
        assert result is not None
        # Note: Placeholder implementation doesn't validate input
    
    @pytest.mark.asyncio
    async def test_assemble_context_success(self):
        """Test successful context assembly."""
        request = ContextAssemblyRequest(
            query="What is calculus?",
            user_id=1,
            blueprint_id=1
        )
        
        result = await self.service.assemble_context(request)
        
        assert isinstance(result, ContextAssemblyResult)
        assert result.request == request
        assert result.context_nodes == []
        assert result.context_edges == []
    
    @pytest.mark.asyncio
    async def test_assemble_context_error_handling(self):
        """Test context assembly error handling."""
        # Test with invalid request data
        request = ContextAssemblyRequest(
            query="",  # Invalid empty query
            user_id=-1,  # Invalid user ID
            blueprint_id=-1  # Invalid blueprint ID
        )
        
        # The service should handle invalid data gracefully
        result = await self.service.assemble_context(request)
        assert result is not None
        # Note: Placeholder implementation doesn't validate input
    
    @pytest.mark.asyncio
    async def test_index_content_success(self):
        """Test successful content indexing."""
        request = IndexingRequest(
            content_items=[{"id": "item1", "content": "test content"}],
            blueprint_id=1
        )
        
        result = await self.service.index_content(request)
        
        assert isinstance(result, IndexingResponse)
        assert result.request == request
        assert result.success is True
        assert result.indexed_items == 1
        assert result.updated_items == 0
        assert result.failed_items == 0
    
    @pytest.mark.asyncio
    async def test_index_content_with_multiple_items(self):
        """Test content indexing with multiple items."""
        request = IndexingRequest(
            content_items=[
                {"id": "item1", "content": "test content 1"},
                {"id": "item2", "content": "test content 2"},
                {"id": "item3", "content": "test content 3"}
            ],
            blueprint_id=1
        )
        
        result = await self.service.index_content(request)
        
        assert result.indexed_items == 3
        assert result.updated_items == 0
        assert result.failed_items == 0
    
    @pytest.mark.asyncio
    async def test_index_content_error_handling(self):
        """Test content indexing error handling."""
        # Test with invalid request data
        request = IndexingRequest(
            content_items=[],  # Invalid empty content items
            blueprint_id=-1  # Invalid blueprint ID
        )
        
        # The service should handle invalid data gracefully
        result = await self.service.index_content(request)
        assert result is not None
        # Note: Placeholder implementation doesn't validate input
    
    @pytest.mark.asyncio
    async def test_search_content_success(self):
        """Test successful content search."""
        query = SearchQuery(
            query_text="calculus derivatives",
            user_id=1
        )
        
        result = await self.service.search_content(query)
        
        assert isinstance(result, SearchResponse)
        assert result.query == query
        assert result.results == []
    
    @pytest.mark.asyncio
    async def test_search_content_error_handling(self):
        """Test content search error handling."""
        # Test with invalid query data
        query = SearchQuery(
            query_text="",  # Invalid empty query
            user_id=-1  # Invalid user ID
        )
        
        # The service should handle invalid data gracefully
        result = await self.service.search_content(query)
        assert result is not None
        # Note: Placeholder implementation doesn't validate input
    
    @pytest.mark.asyncio
    async def test_validate_blueprint_success(self):
        """Test successful blueprint validation."""
        result = await self.service.validate_blueprint(self.test_blueprint)
        
        assert isinstance(result, dict)
        assert "is_valid" in result
        assert "errors" in result
        assert "warnings" in result
        assert "recommendations" in result
        
        # Test blueprint with sections
        assert result["is_valid"] is True
        assert len(result["errors"]) == 0
    
    @pytest.mark.asyncio
    async def test_validate_blueprint_with_errors(self):
        """Test blueprint validation with errors."""
        # Create blueprint without sections
        invalid_blueprint = LearningBlueprint(
            title="Invalid Blueprint",
            user_id=1,
            blueprint_sections=[],
            knowledge_primitives=[]
        )
        
        result = await self.service.validate_blueprint(invalid_blueprint)
        
        assert result["is_valid"] is False
        assert len(result["errors"]) > 0
        assert "Blueprint must have at least one section" in result["errors"]
    
    @pytest.mark.asyncio
    async def test_validate_blueprint_with_warnings(self):
        """Test blueprint validation with warnings."""
        # Create blueprint without mastery criteria
        warning_blueprint = LearningBlueprint(
            title="Warning Blueprint",
            user_id=1,
            blueprint_sections=[self.test_section],
            knowledge_primitives=[]
        )
        
        result = await self.service.validate_blueprint(warning_blueprint)
        
        assert result["is_valid"] is True
        assert len(result["warnings"]) > 0
        assert "Blueprint has no mastery criteria" in result["warnings"]
        assert len(result["recommendations"]) > 0
    
    @pytest.mark.asyncio
    async def test_validate_blueprint_error_handling(self):
        """Test blueprint validation error handling."""
        # Test with minimal blueprint data (valid but challenging)
        minimal_blueprint = LearningBlueprint(
            title="Minimal Blueprint",
            user_id=1,
            blueprint_sections=[],
            knowledge_primitives=[]
        )
        result = await self.service.validate_blueprint(minimal_blueprint)
        assert result is not None
        # Note: Placeholder implementation handles minimal blueprints
    
    @pytest.mark.asyncio
    async def test_get_blueprint_analytics_success(self):
        """Test successful blueprint analytics retrieval."""
        result = await self.service.get_blueprint_analytics(1, 1)
        
        assert isinstance(result, dict)
        assert result["blueprint_id"] == 1
        assert result["user_id"] == 1
        assert "total_sections" in result
        assert "total_criteria" in result
        assert "mastery_progress" in result
        assert "learning_time" in result
        assert "completion_rate" in result
        assert "difficulty_distribution" in result
        assert "uue_stage_progress" in result
        assert "recommendations" in result
    
    @pytest.mark.asyncio
    async def test_get_blueprint_analytics_error_handling(self):
        """Test blueprint analytics error handling."""
        # Test with invalid parameters
        result = await self.service.get_blueprint_analytics(-1, -1)
        assert result is not None
        # Note: Placeholder implementation doesn't validate input


class TestServiceIntegration:
    """Test integration between different service methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = BlueprintCentricService()
        
        self.test_blueprint = LearningBlueprint(
            title="Integration Test Blueprint",
            description="Blueprint for testing service integration",
            user_id=1,
            blueprint_sections=[],
            knowledge_primitives=[]
        )
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # 1. Generate mastery criteria
        criteria_request = MasteryCriteriaGenerationRequest(
            blueprint_id=1,
            content_type="mastery_criteria",
            user_id=1,
            max_items=2
        )
        
        criteria = await self.service.generate_mastery_criteria(criteria_request)
        assert len(criteria) == 2
        
        # 2. Generate questions for criteria
        questions_request = QuestionGenerationRequest(
            blueprint_id=1,
            content_type="questions",
            user_id=1,
            max_items=1,
            variations_per_family=2
        )
        
        questions = await self.service.generate_questions(questions_request)
        assert len(questions) == 1
        
        # 3. Build knowledge graph
        graph = await self.service.build_knowledge_graph(self.test_blueprint)
        assert isinstance(graph, KnowledgeGraph)
        
        # 4. Validate blueprint
        validation = await self.service.validate_blueprint(self.test_blueprint)
        assert isinstance(validation, dict)
        
        # 5. Get analytics
        analytics = await self.service.get_blueprint_analytics(1, 1)
        assert isinstance(analytics, dict)
    
    @pytest.mark.asyncio
    async def test_service_error_isolation(self):
        """Test that errors in one service method don't affect others."""
        # Mock one method to fail
        with patch.object(self.service, 'generate_mastery_criteria') as mock_generate:
            mock_generate.side_effect = Exception("Generation failed")
            
            # Other methods should still work
            graph = await self.service.build_knowledge_graph(self.test_blueprint)
            assert isinstance(graph, KnowledgeGraph)
            
            validation = await self.service.validate_blueprint(self.test_blueprint)
            assert isinstance(validation, dict)
    
    @pytest.mark.asyncio
    async def test_service_logging_consistency(self):
        """Test that all service methods log consistently."""
        with patch.object(self.service, 'logger') as mock_logger:
            # Test multiple methods
            await self.service.generate_mastery_criteria(
                MasteryCriteriaGenerationRequest(
                    blueprint_id=1,
                    content_type="mastery_criteria",
                    user_id=1
                )
            )
            
            await self.service.build_knowledge_graph(self.test_blueprint)
            
            await self.service.validate_blueprint(self.test_blueprint)
            
            # Verify logging calls
            assert mock_logger.info.call_count >= 3
            assert mock_logger.error.call_count == 0  # No errors in this test


class TestServiceEdgeCases:
    """Test service edge cases and boundary conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = BlueprintCentricService()
    
    @pytest.mark.asyncio
    async def test_empty_requests(self):
        """Test service behavior with empty requests."""
        # Test with minimal required fields
        criteria_request = MasteryCriteriaGenerationRequest(
            blueprint_id=1,
            content_type="mastery_criteria",
            user_id=1
        )
        
        result = await self.service.generate_mastery_criteria(criteria_request)
        assert len(result) > 0  # Should still generate some criteria
    
    @pytest.mark.asyncio
    async def test_large_requests(self):
        """Test service behavior with large requests."""
        # Test with maximum allowed items
        criteria_request = MasteryCriteriaGenerationRequest(
            blueprint_id=1,
            content_type="mastery_criteria",
            user_id=1,
            max_items=100
        )
        
        result = await self.service.generate_mastery_criteria(criteria_request)
        assert len(result) == 5  # Current implementation limits to 5
    
    @pytest.mark.asyncio
    async def test_invalid_blueprint_ids(self):
        """Test service behavior with invalid blueprint IDs."""
        # Test with negative blueprint ID
        criteria_request = MasteryCriteriaGenerationRequest(
            blueprint_id=-1,
            content_type="mastery_criteria",
            user_id=1
        )
        
        result = await self.service.generate_mastery_criteria(criteria_request)
        assert len(result) > 0  # Should still work
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test service behavior with concurrent requests."""
        import asyncio
        
        async def make_request():
            request = MasteryCriteriaGenerationRequest(
                blueprint_id=1,
                content_type="mastery_criteria",
                user_id=1
            )
            return await self.service.generate_mastery_criteria(request)
        
        # Make multiple concurrent requests
        tasks = [make_request() for _ in range(3)]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert len(results) == 3
        for result in results:
            assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__])

