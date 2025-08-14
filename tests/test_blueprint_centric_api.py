"""
Comprehensive tests for Blueprint-Centric API Endpoints

This test suite covers all API endpoints including request validation,
response formatting, error handling, and integration with services.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import json

from app.api.v1.blueprint_centric import router
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
    ContextAssemblyRequest, ContextAssemblyResult
)
from app.models.vector_store import (
    SearchQuery, SearchResponse, IndexingRequest, IndexingResponse
)


# Create test client with FastAPI app
from fastapi import FastAPI

app = FastAPI()
app.include_router(router)
client = TestClient(app)


class TestContentGenerationEndpoints:
    """Test content generation API endpoints."""
    
    def test_generate_mastery_criteria_success(self):
        """Test successful mastery criteria generation endpoint."""
        request_data = {
            "blueprint_id": 1,
            "content_type": "mastery_criteria",
            "user_id": 1,
            "max_items": 3,
            "target_mastery_threshold": 0.8,
            "balance_uue_stages": True
        }
        
        with patch('app.api.v1.blueprint_centric.blueprint_service.generate_mastery_criteria') as mock_generate:
            mock_generate.return_value = [
                GeneratedMasteryCriterion(
                    title="Generated Criterion 1",
                    description="Test criterion 1",
                    uue_stage=UueStage.UNDERSTAND,
                    weight=1.0,
                    complexity_score=3.0,
                    assessment_type=AssessmentType.QUESTION_BASED,
                    mastery_threshold=0.8
                ),
                GeneratedMasteryCriterion(
                    title="Generated Criterion 2",
                    description="Test criterion 2",
                    uue_stage=UueStage.USE,
                    weight=2.0,
                    complexity_score=5.0,
                    assessment_type=AssessmentType.QUESTION_BASED,
                    mastery_threshold=0.8
                )
            ]
            
            response = client.post("/v1/blueprint-centric/mastery-criteria/generate", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2
            assert data[0]["title"] == "Generated Criterion 1"
            assert data[0]["uue_stage"] == "UNDERSTAND"
            assert data[1]["title"] == "Generated Criterion 2"
            assert data[1]["uue_stage"] == "USE"
            
            # Verify service was called
            mock_generate.assert_called_once()
            call_args = mock_generate.call_args[0][0]
            assert call_args.blueprint_id == 1
            assert call_args.max_items == 3
            assert call_args.target_mastery_threshold == 0.8
    
    def test_generate_mastery_criteria_validation_error(self):
        """Test mastery criteria generation with validation errors."""
        # Test with invalid content type
        request_data = {
            "blueprint_id": 1,
            "content_type": "invalid_type",
            "user_id": 1
        }
        
        response = client.post("/v1/blueprint-centric/mastery-criteria/generate", json=request_data)
        assert response.status_code == 422  # Validation error
        
        # Test with invalid max_items
        request_data = {
            "blueprint_id": 1,
            "content_type": "mastery_criteria",
            "user_id": 1,
            "max_items": 0  # Invalid
        }
        
        response = client.post("/v1/blueprint-centric/mastery-criteria/generate", json=request_data)
        assert response.status_code == 422
    
    def test_generate_mastery_criteria_service_error(self):
        """Test mastery criteria generation when service fails."""
        request_data = {
            "blueprint_id": 1,
            "content_type": "mastery_criteria",
            "user_id": 1
        }
        
        with patch('app.api.v1.blueprint_centric.blueprint_service.generate_mastery_criteria') as mock_generate:
            mock_generate.side_effect = Exception("Service error")
            
            response = client.post("/v1/blueprint-centric/mastery-criteria/generate", json=request_data)
            
            assert response.status_code == 500
            data = response.json()
            assert "Service error" in data["detail"]
    
    def test_generate_questions_success(self):
        """Test successful question generation endpoint."""
        request_data = {
            "blueprint_id": 1,
            "content_type": "questions",
            "user_id": 1,
            "max_items": 2,
            "variations_per_family": 3,
            "question_types": ["multiple_choice", "fill_blank"],
            "include_explanations": True,
            "generate_question_families": True
        }
        
        with patch('app.api.v1.blueprint_centric.blueprint_service.generate_questions') as mock_generate:
            mock_generate.return_value = [
                QuestionFamily(
                    id="family_1",
                    mastery_criterion_id="criterion_1",
                    base_question="What is calculus?",
                    variations=[],
                    difficulty=DifficultyLevel.BEGINNER,
                    question_type=QuestionType.MULTIPLE_CHOICE,
                    uue_stage=UueStage.UNDERSTAND
                )
            ]
            
            response = client.post("/v1/blueprint-centric/questions/generate", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["id"] == "family_1"
            assert data[0]["base_question"] == "What is calculus?"
            
            # Verify service was called
            mock_generate.assert_called_once()
            call_args = mock_generate.call_args[0][0]
            assert call_args.blueprint_id == 1
            assert call_args.max_items == 2
            assert call_args.variations_per_family == 3
    
    def test_generate_questions_validation_error(self):
        """Test question generation with validation errors."""
        # Test with invalid variations_per_family
        request_data = {
            "blueprint_id": 1,
            "content_type": "questions",
            "user_id": 1,
            "variations_per_family": 0  # Invalid
        }
        
        response = client.post("/v1/blueprint-centric/questions/generate", json=request_data)
        assert response.status_code == 422


class TestKnowledgeGraphEndpoints:
    """Test knowledge graph API endpoints."""
    
    def test_build_knowledge_graph_success(self):
        """Test successful knowledge graph building endpoint."""
        with patch('app.api.v1.blueprint_centric.blueprint_service.build_knowledge_graph') as mock_build:
            mock_build.return_value = Mock(
                id="graph_1",
                total_nodes=5,
                total_edges=8
            )
            
            response = client.post("/v1/blueprint-centric/knowledge-graph/build/1")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["graph_id"] == "graph_1"
            assert data["total_nodes"] == 5
            assert data["total_edges"] == 8
            
            # Verify service was called
            mock_build.assert_called_once()
    
    def test_build_knowledge_graph_service_error(self):
        """Test knowledge graph building when service fails."""
        with patch('app.api.v1.blueprint_centric.blueprint_service.build_knowledge_graph') as mock_build:
            mock_build.side_effect = Exception("Service error")
            
            response = client.post("/v1/blueprint-centric/knowledge-graph/build/1")
            
            assert response.status_code == 500
            data = response.json()
            assert "Service error" in data["detail"]
    
    def test_discover_learning_paths_success(self):
        """Test successful learning path discovery endpoint."""
        request_data = {
            "start_criterion_id": "criterion_1",
            "target_criterion_id": "criterion_2",
            "user_id": 1,
            "blueprint_id": 1,
            "max_path_length": 5,
            "preferred_uue_stages": ["UNDERSTAND", "USE"],
            "include_prerequisites": True
        }
        
        with patch('app.api.v1.blueprint_centric.blueprint_service.discover_learning_paths') as mock_discover:
            mock_discover.return_value = LearningPathDiscoveryResult(
                request=PathDiscoveryRequest(**request_data),
                primary_path=[],
                alternative_paths=[]
            )
            
            response = client.post("/v1/blueprint-centric/learning-paths/discover", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert "request" in data
            assert "primary_path" in data
            assert "alternative_paths" in data
            
            # Verify service was called
            mock_discover.assert_called_once()
    
    def test_discover_learning_paths_validation_error(self):
        """Test learning path discovery with validation errors."""
        # Test with invalid max_path_length
        request_data = {
            "start_criterion_id": "criterion_1",
            "target_criterion_id": "criterion_2",
            "user_id": 1,
            "blueprint_id": 1,
            "max_path_length": 1  # Invalid (minimum is 2)
        }
        
        response = client.post("/v1/blueprint-centric/learning-paths/discover", json=request_data)
        assert response.status_code == 422
    
    def test_assemble_context_success(self):
        """Test successful context assembly endpoint."""
        request_data = {
            "query": "What is calculus?",
            "user_id": 1,
            "blueprint_id": 1,
            "max_context_nodes": 20,
            "include_relationships": True,
            "context_depth": 2
        }
        
        with patch('app.api.v1.blueprint_centric.blueprint_service.assemble_context') as mock_assemble:
            mock_assemble.return_value = ContextAssemblyResult(
                request=ContextAssemblyRequest(**request_data),
                context_nodes=[],
                context_edges=[]
            )
            
            response = client.post("/v1/blueprint-centric/context/assemble", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert "request" in data
            assert "context_nodes" in data
            assert "context_edges" in data
            
            # Verify service was called
            mock_assemble.assert_called_once()
    
    def test_assemble_context_validation_error(self):
        """Test context assembly with validation errors."""
        # Test with invalid max_context_nodes
        request_data = {
            "query": "What is calculus?",
            "user_id": 1,
            "blueprint_id": 1,
            "max_context_nodes": 3  # Invalid (minimum is 5)
        }
        
        response = client.post("/v1/blueprint-centric/context/assemble", json=request_data)
        assert response.status_code == 422


class TestVectorStoreEndpoints:
    """Test vector store API endpoints."""
    
    def test_index_content_success(self):
        """Test successful content indexing endpoint."""
        request_data = {
            "content_items": [
                {"id": "item1", "content": "test content 1"},
                {"id": "item2", "content": "test content 2"}
            ],
            "blueprint_id": 1,
            "indexing_strategy": "hierarchical",
            "update_existing": False,
            "batch_size": 100
        }
        
        with patch('app.api.v1.blueprint_centric.blueprint_service.index_content') as mock_index:
            mock_index.return_value = IndexingResponse(
                request=IndexingRequest(**request_data),
                success=True,
                indexed_items=2,
                updated_items=0,
                failed_items=0
            )
            
            response = client.post("/v1/blueprint-centric/vector-store/index", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["indexed_items"] == 2
            assert data["failed_items"] == 0
            
            # Verify service was called
            mock_index.assert_called_once()
    
    def test_index_content_validation_error(self):
        """Test content indexing with validation errors."""
        # Test with invalid batch_size
        request_data = {
            "content_items": [{"id": "item1", "content": "test content"}],
            "blueprint_id": 1,
            "batch_size": 0  # Invalid
        }
        
        response = client.post("/v1/blueprint-centric/vector-store/index", json=request_data)
        assert response.status_code == 422
    
    def test_search_content_success(self):
        """Test successful content search endpoint."""
        request_data = {
            "query_text": "calculus derivatives",
            "user_id": 1,
            "max_results": 20,
            "similarity_threshold": 0.7,
            "include_hierarchy": True,
            "include_graph_context": True
        }
        
        with patch('app.api.v1.blueprint_centric.blueprint_service.search_content') as mock_search:
            mock_search.return_value = SearchResponse(
                query=SearchQuery(**request_data),
                results=[]
            )
            
            response = client.post("/v1/blueprint-centric/vector-store/search", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert "query" in data
            assert "results" in data
            
            # Verify service was called
            mock_search.assert_called_once()
    
    def test_search_content_validation_error(self):
        """Test content search with validation errors."""
        # Test with invalid max_results
        request_data = {
            "query_text": "calculus derivatives",
            "user_id": 1,
            "max_results": 0  # Invalid
        }
        
        response = client.post("/v1/blueprint-centric/vector-store/search", json=request_data)
        assert response.status_code == 422


class TestBlueprintManagementEndpoints:
    """Test blueprint management API endpoints."""
    
    def test_validate_blueprint_success(self):
        """Test successful blueprint validation endpoint."""
        with patch('app.api.v1.blueprint_centric.blueprint_service.validate_blueprint') as mock_validate:
            mock_validate.return_value = {
                "is_valid": True,
                "errors": [],
                "warnings": ["Consider adding more mastery criteria"],
                "recommendations": ["Add criteria for different UUE stages"]
            }
            
            response = client.post("/v1/blueprint-centric/blueprint/validate/1")
            
            assert response.status_code == 200
            data = response.json()
            assert data["is_valid"] is True
            assert len(data["errors"]) == 0
            assert len(data["warnings"]) == 1
            assert len(data["recommendations"]) == 1
            
            # Verify service was called
            mock_validate.assert_called_once()
    
    def test_validate_blueprint_service_error(self):
        """Test blueprint validation when service fails."""
        with patch('app.api.v1.blueprint_centric.blueprint_service.validate_blueprint') as mock_validate:
            mock_validate.side_effect = Exception("Service error")
            
            response = client.post("/v1/blueprint-centric/blueprint/validate/1")
            
            assert response.status_code == 500
            data = response.json()
            assert "Service error" in data["detail"]
    
    def test_get_blueprint_analytics_success(self):
        """Test successful blueprint analytics endpoint."""
        with patch('app.api.v1.blueprint_centric.blueprint_service.get_blueprint_analytics') as mock_analytics:
            mock_analytics.return_value = {
                "blueprint_id": 1,
                "user_id": 1,
                "total_sections": 5,
                "total_criteria": 20,
                "mastery_progress": 0.75,
                "learning_time": 120,
                "completion_rate": 0.6,
                "difficulty_distribution": {"BEGINNER": 8, "INTERMEDIATE": 10, "ADVANCED": 2},
                "uue_stage_progress": {"UNDERSTAND": 0.8, "USE": 0.7, "EXPLORE": 0.5},
                "recommendations": ["Focus on advanced concepts", "Practice more USE stage criteria"]
            }
            
            response = client.get("/v1/blueprint-centric/blueprint/analytics/1?user_id=1")
            
            assert response.status_code == 200
            data = response.json()
            assert data["blueprint_id"] == 1
            assert data["user_id"] == 1
            assert data["total_sections"] == 5
            assert data["mastery_progress"] == 0.75
            assert len(data["recommendations"]) == 2
            
            # Verify service was called
            mock_analytics.assert_called_once_with(1, 1)
    
    def test_get_blueprint_analytics_missing_user_id(self):
        """Test blueprint analytics with missing user_id parameter."""
        response = client.get("/v1/blueprint-centric/blueprint/analytics/1")
        assert response.status_code == 422  # Validation error


class TestHealthAndStatusEndpoints:
    """Test health and status API endpoints."""
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/v1/blueprint-centric/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "blueprint-centric"
        assert "timestamp" in data
        assert "version" in data
    
    def test_service_status(self):
        """Test service status endpoint."""
        response = client.get("/v1/blueprint-centric/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "blueprint-centric"
        assert data["status"] == "operational"
        assert "components" in data
        assert "timestamp" in data
        
        # Check component statuses
        components = data["components"]
        assert components["content_generation"] == "operational"
        assert components["knowledge_graph"] == "operational"
        assert components["vector_store"] == "operational"
        assert components["mastery_tracking"] == "operational"


class TestAPIErrorHandling:
    """Test API error handling and edge cases."""
    
    def test_malformed_json_request(self):
        """Test handling of malformed JSON requests."""
        response = client.post(
            "/v1/blueprint-centric/mastery-criteria/generate",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_missing_required_fields(self):
        """Test handling of requests with missing required fields."""
        request_data = {
            "blueprint_id": 1
            # Missing user_id and content_type
        }
        
        response = client.post("/v1/blueprint-centric/mastery-criteria/generate", json=request_data)
        assert response.status_code == 422
    
    def test_invalid_enum_values(self):
        """Test handling of invalid enum values."""
        request_data = {
            "blueprint_id": 1,
            "content_type": "mastery_criteria",
            "user_id": 1,
            "style": "invalid_style"  # Invalid enum value
        }
        
        response = client.post("/v1/blueprint-centric/mastery-criteria/generate", json=request_data)
        assert response.status_code == 422
    
    def test_large_request_bodies(self):
        """Test handling of large request bodies."""
        # Create a large content item
        large_content = "x" * 10000  # 10KB content
        
        request_data = {
            "content_items": [
                {"id": "item1", "content": large_content}
            ],
            "blueprint_id": 1
        }
        
        response = client.post("/v1/blueprint-centric/vector-store/index", json=request_data)
        # Should handle large requests gracefully
        assert response.status_code in [200, 413]  # 413 if too large
    
    def test_concurrent_requests(self):
        """Test handling of concurrent requests."""
        import threading
        import time
        
        results = []
        errors = []
        
        def make_request():
            try:
                response = client.get("/v1/blueprint-centric/health")
                results.append(response.status_code)
            except Exception as e:
                errors.append(str(e))
        
        # Make multiple concurrent requests
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert len(results) == 5
        assert all(status == 200 for status in results)
        assert len(errors) == 0


class TestAPIIntegration:
    """Test API integration scenarios."""
    
    def test_complete_workflow(self):
        """Test complete API workflow from content generation to analytics."""
        with patch('app.api.v1.blueprint_centric.blueprint_service') as mock_service:
            # Mock all service methods
            mock_service.generate_mastery_criteria = AsyncMock(return_value=[
                GeneratedMasteryCriterion(
                    title="Test Criterion",
                    description="Test description",
                    uue_stage=UueStage.UNDERSTAND,
                    weight=1.0,
                    complexity_score=3.0,
                    assessment_type=AssessmentType.QUESTION_BASED,
                    mastery_threshold=0.8
                )
            ])
            
            mock_service.generate_questions = AsyncMock(return_value=[
                QuestionFamily(
                    id="family_1",
                    mastery_criterion_id="criterion_1",
                    base_question="Test question?",
                    variations=[],
                    difficulty=DifficultyLevel.BEGINNER,
                    question_type=QuestionType.MULTIPLE_CHOICE,
                    uue_stage=UueStage.UNDERSTAND
                )
            ])

            mock_service.build_knowledge_graph = AsyncMock(return_value=Mock(
                id="graph_1",
                total_nodes=3,
                total_edges=2
            ))

            mock_service.validate_blueprint = AsyncMock(return_value={
                "is_valid": True,
                "errors": [],
                "warnings": [],
                "recommendations": []
            })

            mock_service.get_blueprint_analytics = AsyncMock(return_value={
                "blueprint_id": 1,
                "user_id": 1,
                "total_sections": 1,
                "total_criteria": 1,
                "mastery_progress": 0.0,
                "learning_time": 0,
                "completion_rate": 0.0,
                "difficulty_distribution": {},
                "uue_stage_progress": {},
                "recommendations": []
            })
            
            # 1. Generate mastery criteria
            criteria_response = client.post("/v1/blueprint-centric/mastery-criteria/generate", json={
                "blueprint_id": 1,
                "content_type": "mastery_criteria",
                "user_id": 1
            })
            assert criteria_response.status_code == 200
            
            # 2. Generate questions
            questions_response = client.post("/v1/blueprint-centric/questions/generate", json={
                "blueprint_id": 1,
                "content_type": "questions",
                "user_id": 1
            })
            assert questions_response.status_code == 200
            
            # 3. Build knowledge graph
            graph_response = client.post("/v1/blueprint-centric/knowledge-graph/build/1")
            assert graph_response.status_code == 200
            
            # 4. Validate blueprint
            validation_response = client.post("/v1/blueprint-centric/blueprint/validate/1")
            assert validation_response.status_code == 200
            
            # 5. Get analytics
            analytics_response = client.get("/v1/blueprint-centric/blueprint/analytics/1?user_id=1")
            assert analytics_response.status_code == 200
            
            # Verify all service methods were called
            assert mock_service.generate_mastery_criteria.called
            assert mock_service.generate_questions.called
            assert mock_service.build_knowledge_graph.called
            assert mock_service.validate_blueprint.called
            assert mock_service.get_blueprint_analytics.called


if __name__ == "__main__":
    pytest.main([__file__])

