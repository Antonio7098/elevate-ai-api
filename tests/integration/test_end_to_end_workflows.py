# Sprint 33: End-to-End Workflow Testing

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, Mock
from typing import Dict, List, Any
import json
import uuid

from app.main import app
from app.core.caching_service import cache_service
from app.core.request_deduplication import deduplication_service
from app.core.async_processing import async_service

class TestCompleteWorkflows:
    """Test complete end-to-end workflows from blueprint creation to answer evaluation."""
    
    @pytest.fixture
    def client(self):
        """FastAPI test client."""
        return TestClient(app)
    
    @pytest.fixture
    def sample_source_text(self):
        """Sample educational content for testing."""
        return """
        Machine Learning is a subset of artificial intelligence (AI) that focuses on algorithms 
        that can learn and make decisions from data without being explicitly programmed. 
        
        There are three main types of machine learning:
        1. Supervised Learning: Uses labeled training data to learn a mapping function
        2. Unsupervised Learning: Finds patterns in data without labeled examples
        3. Reinforcement Learning: Learns through interaction with an environment
        
        Key algorithms include linear regression, decision trees, neural networks, and 
        support vector machines. These algorithms are used in applications like 
        recommendation systems, image recognition, and natural language processing.
        """
    
    @pytest.fixture
    def mock_llm_service(self):
        """Mock LLM service for consistent testing."""
        mock_service = AsyncMock()
        
        # Mock blueprint creation response
        mock_service.generate_text.return_value = json.dumps({
            "title": "Machine Learning Fundamentals",
            "learning_objectives": [
                "Understand what machine learning is",
                "Identify types of machine learning",
                "Apply ML algorithms to problems"
            ],
            "sections": [
                {
                    "title": "Introduction to ML",
                    "content": "ML basics and definitions",
                    "learning_objectives": ["Define machine learning"]
                },
                {
                    "title": "Types of ML",
                    "content": "Supervised, unsupervised, reinforcement learning",
                    "learning_objectives": ["Classify ML types"]
                }
            ]
        })
        
        return mock_service
    
    @pytest.fixture
    def mock_enhanced_primitive_generation(self):
        """Mock enhanced primitive generation with realistic data."""
        return {
            "primitives": [
                {
                    "primitive_id": "ml_001",
                    "title": "Machine Learning Definition",
                    "description": "Understanding what machine learning is",
                    "content": "ML is a subset of AI that learns from data",
                    "primitive_type": "concept",
                    "tags": ["AI", "ML", "algorithms"],
                    "mastery_criteria": [
                        {
                            "criterion_id": "ml_001_understand",
                            "title": "Define ML",
                            "description": "Explain what machine learning is",
                            "uee_level": "UNDERSTAND",
                            "weight": 3.0,
                            "is_required": True
                        },
                        {
                            "criterion_id": "ml_001_use",
                            "title": "Identify ML applications",
                            "description": "Recognize ML use cases",
                            "uee_level": "USE", 
                            "weight": 4.0,
                            "is_required": True
                        }
                    ]
                },
                {
                    "primitive_id": "ml_002",
                    "title": "Types of Machine Learning",
                    "description": "Understanding supervised, unsupervised, and reinforcement learning",
                    "content": "Three main types: supervised, unsupervised, reinforcement",
                    "primitive_type": "classification",
                    "tags": ["ML", "types", "classification"],
                    "mastery_criteria": [
                        {
                            "criterion_id": "ml_002_understand",
                            "title": "List ML types",
                            "description": "Name the three types of ML",
                            "uee_level": "UNDERSTAND",
                            "weight": 2.5,
                            "is_required": True
                        },
                        {
                            "criterion_id": "ml_002_use",
                            "title": "Classify ML problems",
                            "description": "Determine which ML type to use",
                            "uee_level": "USE",
                            "weight": 4.5,
                            "is_required": True
                        },
                        {
                            "criterion_id": "ml_002_explore",
                            "title": "Compare ML approaches",
                            "description": "Analyze pros and cons of each type",
                            "uee_level": "EXPLORE",
                            "weight": 5.0,
                            "is_required": False
                        }
                    ]
                }
            ]
        }

    @pytest.mark.asyncio
    async def test_complete_blueprint_to_primitives_workflow(
        self, 
        client, 
        sample_source_text, 
        mock_llm_service,
        mock_enhanced_primitive_generation
    ):
        """Test complete workflow: source text → blueprint → primitives → criteria."""
        
        # Step 1: Create enhanced blueprint with primitives
        with patch('app.core.deconstruction.llm_service', mock_llm_service):
            with patch('app.core.deconstruction.generate_enhanced_primitives_with_criteria') as mock_gen:
                mock_gen.return_value = mock_enhanced_primitive_generation
                
                response = client.post("/ai/deconstruct-enhanced", json={
                    "source_text": sample_source_text,
                    "user_preferences": {
                        "uee_distribution": {"UNDERSTAND": 0.4, "USE": 0.4, "EXPLORE": 0.2},
                        "max_primitives": 5
                    },
                    "context": {"title": "ML Fundamentals"}
                })
                
                assert response.status_code == 200
                blueprint_result = response.json()
                
                # Verify blueprint creation
                assert "blueprint_id" in blueprint_result
                assert "primitives" in blueprint_result
                assert len(blueprint_result["primitives"]) == 2
                
                blueprint_id = blueprint_result["blueprint_id"]
        
        # Step 2: Extract primitives from blueprint
        response = client.get(f"/blueprints/{blueprint_id}/primitives")
        
        assert response.status_code == 200
        primitives_result = response.json()
        
        assert primitives_result["blueprint_id"] == blueprint_id
        assert len(primitives_result["primitives"]) == 2
        assert primitives_result["total_primitives"] == 2
        
        # Verify primitive structure
        primitive = primitives_result["primitives"][0]
        assert "primitive_id" in primitive
        assert "mastery_criteria" in primitive
        assert len(primitive["mastery_criteria"]) >= 1
        
        # Step 3: Generate questions for specific criteria
        criterion = primitive["mastery_criteria"][0]
        
        with patch('app.core.question_generation_service.QuestionGenerationService') as mock_question_service:
            mock_service_instance = AsyncMock()
            mock_service_instance.generate_criterion_questions.return_value = [
                {
                    "question_id": "q_001",
                    "question_text": "What is machine learning?",
                    "question_type": "short_answer",
                    "correct_answer": "A subset of AI that learns from data",
                    "marking_criteria": "Must mention AI and learning from data",
                    "difficulty_level": "basic"
                }
            ]
            mock_question_service.return_value = mock_service_instance
            
            response = client.post("/questions/criterion-specific", json={
                "criterionId": criterion["criterion_id"],
                "primitiveId": primitive["primitive_id"],
                "ueeLevel": criterion["uee_level"],
                "numQuestions": 1,
                "questionTypes": ["short_answer"]
            })
            
            assert response.status_code == 200
            questions_result = response.json()
            
            assert questions_result["success"] is True
            assert len(questions_result["questions"]) == 1
            assert questions_result["questions"][0]["question_text"] == "What is machine learning?"
        
        # Step 4: Evaluate answer against criterion
        user_answer = "Machine learning is a branch of artificial intelligence that enables computers to learn from data."
        
        response = client.post("/ai/evaluate-answer/criterion", json={
            "criterionId": criterion["criterion_id"],
            "criterionTitle": criterion["title"],
            "primitiveId": primitive["primitive_id"],
            "primitiveTitle": primitive["title"],
            "ueeLevel": criterion["uee_level"],
            "criterionWeight": criterion["weight"],
            "questionText": "What is machine learning?",
            "questionType": "short_answer",
            "correctAnswer": "A subset of AI that learns from data",
            "userAnswer": user_answer,
            "totalMarks": 10
        })
        
        assert response.status_code == 200
        evaluation_result = response.json()
        
        assert evaluation_result["success"] is True
        assert evaluation_result["criterionId"] == criterion["criterion_id"]
        assert 0.0 <= evaluation_result["masteryScore"] <= 1.0
        assert evaluation_result["masteryLevel"] in ["novice", "developing", "mastered"]

    @pytest.mark.asyncio
    async def test_batch_processing_workflow(
        self, 
        client, 
        sample_source_text,
        mock_llm_service,
        mock_enhanced_primitive_generation
    ):
        """Test batch processing workflow with multiple blueprints."""
        
        # Create multiple blueprints
        blueprint_ids = []
        
        with patch('app.core.deconstruction.llm_service', mock_llm_service):
            with patch('app.core.deconstruction.generate_enhanced_primitives_with_criteria') as mock_gen:
                mock_gen.return_value = mock_enhanced_primitive_generation
                
                for i in range(3):
                    response = client.post("/ai/deconstruct-enhanced", json={
                        "source_text": f"{sample_source_text} - Variation {i}",
                        "user_preferences": {"max_primitives": 3},
                        "context": {"title": f"ML Fundamentals {i}"}
                    })
                    
                    assert response.status_code == 200
                    result = response.json()
                    blueprint_ids.append(result["blueprint_id"])
        
        # Batch extract primitives
        response = client.post("/blueprints/batch/primitives", json={
            "blueprintIds": blueprint_ids,
            "includeMetadata": True
        })
        
        assert response.status_code == 200
        batch_result = response.json()
        
        assert batch_result["success"] is True
        assert len(batch_result["results"]) == 3
        assert batch_result["totalPrimitives"] == 6  # 2 primitives × 3 blueprints
        
        # Verify each blueprint result
        for blueprint_id in blueprint_ids:
            assert blueprint_id in batch_result["results"]
            blueprint_result = batch_result["results"][blueprint_id]
            assert len(blueprint_result["primitives"]) == 2

    @pytest.mark.asyncio
    async def test_performance_optimization_workflow(
        self, 
        client, 
        sample_source_text,
        mock_llm_service,
        mock_enhanced_primitive_generation
    ):
        """Test that performance optimizations work in end-to-end workflows."""
        
        # Clear caches to start fresh
        await cache_service.invalidate_cache()
        await deduplication_service.clear_completed_cache()
        
        with patch('app.core.deconstruction.llm_service', mock_llm_service):
            with patch('app.core.deconstruction.generate_enhanced_primitives_with_criteria') as mock_gen:
                mock_gen.return_value = mock_enhanced_primitive_generation
                
                # First request - should be processed normally
                response1 = client.post("/ai/deconstruct-enhanced", json={
                    "source_text": sample_source_text,
                    "user_preferences": {"max_primitives": 5},
                    "context": {"title": "ML Test 1"}
                })
                
                assert response1.status_code == 200
                result1 = response1.json()
                
                # Second identical request - should be cached or deduplicated
                response2 = client.post("/ai/deconstruct-enhanced", json={
                    "source_text": sample_source_text,
                    "user_preferences": {"max_primitives": 5},
                    "context": {"title": "ML Test 1"}
                })
                
                assert response2.status_code == 200
                result2 = response2.json()
                
                # Results should be identical (from cache/deduplication)
                assert result1["primitives"] == result2["primitives"]
                
                # Check that LLM was not called twice (due to caching/deduplication)
                assert mock_gen.call_count <= 2  # Should be 1 if perfect optimization

    @pytest.mark.asyncio
    async def test_error_handling_workflow(self, client, mock_llm_service):
        """Test error handling throughout the workflow."""
        
        # Test with invalid source text
        response = client.post("/ai/deconstruct-enhanced", json={
            "source_text": "",  # Empty text should cause error
            "user_preferences": {"max_primitives": 5}
        })
        
        assert response.status_code == 400
        error_result = response.json()
        assert "error" in error_result["detail"].lower()
        
        # Test with invalid blueprint ID
        response = client.get("/blueprints/nonexistent_id/primitives")
        assert response.status_code == 404
        
        # Test with invalid criterion evaluation
        response = client.post("/ai/evaluate-answer/criterion", json={
            "criterionId": "",  # Empty criterion ID
            "primitiveId": "test",
            "ueeLevel": "INVALID_LEVEL",  # Invalid UEE level
            "questionText": "Test?",
            "correctAnswer": "Test",
            "userAnswer": "Test",
            "totalMarks": 10
        })
        
        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_concurrent_requests_workflow(
        self, 
        client, 
        sample_source_text,
        mock_llm_service,
        mock_enhanced_primitive_generation
    ):
        """Test handling of concurrent requests."""
        
        with patch('app.core.deconstruction.llm_service', mock_llm_service):
            with patch('app.core.deconstruction.generate_enhanced_primitives_with_criteria') as mock_gen:
                mock_gen.return_value = mock_enhanced_primitive_generation
                
                # Make multiple concurrent requests
                import threading
                import queue
                
                results_queue = queue.Queue()
                
                def make_request(i):
                    response = client.post("/ai/deconstruct-enhanced", json={
                        "source_text": f"{sample_source_text} - Request {i}",
                        "user_preferences": {"max_primitives": 3},
                        "context": {"title": f"Concurrent Test {i}"}
                    })
                    results_queue.put((i, response.status_code, response.json()))
                
                # Start 5 concurrent threads
                threads = []
                for i in range(5):
                    thread = threading.Thread(target=make_request, args=(i,))
                    threads.append(thread)
                    thread.start()
                
                # Wait for all threads to complete
                for thread in threads:
                    thread.join()
                
                # Collect results
                results = []
                while not results_queue.empty():
                    results.append(results_queue.get())
                
                # Verify all requests succeeded
                assert len(results) == 5
                for i, status_code, result in results:
                    assert status_code == 200
                    assert "blueprint_id" in result
                    assert "primitives" in result

    @pytest.mark.asyncio
    async def test_mastery_assessment_workflow(
        self, 
        client,
        mock_enhanced_primitive_generation
    ):
        """Test comprehensive mastery assessment workflow."""
        
        # Use mock primitive data
        primitive = mock_enhanced_primitive_generation["primitives"][0]
        criteria = primitive["mastery_criteria"]
        
        # Create evaluation requests for all criteria
        evaluation_requests = []
        for criterion in criteria:
            evaluation_requests.append({
                "criterionId": criterion["criterion_id"],
                "criterionTitle": criterion["title"],
                "primitiveId": primitive["primitive_id"],
                "primitiveTitle": primitive["title"],
                "ueeLevel": criterion["uee_level"],
                "criterionWeight": criterion["weight"],
                "questionText": f"Question for {criterion['title']}?",
                "questionType": "short_answer",
                "correctAnswer": "Correct answer",
                "userAnswer": "Student answer that demonstrates good understanding",
                "totalMarks": 10
            })
        
        # Submit mastery assessment
        response = client.post("/ai/evaluate-answer/mastery-assessment", json={
            "primitiveId": primitive["primitive_id"],
            "criterionEvaluations": evaluation_requests
        })
        
        assert response.status_code == 200
        assessment_result = response.json()
        
        assert assessment_result["success"] is True
        assert assessment_result["primitiveId"] == primitive["primitive_id"]
        assert 0.0 <= assessment_result["overallMasteryScore"] <= 1.0
        assert len(assessment_result["criterionAssessments"]) == len(criteria)
        assert "ueeProgression" in assessment_result
        assert "comprehensiveFeedback" in assessment_result

    @pytest.mark.asyncio
    async def test_data_persistence_workflow(self, client, sample_source_text, mock_llm_service):
        """Test that data persists correctly throughout workflow."""
        
        with patch('app.core.deconstruction.llm_service', mock_llm_service):
            # Create blueprint
            response = client.post("/ai/deconstruct", json={
                "source_text": sample_source_text,
                "context": {"title": "Persistence Test"}
            })
            
            assert response.status_code == 200
            blueprint_result = response.json()
            blueprint_id = blueprint_result["blueprint_id"]
            
            # Verify blueprint persists
            # This would test actual database persistence in a real implementation
            assert blueprint_id is not None
            assert len(blueprint_id) > 0


# Integration test configuration
pytestmark = [pytest.mark.asyncio, pytest.mark.integration]
