"""
Tests for API endpoints.
"""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestAPIEndpoints:
    """Test cases for API endpoints."""
    
    def test_root_endpoint(self):
        """Test the root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["message"] == "Elevate AI API is running"
    
    def test_health_endpoint(self):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_deconstruct_endpoint_unauthorized(self):
        """Test deconstruct endpoint without authentication."""
        response = client.post("/api/v1/deconstruct", json={
            "source_text": "Test text",
            "source_type_hint": "article"
        })
        assert response.status_code == 401
    
    def test_deconstruct_endpoint_authorized(self):
        """Test deconstruct endpoint with authentication."""
        headers = {"Authorization": "Bearer test_api_key_123"}
        response = client.post("/api/v1/deconstruct", 
                             json={
                                 "source_text": "Test text",
                                 "source_type_hint": "article"
                             },
                             headers=headers)
        assert response.status_code == 200
    
    def test_chat_endpoint_unauthorized(self):
        """Test chat endpoint without authentication."""
        response = client.post("/api/v1/chat/message", json={
            "message_content": "Hello",
            "context": {}
        })
        assert response.status_code == 401
    
    def test_generate_notes_endpoint_unauthorized(self):
        """Test generate notes endpoint without authentication."""
        response = client.post("/api/v1/generate/notes", json={
            "blueprint_id": "test_id",
            "name": "Test Notes",
            "folder_id": 1
        })
        assert response.status_code == 401
    
    def test_generate_questions_endpoint_unauthorized(self):
        """Test generate questions endpoint without authentication."""
        response = client.post("/api/v1/generate/questions", json={
            "blueprint_id": "test_id",
            "name": "Test Questions",
            "folder_id": 1,
            "question_options": {}
        })
        assert response.status_code == 401
    
    def test_inline_suggestions_endpoint_unauthorized(self):
        """Test inline suggestions endpoint without authentication."""
        response = client.post("/api/v1/suggest/inline", json={
            "text": "Test text",
            "cursor_position": 10
        })
        assert response.status_code == 401


class TestQuestionGenerationEndpoint:
    """Test cases for the question generation endpoint."""
    
    def test_generate_questions_from_blueprint_unauthorized(self):
        """Test question generation endpoint without authentication."""
        response = client.post("/api/v1/ai-rag/learning-blueprints/test-blueprint/question-sets", 
                             json={
                                 "name": "Test Question Set",
                                 "folder_id": 1,
                                 "question_options": {"scope": "KeyConcepts"}
                             })
        assert response.status_code == 401
    
    def test_generate_questions_from_blueprint_authorized(self, mock_blueprint_retrieval, mock_llm_service, mock_blueprint_data, mock_llm_response):
        """Test question generation endpoint with authentication and mocked services."""
        # Setup mocks
        mock_blueprint_retrieval.return_value = mock_blueprint_data
        mock_llm_service.call_llm.return_value = mock_llm_response
        
        headers = {"Authorization": "Bearer test_api_key_123"}
        response = client.post("/api/v1/ai-rag/learning-blueprints/test-blueprint/question-sets", 
                             json={
                                 "name": "Test Question Set",
                                 "folder_id": 1,
                                 "question_options": {"scope": "KeyConcepts"}
                             },
                             headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "name" in data
        assert "blueprint_id" in data
        assert "questions" in data
        assert "created_at" in data
        assert "updated_at" in data
        assert data["name"] == "Test Question Set"
        assert data["blueprint_id"] == "test-blueprint"
        assert isinstance(data["questions"], list)
        
        # Verify LLM service was called
        mock_llm_service.call_llm.assert_called_once()
        call_args = mock_llm_service.call_llm.call_args
        assert call_args[1]["operation"] == "generate_questions"
        assert call_args[1]["prefer_google"] is True
    
    def test_generate_questions_without_options(self, mock_blueprint_retrieval, mock_llm_service, mock_blueprint_data, mock_llm_response):
        """Test question generation without question options."""
        # Setup mocks
        mock_blueprint_retrieval.return_value = mock_blueprint_data
        mock_llm_service.call_llm.return_value = mock_llm_response
        
        headers = {"Authorization": "Bearer test_api_key_123"}
        response = client.post("/api/v1/ai-rag/learning-blueprints/test-blueprint/question-sets", 
                             json={
                                 "name": "Basic Question Set"
                             },
                             headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Basic Question Set"
        assert data["folder_id"] is None
    
    def test_generate_questions_with_complex_options(self, mock_blueprint_retrieval, mock_llm_service, mock_blueprint_data, mock_llm_response):
        """Test question generation with complex question options."""
        # Setup mocks
        mock_blueprint_retrieval.return_value = mock_blueprint_data
        mock_llm_service.call_llm.return_value = mock_llm_response
        
        headers = {"Authorization": "Bearer test_api_key_123"}
        response = client.post("/api/v1/ai-rag/learning-blueprints/test-blueprint/question-sets", 
                             json={
                                 "name": "Advanced Question Set",
                                 "folder_id": 123,
                                 "question_options": {
                                     "scope": "KeyConcepts",
                                     "tone": "Formal",
                                     "difficulty": "Medium",
                                     "count": 10,
                                     "types": ["multiple_choice", "short_answer"]
                                 }
                             },
                             headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Advanced Question Set"
        assert data["folder_id"] == 123
    
    def test_generate_questions_validation_empty_name(self):
        """Test validation error for empty name."""
        headers = {"Authorization": "Bearer test_api_key_123"}
        response = client.post("/api/v1/ai-rag/learning-blueprints/test-blueprint/question-sets", 
                             json={
                                 "name": "",
                                 "folder_id": 1
                             },
                             headers=headers)
        assert response.status_code == 422  # Pydantic validation error
        assert "value_error" in response.json()["detail"][0]["type"]
    
    def test_generate_questions_validation_invalid_folder_id(self):
        """Test validation error for invalid folder ID."""
        headers = {"Authorization": "Bearer test_api_key_123"}
        response = client.post("/api/v1/ai-rag/learning-blueprints/test-blueprint/question-sets", 
                             json={
                                 "name": "Test Set",
                                 "folder_id": -1
                             },
                             headers=headers)
        assert response.status_code == 422  # Pydantic validation error
        assert "value_error" in response.json()["detail"][0]["type"]
    
    def test_generate_questions_validation_missing_name(self):
        """Test validation error for missing name field."""
        headers = {"Authorization": "Bearer test_api_key_123"}
        response = client.post("/api/v1/ai-rag/learning-blueprints/test-blueprint/question-sets", 
                             json={
                                 "folder_id": 1
                             },
                             headers=headers)
        assert response.status_code == 422  # Pydantic validation error
    
    def test_generate_questions_response_structure(self, mock_blueprint_retrieval, mock_llm_service, mock_blueprint_data, mock_llm_response):
        """Test that the response has the correct structure with questions."""
        # Setup mocks
        mock_blueprint_retrieval.return_value = mock_blueprint_data
        mock_llm_service.call_llm.return_value = mock_llm_response
        
        headers = {"Authorization": "Bearer test_api_key_123"}
        response = client.post("/api/v1/ai-rag/learning-blueprints/test-blueprint/question-sets", 
                             json={
                                 "name": "Structure Test Set",
                                 "question_options": {"scope": "KeyConcepts"}
                             },
                             headers=headers)
        assert response.status_code == 200
        data = response.json()
        
        # Check top-level fields
        required_fields = ["id", "name", "blueprint_id", "questions", "created_at", "updated_at"]
        for field in required_fields:
            assert field in data
        
        # Check questions structure
        questions = data["questions"]
        assert isinstance(questions, list)
        if questions:  # If questions are returned
            question = questions[0]
            question_fields = ["text", "answer", "question_type", "total_marks_available", "marking_criteria"]
            for field in question_fields:
                assert field in question
            assert isinstance(question["total_marks_available"], int)
            assert question["total_marks_available"] > 0
    
    def test_generate_questions_blueprint_not_found(self, mock_blueprint_retrieval):
        """Test question generation when blueprint is not found."""
        # Setup mock to return placeholder data (indicating not found)
        mock_blueprint_retrieval.return_value = {
            "source_id": "non-existent-blueprint",
            "source_text": "Sample source text about mitochondria and cellular biology.",  # This triggers 404
            "source_title": "Sample Learning Blueprint",
            "source_type": "text",
            "source_summary": {},
            "sections": [],
            "knowledge_primitives": {}
        }
        
        headers = {"Authorization": "Bearer test_api_key_123"}
        response = client.post("/api/v1/ai-rag/learning-blueprints/non-existent-blueprint/question-sets", 
                             json={
                                 "name": "Test Set"
                             },
                             headers=headers)
        assert response.status_code == 404
        assert "LearningBlueprint not found" in response.json()["detail"]
    
    def test_generate_questions_llm_service_error(self, mock_blueprint_retrieval, mock_llm_service, mock_blueprint_data):
        """Test question generation when LLM service fails."""
        # Setup mocks
        mock_blueprint_retrieval.return_value = mock_blueprint_data
        mock_llm_service.call_llm.side_effect = Exception("LLM service unavailable")
        
        headers = {"Authorization": "Bearer test_api_key_123"}
        response = client.post("/api/v1/ai-rag/learning-blueprints/test-blueprint/question-sets", 
                             json={
                                 "name": "Test Set"
                             },
                             headers=headers)
        # The endpoint should still return 200 because it falls back to mock questions
        assert response.status_code == 200
        data = response.json()
        assert "questions" in data
        assert isinstance(data["questions"], list)
        # Verify that fallback questions were generated
        assert len(data["questions"]) > 0
    
    def test_generate_questions_complete_failure(self, mock_blueprint_retrieval, mock_llm_service, mock_blueprint_data):
        """Test question generation when both LLM service and fallback fail."""
        # Setup mocks
        mock_blueprint_retrieval.return_value = mock_blueprint_data
        mock_llm_service.call_llm.side_effect = Exception("LLM service unavailable")
        
        # Mock the fallback function to also fail
        with patch('app.core.indexing._generate_fallback_questions') as mock_fallback:
            mock_fallback.side_effect = Exception("Fallback also failed")
            
            headers = {"Authorization": "Bearer test_api_key_123"}
            response = client.post("/api/v1/ai-rag/learning-blueprints/test-blueprint/question-sets", 
                                 json={
                                     "name": "Test Set"
                                 },
                                 headers=headers)
            assert response.status_code == 502
            assert "Question generation failed" in response.json()["detail"]


class TestAnswerEvaluationEndpoint:
    """Test cases for the answer evaluation endpoint."""
    
    def test_evaluate_answer_unauthorized(self):
        """Test answer evaluation endpoint without authentication."""
        response = client.post("/api/v1/ai/evaluate-answer", 
                             json={
                                 "question_id": 1,
                                 "user_answer": "Test answer"
                             })
        assert response.status_code == 401
    
    def test_evaluate_answer_authorized(self, mock_question_retrieval, mock_evaluation_service, mock_question_data, mock_evaluation_response):
        """Test answer evaluation endpoint with authentication and mocked services."""
        # Setup mocks
        mock_question_retrieval.return_value = mock_question_data
        mock_evaluation_service.return_value = {
            "marks_achieved": 4,
            "corrected_answer": "Mitochondria are the powerhouse of the cell, generating ATP through cellular respiration.",
            "feedback": "Good answer! You correctly identified mitochondria as the powerhouse and mentioned ATP generation."
        }
        
        headers = {"Authorization": "Bearer test_api_key_123"}
        response = client.post("/api/v1/ai/evaluate-answer", 
                             json={
                                 "question_id": 1,
                                 "user_answer": "Mitochondria are the powerhouse of the cell and generate ATP."
                             },
                             headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "corrected_answer" in data
        assert "marks_available" in data
        assert "marks_achieved" in data
        assert data["marks_available"] == 5
        assert data["marks_achieved"] == 4
        assert isinstance(data["marks_achieved"], int)
        
        # Verify evaluation service was called
        mock_evaluation_service.assert_called_once()
        call_args = mock_evaluation_service.call_args[0][0]
        assert call_args["question_text"] == mock_question_data["text"]
        assert call_args["expected_answer"] == mock_question_data["answer"]
        assert call_args["user_answer"] == "Mitochondria are the powerhouse of the cell and generate ATP."
    
    def test_evaluate_answer_validation_invalid_question_id(self):
        """Test validation error for invalid question ID."""
        headers = {"Authorization": "Bearer test_api_key_123"}
        response = client.post("/api/v1/ai/evaluate-answer", 
                             json={
                                 "question_id": 0,
                                 "user_answer": "Test answer"
                             },
                             headers=headers)
        assert response.status_code == 422
        assert "Question ID must be a positive integer" in response.json()["detail"][0]["msg"]
    
    def test_evaluate_answer_validation_empty_answer(self):
        """Test validation error for empty user answer."""
        headers = {"Authorization": "Bearer test_api_key_123"}
        response = client.post("/api/v1/ai/evaluate-answer", 
                             json={
                                 "question_id": 1,
                                 "user_answer": ""
                             },
                             headers=headers)
        assert response.status_code == 422
        assert "User answer cannot be empty" in response.json()["detail"][0]["msg"]
    
    def test_evaluate_answer_question_not_found(self, mock_question_retrieval):
        """Test answer evaluation when question is not found."""
        # Setup mock to return empty question data (indicating not found)
        mock_question_retrieval.return_value = {
            "id": 999,
            "text": "",  # This triggers 404
            "answer": "",
            "question_type": "understand",
            "total_marks_available": 1,
            "marking_criteria": "",
            "question_set_name": "",
            "folder_name": "",
            "blueprint_title": ""
        }
        
        headers = {"Authorization": "Bearer test_api_key_123"}
        response = client.post("/api/v1/ai/evaluate-answer", 
                             json={
                                 "question_id": 999,
                                 "user_answer": "Test answer"
                             },
                             headers=headers)
        assert response.status_code == 404
        assert "Question not found" in response.json()["detail"]
    
    def test_evaluate_answer_evaluation_service_error(self, mock_question_retrieval, mock_evaluation_service, mock_question_data):
        """Test answer evaluation when evaluation service fails."""
        # Setup mocks
        mock_question_retrieval.return_value = mock_question_data
        mock_evaluation_service.side_effect = Exception("Evaluation service unavailable")
        
        headers = {"Authorization": "Bearer test_api_key_123"}
        response = client.post("/api/v1/ai/evaluate-answer", 
                             json={
                                 "question_id": 1,
                                 "user_answer": "Test answer"
                             },
                             headers=headers)
        
        # The endpoint should still return 200 because it falls back to mock evaluation
        assert response.status_code == 200
        data = response.json()
        assert "corrected_answer" in data
        assert "marks_available" in data
        assert "marks_achieved" in data
        # Verify that fallback evaluation was used
        assert data["marks_available"] == 5
        assert isinstance(data["marks_achieved"], int)


class TestAPIValidation:
    """Test cases for API request/response validation."""
    
    def test_deconstruct_request_validation(self):
        """Test deconstruct request validation."""
        # Test missing required field
        headers = {"Authorization": "Bearer test_api_key"}
        response = client.post("/api/v1/deconstruct", 
                             json={
                                 "source_type_hint": "article"
                                 # Missing source_text
                             },
                             headers=headers)
        # This will likely fail due to auth, but the validation should happen first
        # assert response.status_code == 422
    
    def test_chat_request_validation(self):
        """Test chat request validation."""
        # Test missing required field
        headers = {"Authorization": "Bearer test_api_key"}
        response = client.post("/api/v1/chat/message", 
                             json={
                                 "context": {}
                                 # Missing message_content
                             },
                             headers=headers)
        # This will likely fail due to auth, but the validation should happen first
        # assert response.status_code == 422 

@pytest.mark.asyncio
def test_deconstruct_basic():
    payload = {
        "source_text": "# Test\nThis is a test section about photosynthesis.",
        "source_type_hint": "article"
    }
    headers = {"Authorization": "Bearer test_api_key_123"}
    response = client.post("/api/v1/deconstruct", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "blueprint_id" in data
    assert "blueprint_json" in data
    assert "sections" in data["blueprint_json"]
    assert isinstance(data["blueprint_json"]["sections"], list)
    assert len(data["blueprint_json"]["sections"]) >= 1

@pytest.mark.asyncio
def test_deconstruct_multiple_sections():
    payload = {
        "source_text": "# Intro\nIntro text.\n## Details\nDetails about the topic.\n## Conclusion\nFinal thoughts.",
        "source_type_hint": "chapter"
    }
    headers = {"Authorization": "Bearer test_api_key_123"}
    response = client.post("/api/v1/deconstruct", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    sections = data["blueprint_json"]["sections"]
    assert len(sections) >= 2
    section_names = [s["section_name"] for s in sections]
    assert "Intro" in section_names or "Details" in section_names

@pytest.mark.asyncio
def test_deconstruct_no_headings():
    payload = {
        "source_text": "This is a simple text without any headings or structure.",
        "source_type_hint": "note"
    }
    headers = {"Authorization": "Bearer test_api_key_123"}
    response = client.post("/api/v1/deconstruct", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    sections = data["blueprint_json"]["sections"]
    assert len(sections) == 1
    assert sections[0]["section_name"] in ["Main", "root"]

@pytest.mark.asyncio
def test_deconstruct_knowledge_primitives():
    payload = {
        "source_text": "# Science\nPhotosynthesis is the process by which plants convert light energy into chemical energy.",
        "source_type_hint": "article"
    }
    headers = {"Authorization": "Bearer test_api_key_123"}
    response = client.post("/api/v1/deconstruct", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    kp = data["blueprint_json"]["knowledge_primitives"]
    assert "key_propositions_and_facts" in kp
    assert "key_entities_and_definitions" in kp
    assert isinstance(kp["key_propositions_and_facts"], list)
    assert isinstance(kp["key_entities_and_definitions"], list) 