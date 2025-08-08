"""
True Integration Tests: Core API + AI API Communication

This file contains integration tests that require both elevate-core-api and elevate-ai-api
services to be running and communicating with each other. These tests validate the
actual integration between the two systems.

Prerequisites:
1. elevate-core-api running on http://localhost:3000
2. elevate-ai-api running on http://localhost:8000
3. Database and vector store services available
"""

import os
import pytest
import pytest_asyncio
import asyncio
import httpx
import json
from typing import Dict, List, Any
import time
import uuid

# Configuration
CORE_API_BASE_URL = "http://localhost:3000"
AI_API_BASE_URL = "http://localhost:8000"
TEST_TIMEOUT = 30  # seconds

# Authentication headers
AI_API_KEY = os.getenv("AI_API_KEY") or os.getenv("ELEVATE_AI_API_KEY") or os.getenv("API_KEY")
AI_API_HEADERS = {"Authorization": f"Bearer {AI_API_KEY}"} if AI_API_KEY else {}

# Core API uses test token for development
CORE_API_HEADERS = {
    "Authorization": "Bearer test123",
    "x-test-user-id": "1",
    "Content-Type": "application/json"
}

# Default headers for AI API (backward compatibility)
DEFAULT_HEADERS = AI_API_HEADERS


class TestCoreAiApiIntegration:
    """Integration tests for Core API + AI API communication."""
    
    @pytest_asyncio.fixture(scope="class")
    async def verify_services_running(self):
        """Verify both services are running before running integration tests."""
        async with httpx.AsyncClient(headers=DEFAULT_HEADERS) as client:
            try:
                ai_healthy = False
                core_healthy = False
                
                # Check AI API health (this should work without auth)
                try:
                    ai_response = await client.get(f"{AI_API_BASE_URL}/health", timeout=5.0)
                    ai_healthy = ai_response.status_code == 200
                    print(f"📡 AI API health: {ai_response.status_code}")
                except Exception as e:
                    print(f"❌ AI API health check failed: {e}")
                
                # Check Core API health (may require auth - 401 is acceptable if service is running)
                try:
                    core_response = await client.get(f"{CORE_API_BASE_URL}/api/health", timeout=5.0)
                    # Accept both 200 (OK) and 401 (Unauthorized) as signs the service is running
                    core_healthy = core_response.status_code in [200, 401]
                    print(f"📡 Core API health: {core_response.status_code} (auth required is OK)")
                except Exception as e:
                    print(f"❌ Core API health check failed: {e}")
                
                if ai_healthy and core_healthy:
                    print("✅ Both Core API and AI API are running")
                    return True
                else:
                    pytest.skip(f"Services not available - AI API: {ai_healthy}, Core API: {core_healthy}")
                    
            except Exception as e:
                pytest.skip(f"Services not available for integration testing: {str(e)}")
    
    @pytest.fixture
    def sample_educational_content(self):
        """Sample educational content for testing."""
        return """
        Photosynthesis is the process by which plants convert light energy into chemical energy.
        
        The process occurs in two main stages:
        1. Light-dependent reactions: Occur in the thylakoid membranes
        2. Light-independent reactions (Calvin cycle): Occur in the stroma
        
        The overall equation is:
        6CO₂ + 6H₂O + light energy → C₆H₁₂O₆ + 6O₂
        
        Key components include chlorophyll, ATP, and NADPH.
        """

    @pytest.mark.asyncio
    async def test_end_to_end_blueprint_workflow(
        self, 
        verify_services_running, 
        sample_educational_content
    ):
        """
        Test complete workflow:
        1. AI API: Create blueprint from text
        2. Core API: Store blueprint 
        3. AI API: Extract primitives
        4. Core API: Store primitives
        5. Verify data consistency
        """
        async with httpx.AsyncClient(headers=DEFAULT_HEADERS) as client:
            blueprint_id = str(uuid.uuid4())
            
            # Step 1: Create blueprint via AI API
            deconstruct_response = await client.post(
                f"{AI_API_BASE_URL}/api/v1/deconstruct",
                json={
                    "source_text": sample_educational_content,
                    "context": {
                        "title": "Photosynthesis Basics",
                        "subject": "Biology"
                    }
                },
                timeout=TEST_TIMEOUT
            )
            
            assert deconstruct_response.status_code == 200
            blueprint_data = deconstruct_response.json()
            
            # Verify blueprint structure
            assert "blueprint_id" in blueprint_data
            assert "blueprint_json" in blueprint_data
            
            blueprint_id = blueprint_data["blueprint_id"]
            blueprint_json = blueprint_data["blueprint_json"]
            
            # Verify nested blueprint structure
            assert "sections" in blueprint_json
            assert "source_id" in blueprint_json
            
            # Step 2: Store blueprint in Core API
            # Core API expects only sourceText - it will call AI API /deconstruct endpoint internally
            async with httpx.AsyncClient(headers=CORE_API_HEADERS) as core_client:
                core_blueprint_response = await core_client.post(
                    f"{CORE_API_BASE_URL}/api/learning-blueprints",
                    json={
                        "sourceText": """
        Photosynthesis is the process by which plants convert light energy into chemical energy.
        
        The process occurs in two main stages:
        1. Light-dependent reactions: Occur in the thylakoid membranes
        2. Light-independent reactions (Calvin cycle): Occur in the stroma
        
        The overall equation is:
        6CO₂ + 6H₂O + light energy → C₆H₁₂O₆ + 6O₂
        
        Key components include chlorophyll, ATP, and NADPH.
        """.strip()
                    },
                    timeout=TEST_TIMEOUT
                )
            
            assert core_blueprint_response.status_code in [200, 201]
            
            # Step 3: Extract primitives via AI API
            primitives_response = await client.get(
                f"{AI_API_BASE_URL}/api/v1/blueprints/{blueprint_id}/primitives",
                timeout=TEST_TIMEOUT
            )
            
            assert primitives_response.status_code == 200
            primitives_data = primitives_response.json()
            
            # Verify primitives structure
            assert "primitives" in primitives_data
            assert len(primitives_data["primitives"]) > 0
            
            primitive = primitives_data["primitives"][0]
            assert "primitiveId" in primitive
            assert "title" in primitive
            assert "primitiveType" in primitive
            assert "masteryCriteria" in primitive
            
            # Step 4: Store primitives in Core API
            for primitive in primitives_data["primitives"]:
                core_primitive_response = await client.post(
                    f"{CORE_API_BASE_URL}/api/v1/primitives",
                    json={
                        "primitiveId": primitive["primitiveId"],
                        "blueprintId": blueprint_id,
                        "title": primitive["title"],
                        "description": primitive.get("description", ""),
                        "primitiveType": primitive["primitiveType"],
                        "difficultyLevel": primitive.get("difficultyLevel", "INTERMEDIATE"),
                        "content": primitive.get("content", "")
                    },
                    timeout=TEST_TIMEOUT
                )
                
                assert core_primitive_response.status_code in [200, 201]
                
                # Store mastery criteria
                for criterion in primitive["masteryCriteria"]:
                    core_criterion_response = await client.post(
                        f"{CORE_API_BASE_URL}/api/v1/mastery-criteria",
                        json={
                            "criterionId": criterion["criterionId"],
                            "primitiveId": primitive["primitiveId"],
                            "title": criterion["title"],
                            "description": criterion.get("description", ""),
                            "ueeLevel": criterion["ueeLevel"],
                            "weight": criterion["weight"],
                            "isRequired": criterion.get("isRequired", True)
                        },
                        timeout=TEST_TIMEOUT
                    )
                    
                    assert core_criterion_response.status_code in [200, 201]
            
            # Step 5: Verify data consistency between services
            # Check that Core API has the stored data
            core_blueprint_get = await client.get(
                f"{CORE_API_BASE_URL}/api/v1/blueprints/{blueprint_id}",
                timeout=TEST_TIMEOUT
            )
            
            assert core_blueprint_get.status_code == 200
            stored_blueprint = core_blueprint_get.json()
            assert stored_blueprint["blueprintId"] == blueprint_id
            
            # Check that AI API can still access the blueprint
            ai_blueprint_get = await client.get(
                f"{AI_API_BASE_URL}/api/v1/blueprints/{blueprint_id}/primitives",
                timeout=TEST_TIMEOUT
            )
            
            assert ai_blueprint_get.status_code == 200
            
            print(f"✅ End-to-end blueprint workflow successful for blueprint: {blueprint_id}")

    @pytest.mark.asyncio
    async def test_primitive_sync_integration(
        self, 
        verify_services_running,
        sample_educational_content
    ):
        """
        Test primitive synchronization between AI API and Core API:
        1. AI API generates primitives
        2. AI API syncs primitives to Core API
        3. Verify sync status and data consistency
        """
        async with httpx.AsyncClient(headers=DEFAULT_HEADERS) as client:
            blueprint_id = str(uuid.uuid4())
            
            # Step 1: Generate primitives via AI API
            primitive_gen_response = await client.post(
                f"{AI_API_BASE_URL}/api/v1/primitives/generate",
                json={
                    "sourceText": sample_educational_content,
                    "blueprintId": blueprint_id,
                    "userPreferences": {
                        "maxPrimitives": 3,
                        "ueeDistribution": {
                            "UNDERSTAND": 0.4,
                            "USE": 0.4,
                            "EXPLORE": 0.2
                        }
                    }
                },
                timeout=TEST_TIMEOUT
            )
            
            assert primitive_gen_response.status_code == 200
            generation_data = primitive_gen_response.json()
            
            assert "primitives" in generation_data
            assert len(generation_data["primitives"]) > 0
            
            # Step 2: Sync primitives to Core API
            sync_response = await client.post(
                f"{AI_API_BASE_URL}/api/v1/primitives/sync",
                json={
                    "blueprintId": blueprint_id,
                    "syncOptions": {
                        "includeMasteryCriteria": True,
                        "includeQuestions": False
                    }
                },
                timeout=TEST_TIMEOUT
            )
            
            assert sync_response.status_code == 200
            sync_data = sync_response.json()
            
            assert sync_data["success"] == True
            assert sync_data["primitivesProcessed"] > 0
            assert sync_data["criteriaProcessed"] > 0
            
            # Step 3: Verify sync via Core API
            # Allow some time for async processing
            await asyncio.sleep(2)
            
            core_primitives_response = await client.get(
                f"{CORE_API_BASE_URL}/api/v1/blueprints/{blueprint_id}/primitives",
                timeout=TEST_TIMEOUT
            )
            
            if core_primitives_response.status_code == 200:
                core_primitives = core_primitives_response.json()
                assert len(core_primitives) == sync_data["primitivesProcessed"]
                
                # Verify primitive structure in Core API
                for primitive in core_primitives:
                    assert "primitiveId" in primitive
                    assert "primitiveType" in primitive
                    assert "difficultyLevel" in primitive
            
            print(f"✅ Primitive sync integration successful for blueprint: {blueprint_id}")

    @pytest.mark.asyncio
    async def test_question_generation_and_evaluation_integration(
        self,
        verify_services_running
    ):
        """
        Test question generation and evaluation workflow:
        1. AI API generates questions for a criterion
        2. Core API stores questions
        3. AI API evaluates user answers
        4. Core API updates progress tracking
        """
        async with httpx.AsyncClient(headers=DEFAULT_HEADERS) as client:
            # Mock criterion data for testing
            criterion_id = str(uuid.uuid4())
            primitive_id = str(uuid.uuid4())
            
            # Step 1: Generate questions via AI API
            question_gen_response = await client.post(
                f"{AI_API_BASE_URL}/api/v1/questions/criterion-specific",
                json={
                    "criterionId": criterion_id,
                    "primitiveId": primitive_id,
                    "criterionTitle": "Understand photosynthesis equation",
                    "criterionDescription": "Students should be able to write and explain the photosynthesis equation",
                    "ueeLevel": "UNDERSTAND",
                    "questionCount": 2,
                    "questionTypes": ["SHORT_ANSWER", "MULTIPLE_CHOICE"]
                },
                timeout=TEST_TIMEOUT
            )
            
            assert question_gen_response.status_code == 200
            questions_data = question_gen_response.json()
            
            assert "questions" in questions_data
            assert len(questions_data["questions"]) > 0
            
            question = questions_data["questions"][0]
            assert "questionId" in question
            assert "questionText" in question
            assert "questionType" in question
            
            # Step 2: Evaluate an answer via AI API
            evaluation_response = await client.post(
                f"{AI_API_BASE_URL}/api/v1/ai/evaluate-answer",
                json={
                    "questionContext": {
                        "questionId": question["questionId"],
                        "questionText": question["questionText"],
                        "expectedAnswer": question.get("expectedAnswer", "Light energy + CO2 + H2O → Glucose + Oxygen"),
                        "questionType": question["questionType"],
                        "marksAvailable": 5,
                        "markingCriteria": "Must mention key components and overall process"
                    },
                    "userAnswer": "Plants use sunlight to make glucose from carbon dioxide and water"
                },
                timeout=TEST_TIMEOUT
            )
            
            assert evaluation_response.status_code == 200
            evaluation_data = evaluation_response.json()
            
            assert "marksAchieved" in evaluation_data
            assert "marksAvailable" in evaluation_data
            assert "feedback" in evaluation_data
            assert evaluation_data["marksAvailable"] == 5
            assert 0 <= evaluation_data["marksAchieved"] <= 5
            
            print(f"✅ Question generation and evaluation integration successful")

    @pytest.mark.asyncio
    async def test_search_and_rag_integration(
        self,
        verify_services_running,
        sample_educational_content
    ):
        """
        Test search and RAG functionality integration:
        1. AI API indexes content
        2. Core API queries for relevant content
        3. AI API provides RAG-based responses
        """
        async with httpx.AsyncClient(headers=DEFAULT_HEADERS) as client:
            blueprint_id = str(uuid.uuid4())
            
            # Step 1: Index content via AI API
            index_response = await client.post(
                f"{AI_API_BASE_URL}/api/v1/index-blueprint",
                json={
                    "blueprintId": blueprint_id,
                    "blueprintData": {
                        "title": "Photosynthesis Study Guide",
                        "content": sample_educational_content,
                        "sections": [
                            {
                                "title": "Overview",
                                "content": sample_educational_content[:200]
                            }
                        ]
                    }
                },
                timeout=TEST_TIMEOUT
            )
            
            assert index_response.status_code == 200
            index_data = index_response.json()
            assert index_data["success"] == True
            
            # Allow time for indexing
            await asyncio.sleep(3)
            
            # Step 2: Search for content via AI API
            search_response = await client.post(
                f"{AI_API_BASE_URL}/api/v1/search",
                json={
                    "query": "What is photosynthesis?",
                    "blueprintId": blueprint_id,
                    "limit": 5
                },
                timeout=TEST_TIMEOUT
            )
            
            assert search_response.status_code == 200
            search_data = search_response.json()
            
            assert "results" in search_data
            if len(search_data["results"]) > 0:
                result = search_data["results"][0]
                assert "content" in result
                assert "score" in result
                assert result["score"] > 0
            
            # Step 3: Generate RAG response via AI API
            chat_response = await client.post(
                f"{AI_API_BASE_URL}/api/v1/chat/message",
                json={
                    "message": "Explain the process of photosynthesis",
                    "context": {
                        "blueprintId": blueprint_id,
                        "useRag": True
                    }
                },
                timeout=TEST_TIMEOUT
            )
            
            assert chat_response.status_code == 200
            chat_data = chat_response.json()
            
            assert "response" in chat_data
            assert len(chat_data["response"]) > 50  # Meaningful response
            
            print(f"✅ Search and RAG integration successful for blueprint: {blueprint_id}")


# Integration test markers
pytestmark = [pytest.mark.asyncio]
