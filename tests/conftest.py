"""
Test configuration and fixtures for the Elevate AI API tests.
"""

import pytest
from unittest.mock import AsyncMock, patch
from typing import Dict, Any


@pytest.fixture
def mock_blueprint_data() -> Dict[str, Any]:
    """Mock blueprint data for testing."""
    return {
        "source_id": "test-blueprint",
        "source_text": "Photosynthesis is the process by which plants convert sunlight into energy. This process occurs in the chloroplasts and involves several key steps including light absorption, electron transport, and carbon fixation.",
        "source_title": "Test Photosynthesis Blueprint",
        "source_type": "article",
        "source_summary": {
            "core_thesis_or_main_argument": "Photosynthesis is a fundamental biological process that converts light energy into chemical energy",
            "inferred_purpose": "Educational content about plant biology and energy conversion"
        },
        "sections": [
            {
                "section_id": "sec_1",
                "section_name": "Introduction to Photosynthesis",
                "description": "Basic overview of the photosynthesis process"
            },
            {
                "section_id": "sec_2", 
                "section_name": "Key Components",
                "description": "Important elements involved in photosynthesis"
            }
        ],
        "knowledge_primitives": {
            "key_propositions_and_facts": [
                {
                    "id": "prop_1",
                    "statement": "Photosynthesis converts sunlight into chemical energy",
                    "supporting_evidence": ["Light absorption", "Energy conversion"],
                    "sections": ["sec_1"]
                },
                {
                    "id": "prop_2",
                    "statement": "Chloroplasts are the organelles where photosynthesis occurs",
                    "supporting_evidence": ["Organelle structure", "Chlorophyll presence"],
                    "sections": ["sec_1", "sec_2"]
                }
            ],
            "key_entities_and_definitions": [
                {
                    "id": "entity_1",
                    "entity": "Photosynthesis",
                    "definition": "The process by which plants convert light energy into chemical energy",
                    "category": "Process",
                    "sections": ["sec_1"]
                },
                {
                    "id": "entity_2",
                    "entity": "Chloroplasts",
                    "definition": "Organelles in plant cells where photosynthesis occurs",
                    "category": "Organelle",
                    "sections": ["sec_1", "sec_2"]
                }
            ],
            "described_processes_and_steps": [
                {
                    "id": "process_1",
                    "process_name": "Light-dependent reactions",
                    "description": "The first stage of photosynthesis that requires light",
                    "steps": ["Light absorption", "Electron transport", "ATP synthesis"],
                    "sections": ["sec_1"]
                }
            ],
            "identified_relationships": [
                {
                    "id": "rel_1",
                    "relationship_type": "Location",
                    "entity_a": "Photosynthesis",
                    "entity_b": "Chloroplasts",
                    "description": "Photosynthesis occurs within chloroplasts",
                    "sections": ["sec_1", "sec_2"]
                }
            ],
            "implicit_and_open_questions": []
        }
    }


@pytest.fixture
def mock_llm_response() -> str:
    """Mock LLM response for question generation."""
    return '''[
        {
            "text": "What is the primary function of photosynthesis?",
            "answer": "Photosynthesis converts sunlight into chemical energy that plants can use for growth and development.",
            "question_type": "understand",
            "total_marks_available": 2,
            "marking_criteria": "Award 1 mark for mentioning 'sunlight' and 1 mark for 'chemical energy'."
        },
        {
            "text": "Where does photosynthesis occur in plant cells?",
            "answer": "Photosynthesis occurs in the chloroplasts, which are specialized organelles found in plant cells.",
            "question_type": "recall",
            "total_marks_available": 1,
            "marking_criteria": "Award 1 mark for correctly identifying 'chloroplasts'."
        },
        {
            "text": "Explain the relationship between chloroplasts and photosynthesis.",
            "answer": "Chloroplasts are the organelles where photosynthesis takes place. They contain chlorophyll and other pigments that absorb light energy, and have the necessary enzymes and structures to carry out the complex biochemical reactions of photosynthesis.",
            "question_type": "explore",
            "total_marks_available": 3,
            "marking_criteria": "Award 1 mark for identifying chloroplasts as the location, 1 mark for mentioning chlorophyll/pigments, and 1 mark for mentioning enzymes/biochemical reactions."
        }
    ]'''


@pytest.fixture
def mock_llm_service():
    """Mock LLM service for testing."""
    with patch('app.core.llm_service.llm_service') as mock_service:
        mock_service.call_llm = AsyncMock()
        yield mock_service


@pytest.fixture
def mock_blueprint_retrieval():
    """Mock blueprint data retrieval for testing."""
    with patch('app.core.indexing._get_blueprint_data', new_callable=AsyncMock) as mock_retrieval:
        yield mock_retrieval


@pytest.fixture
def mock_question_data() -> Dict[str, Any]:
    """Mock question data for testing - legacy format."""
    return {
        "id": 1,
        "text": "What is the primary function of mitochondria?",
        "answer": "Mitochondria are the powerhouse of the cell, generating ATP through cellular respiration.",
        "question_type": "understand",
        "total_marks_available": 5,
        "marking_criteria": "Award 1 mark for mentioning 'powerhouse', 1 mark for 'ATP', 1 mark for 'cellular respiration', 1 mark for energy generation, and 1 mark for clarity.",
        "question_set_name": "Sample Question Set",
        "folder_name": "Biology",
        "blueprint_title": "Cellular Biology"
    }


@pytest.fixture
def mock_question_context_data() -> Dict[str, Any]:
    """Mock question context data matching QuestionContextDto schema."""
    return {
        "questionId": 1,
        "questionText": "What is the primary function of mitochondria?",
        "expectedAnswer": "Mitochondria are the powerhouse of the cell, generating ATP through cellular respiration.",
        "questionType": "SHORT_ANSWER",
        "marksAvailable": 5,
        "markingCriteria": "Award 1 mark for mentioning 'powerhouse', 1 mark for 'ATP', 1 mark for 'cellular respiration', 1 mark for energy generation, and 1 mark for clarity."
    }


@pytest.fixture
def mock_evaluation_response() -> str:
    """Mock LLM response for answer evaluation."""
    return '''{
        "marks_achieved": 4,
        "corrected_answer": "Mitochondria are the powerhouse of the cell, generating ATP through cellular respiration.",
        "feedback": "Good answer! You correctly identified mitochondria as the powerhouse and mentioned ATP generation. You could have been more specific about cellular respiration."
    }'''


@pytest.fixture
def mock_question_retrieval():
    """Mock question data retrieval for testing."""
    with patch('app.core.indexing._get_question_data') as mock_retrieval:
        yield mock_retrieval


@pytest.fixture
def mock_evaluation_service():
    """Mock answer evaluation service for testing."""
    with patch('app.api.endpoints._call_ai_service_for_evaluation') as mock_service:
        yield mock_service 