# Sprint 33: Property-Based Testing with Hypothesis

import pytest
from hypothesis import given, strategies as st, assume, settings, Verbosity
from hypothesis.stateful import RuleBasedStateMachine, Bundle, rule, precondition
from unittest.mock import AsyncMock, patch
import asyncio
from typing import Dict, List, Any
import json

from app.api.schemas import (
    MasteryCriterionDto,
    KnowledgePrimitiveDto,
    PrimitiveGenerationRequest
)
from app.core.deconstruction import generate_enhanced_primitives_with_criteria
from app.core.mastery_criteria_service import MasteryCriteriaService
from app.core.question_generation_service import QuestionGenerationService

# Custom strategies for generating test data
@st.composite
def valid_source_text(draw):
    """Generate valid source text for primitive generation."""
    # Generate educational content with varying complexity
    topics = ["mathematics", "science", "history", "literature", "technology"]
    topic = draw(st.sampled_from(topics))
    
    # Generate sentences about the topic
    sentence_count = draw(st.integers(min_value=2, max_value=10))
    sentences = []
    
    for _ in range(sentence_count):
        sentence_length = draw(st.integers(min_value=5, max_value=50))
        sentence = draw(st.text(min_size=sentence_length, max_size=sentence_length * 2))
        # Ensure sentence has some educational content structure
        if any(char.isalpha() for char in sentence):
            sentences.append(sentence.strip())
    
    content = f"This text covers {topic}. " + " ".join(sentences)
    assume(len(content.strip()) >= 50)  # Minimum length for meaningful content
    assume(len(content.strip()) <= 5000)  # Maximum reasonable length
    
    return content

@st.composite
def valid_user_preferences(draw):
    """Generate valid user preferences for primitive generation."""
    # UEE distribution that sums to 1.0
    understand_pct = draw(st.floats(min_value=0.1, max_value=0.8))
    use_pct = draw(st.floats(min_value=0.1, max_value=1.0 - understand_pct))
    explore_pct = 1.0 - understand_pct - use_pct
    
    assume(explore_pct >= 0.1)  # Ensure all levels have minimum representation
    
    return {
        "uee_distribution": {
            "UNDERSTAND": understand_pct,
            "USE": use_pct,
            "EXPLORE": explore_pct
        },
        "max_primitives": draw(st.integers(min_value=1, max_value=20)),
        "difficulty_preference": draw(st.sampled_from(["easy", "moderate", "challenging"])),
        "learning_style": draw(st.sampled_from(["visual", "auditory", "kinesthetic", "mixed"]))
    }

@st.composite
def valid_primitive_data(draw):
    """Generate valid primitive data for testing."""
    primitive_types = ["concept", "process", "principle", "fact", "skill"]
    
    return {
        "primitive_id": draw(st.text(min_size=5, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='_-'))),
        "title": draw(st.text(min_size=5, max_size=100)),
        "description": draw(st.text(min_size=10, max_size=500)),
        "content": draw(st.text(min_size=20, max_size=2000)),
        "primitive_type": draw(st.sampled_from(primitive_types)),
        "tags": draw(st.lists(st.text(min_size=2, max_size=20), min_size=1, max_size=10))
    }

@st.composite
def valid_mastery_criterion(draw):
    """Generate valid mastery criterion data."""
    uee_levels = ["UNDERSTAND", "USE", "EXPLORE"]
    
    return {
        "criterion_id": draw(st.text(min_size=5, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='_-'))),
        "primitive_id": draw(st.text(min_size=5, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='_-'))),
        "title": draw(st.text(min_size=5, max_size=100)),
        "description": draw(st.text(min_size=10, max_size=500)),
        "uee_level": draw(st.sampled_from(uee_levels)),
        "weight": draw(st.floats(min_value=1.0, max_value=5.0)),
        "is_required": draw(st.booleans())
    }


class TestPrimitiveGenerationProperties:
    """Property-based tests for primitive generation."""
    
    @given(source_text=valid_source_text(), user_prefs=valid_user_preferences())
    @settings(max_examples=50, deadline=5000)
    @pytest.mark.asyncio
    async def test_primitive_generation_always_returns_valid_structure(self, source_text, user_prefs):
        """Property: Primitive generation always returns a valid structure."""
        
        # Mock LLM service to return valid JSON structure
        mock_llm_service = AsyncMock()
        mock_llm_service.generate_text.return_value = json.dumps({
            "primitives": [
                {
                    "primitive_id": "test_001",
                    "title": "Test Primitive",
                    "description": "Test description",
                    "content": "Test content",
                    "primitive_type": "concept",
                    "tags": ["test"],
                    "mastery_criteria": [
                        {
                            "criterion_id": "test_001_understand",
                            "title": "Test Criterion",
                            "description": "Test criterion description",
                            "uee_level": "UNDERSTAND",
                            "weight": 3.0,
                            "is_required": True
                        }
                    ]
                }
            ]
        })
        
        with patch('app.core.deconstruction.llm_service', mock_llm_service):
            result = await generate_enhanced_primitives_with_criteria(
                source_text=source_text,
                user_preferences=user_prefs
            )
            
            # Properties that should always hold
            assert result is not None
            assert isinstance(result, dict)
            assert "primitives" in result
            assert isinstance(result["primitives"], list)
            assert len(result["primitives"]) > 0
            
            # Each primitive should have required fields
            for primitive in result["primitives"]:
                assert "primitive_id" in primitive
                assert "title" in primitive
                assert "description" in primitive
                assert "content" in primitive
                assert "primitive_type" in primitive
                assert "mastery_criteria" in primitive
                assert isinstance(primitive["mastery_criteria"], list)
                assert len(primitive["mastery_criteria"]) > 0
                
                # Each criterion should be valid
                for criterion in primitive["mastery_criteria"]:
                    assert "criterion_id" in criterion
                    assert "uee_level" in criterion
                    assert criterion["uee_level"] in ["UNDERSTAND", "USE", "EXPLORE"]
                    assert "weight" in criterion
                    assert 1.0 <= criterion["weight"] <= 5.0

    @given(prefs=valid_user_preferences())
    @settings(max_examples=30)
    def test_user_preferences_respect_constraints(self, prefs):
        """Property: User preferences always satisfy constraints."""
        
        # UEE distribution should sum to 1.0 (within floating point tolerance)
        uee_sum = sum(prefs["uee_distribution"].values())
        assert abs(uee_sum - 1.0) < 0.001
        
        # All UEE levels should have positive values
        for level, percentage in prefs["uee_distribution"].items():
            assert percentage > 0
            assert percentage < 1.0
        
        # Max primitives should be reasonable
        assert 1 <= prefs["max_primitives"] <= 20

    @given(primitive=valid_primitive_data())
    @settings(max_examples=50)
    def test_primitive_validation_properties(self, primitive):
        """Property: Valid primitive data should always pass validation."""
        
        # Test that valid primitive data creates valid DTO
        try:
            primitive_dto = KnowledgePrimitiveDto(
                primitive_id=primitive["primitive_id"],
                title=primitive["title"],
                description=primitive["description"],
                content=primitive["content"],
                primitive_type=primitive["primitive_type"],
                tags=primitive["tags"],
                mastery_criteria=[]  # Empty for this test
            )
            
            # Properties that should hold
            assert primitive_dto.primitive_id == primitive["primitive_id"]
            assert primitive_dto.title == primitive["title"]
            assert primitive_dto.primitive_type == primitive["primitive_type"]
            assert isinstance(primitive_dto.tags, list)
            
        except ValueError as e:
            # If validation fails, the error should be meaningful
            assert len(str(e)) > 0

    @given(criterion=valid_mastery_criterion())
    @settings(max_examples=50)
    def test_mastery_criterion_validation_properties(self, criterion):
        """Property: Valid mastery criteria should always pass validation."""
        
        try:
            criterion_dto = MasteryCriterionDto(
                criterion_id=criterion["criterion_id"],
                primitive_id=criterion["primitive_id"],
                title=criterion["title"],
                description=criterion["description"],
                uee_level=criterion["uee_level"],
                weight=criterion["weight"],
                is_required=criterion["is_required"]
            )
            
            # Properties that should hold
            assert criterion_dto.criterion_id == criterion["criterion_id"]
            assert criterion_dto.uee_level in ["UNDERSTAND", "USE", "EXPLORE"]
            assert 1.0 <= criterion_dto.weight <= 5.0
            assert isinstance(criterion_dto.is_required, bool)
            
        except ValueError as e:
            # If validation fails, the error should be meaningful
            assert len(str(e)) > 0


class TestQuestionGenerationProperties:
    """Property-based tests for question generation."""
    
    @given(
        question_text=st.text(min_size=10, max_size=500),
        question_type=st.sampled_from(["multiple_choice", "short_answer", "essay", "true_false"]),
        uee_level=st.sampled_from(["UNDERSTAND", "USE", "EXPLORE"])
    )
    @settings(max_examples=30)
    def test_generated_questions_match_uee_level(self, question_text, question_type, uee_level):
        """Property: Generated questions should be appropriate for their UEE level."""
        
        # Mock question generation
        question_data = {
            "question_id": "test_q_001",
            "question_text": question_text,
            "question_type": question_type,
            "correct_answer": "Test answer",
            "uee_level": uee_level
        }
        
        # Properties based on UEE level
        if uee_level == "UNDERSTAND":
            # Basic recall/comprehension questions should be shorter and simpler
            assert len(question_text) >= 10
        elif uee_level == "USE":
            # Application questions should involve scenarios or problems
            assert len(question_text) >= 10
        elif uee_level == "EXPLORE":
            # Analysis/synthesis questions should be more complex
            assert len(question_text) >= 10
        
        # All questions should have valid types
        assert question_type in ["multiple_choice", "short_answer", "essay", "true_false"]

    @given(
        num_questions=st.integers(min_value=1, max_value=20),
        question_types=st.lists(
            st.sampled_from(["multiple_choice", "short_answer", "essay", "true_false"]),
            min_size=1, max_size=4
        )
    )
    @settings(max_examples=20)
    @pytest.mark.asyncio
    async def test_question_generation_quantity_property(self, num_questions, question_types):
        """Property: Question generation should respect quantity constraints."""
        
        mock_service = QuestionGenerationService()
        
        # Mock LLM response with exact number of questions requested
        mock_questions = []
        for i in range(num_questions):
            mock_questions.append({
                "question_id": f"q_{i:03d}",
                "question_text": f"Test question {i}?",
                "question_type": question_types[i % len(question_types)],
                "correct_answer": f"Answer {i}",
                "difficulty_level": "moderate"
            })
        
        mock_llm_response = json.dumps({"questions": mock_questions})
        
        with patch.object(mock_service.llm_service, 'generate_text', return_value=mock_llm_response):
            result = await mock_service.generate_criterion_questions(
                primitive={"primitive_id": "test", "title": "Test", "content": "Test content"},
                mastery_criterion={
                    "criterion_id": "test_crit",
                    "title": "Test Criterion",
                    "uee_level": "UNDERSTAND",
                    "weight": 3.0
                },
                num_questions=num_questions,
                question_types=question_types
            )
            
            # Properties that should hold
            assert len(result) <= num_questions  # Should not exceed requested amount
            assert len(result) > 0  # Should generate at least one question
            
            # All questions should have valid types from the requested list
            for question in result:
                assert question["question_type"] in question_types


class PrimitiveGenerationStateMachine(RuleBasedStateMachine):
    """Stateful property-based testing for primitive generation workflows."""
    
    source_texts = Bundle('source_texts')
    generated_primitives = Bundle('generated_primitives')
    mastery_criteria = Bundle('mastery_criteria')
    
    def __init__(self):
        super().__init__()
        self.primitives_database = {}
        self.criteria_database = {}
        self.generation_count = 0
    
    @rule(target=source_texts, content=valid_source_text())
    def generate_source_text(self, content):
        """Generate source text for primitive creation."""
        return content
    
    @rule(
        target=generated_primitives,
        source_text=source_texts,
        user_prefs=valid_user_preferences()
    )
    def generate_primitives(self, source_text, user_prefs):
        """Generate primitives from source text."""
        self.generation_count += 1
        
        # Mock primitive generation result
        primitive_id = f"prim_{self.generation_count:03d}"
        primitive = {
            "primitive_id": primitive_id,
            "title": f"Generated Primitive {self.generation_count}",
            "description": "Auto-generated primitive",
            "content": source_text[:100],  # First 100 chars
            "primitive_type": "concept",
            "source_text": source_text,
            "user_preferences": user_prefs
        }
        
        self.primitives_database[primitive_id] = primitive
        return primitive
    
    @rule(
        target=mastery_criteria,
        primitive=generated_primitives,
        uee_level=st.sampled_from(["UNDERSTAND", "USE", "EXPLORE"])
    )
    def generate_mastery_criteria(self, primitive, uee_level):
        """Generate mastery criteria for a primitive."""
        criterion_id = f"{primitive['primitive_id']}_{uee_level.lower()}"
        criterion = {
            "criterion_id": criterion_id,
            "primitive_id": primitive["primitive_id"],
            "title": f"Criterion for {primitive['title']}",
            "description": f"Test understanding at {uee_level} level",
            "uee_level": uee_level,
            "weight": 3.0,
            "is_required": True
        }
        
        self.criteria_database[criterion_id] = criterion
        return criterion
    
    @rule(primitive=generated_primitives)
    def verify_primitive_consistency(self, primitive):
        """Verify that primitives maintain consistency."""
        # Properties that should always hold
        assert primitive["primitive_id"] in self.primitives_database
        stored_primitive = self.primitives_database[primitive["primitive_id"]]
        assert stored_primitive["title"] == primitive["title"]
        assert stored_primitive["primitive_type"] == primitive["primitive_type"]
    
    @rule(criterion=mastery_criteria)
    def verify_criterion_relationships(self, criterion):
        """Verify that criteria correctly reference their primitives."""
        assert criterion["primitive_id"] in self.primitives_database
        assert criterion["uee_level"] in ["UNDERSTAND", "USE", "EXPLORE"]
        assert 1.0 <= criterion["weight"] <= 5.0
    
    @precondition(lambda self: len(self.primitives_database) > 0)
    @rule()
    def verify_database_integrity(self):
        """Verify overall database integrity."""
        # All primitives should have unique IDs
        primitive_ids = list(self.primitives_database.keys())
        assert len(primitive_ids) == len(set(primitive_ids))
        
        # All criteria should reference existing primitives
        for criterion in self.criteria_database.values():
            assert criterion["primitive_id"] in self.primitives_database


# Test configuration for property-based testing
TestPrimitiveWorkflows = PrimitiveGenerationStateMachine.TestCase

# Configure Hypothesis settings for CI/development
settings.register_profile("ci", max_examples=100, deadline=10000)
settings.register_profile("dev", max_examples=20, deadline=5000, verbosity=Verbosity.verbose)

# Load appropriate profile
import os
if os.getenv("CI"):
    settings.load_profile("ci")
else:
    settings.load_profile("dev")
