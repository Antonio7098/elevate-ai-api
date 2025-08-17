"""
Tests for the Generation Orchestrator

Tests the sequential generation workflow:
source → blueprint → sections → primitives → mastery criteria → questions
with user editing capabilities between each step.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from app.core.generation_orchestrator import (
    GenerationOrchestrator,
    GenerationStep,
    GenerationStatus,
    UserEditRequest,
    GenerationProgress
)


class TestGenerationOrchestrator:
    """Test cases for the GenerationOrchestrator class."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create a fresh orchestrator instance for each test."""
        return GenerationOrchestrator()
    
    @pytest.fixture
    def sample_source_content(self):
        """Sample source content for testing."""
        return """
        Photosynthesis is the process by which plants convert light energy into chemical energy.
        This process occurs in the chloroplasts and involves two main stages: light reactions
        and the Calvin cycle. Chlorophyll, the green pigment in plants, plays a crucial role
        in capturing light energy.
        """
    
    @pytest.fixture
    def sample_user_preferences(self):
        """Sample user preferences for testing."""
        return {
            "learning_style": "visual",
            "difficulty": "intermediate",
            "focus_areas": ["biology", "chemistry"]
        }
    
    @pytest.fixture
    def sample_session_id(self):
        """Sample session ID for testing."""
        return "test_session_001"
    
    def test_orchestrator_initialization(self, orchestrator):
        """Test that orchestrator initializes correctly."""
        assert orchestrator.generation_sessions == {}
    
    @pytest.mark.asyncio
    async def test_start_generation_session_success(self, orchestrator, sample_source_content, 
                                                   sample_user_preferences, sample_session_id):
        """Test successful start of generation session."""
        # Start session
        progress = await orchestrator.start_generation_session(
            session_id=sample_session_id,
            source_content=sample_source_content,
            source_type="textbook_chapter",
            user_preferences=sample_user_preferences
        )
        
        # Verify session was created
        assert sample_session_id in orchestrator.generation_sessions
        assert orchestrator.generation_sessions[sample_session_id] == progress
        
        # Verify initial state
        assert progress.current_step == GenerationStep.SOURCE_ANALYSIS
        assert progress.status == GenerationStatus.READY_FOR_NEXT
        assert len(progress.completed_steps) == 1
        assert GenerationStep.SOURCE_ANALYSIS in progress.completed_steps
        
        # Verify content was stored
        assert progress.current_content is not None
        assert progress.current_content["source_content"] == sample_source_content
        assert progress.current_content["source_type"] == "textbook_chapter"
        assert progress.current_content["user_preferences"] == sample_user_preferences
    
    @pytest.mark.asyncio
    async def test_start_generation_session_without_preferences(self, orchestrator, 
                                                              sample_source_content, sample_session_id):
        """Test starting session without user preferences."""
        progress = await orchestrator.start_generation_session(
            session_id=sample_session_id,
            source_content=sample_source_content,
            source_type="textbook_chapter"
        )
        
        # Verify default preferences
        assert progress.current_content["user_preferences"] == {}
    
    @pytest.mark.asyncio
    async def test_proceed_to_next_step_success(self, orchestrator, sample_session_id):
        """Test successful progression to next step."""
        # Setup: Create a session that's ready for next step
        orchestrator.generation_sessions[sample_session_id] = GenerationProgress(
            session_id=sample_session_id,
            current_step=GenerationStep.SOURCE_ANALYSIS,
            status=GenerationStatus.READY_FOR_NEXT,
            completed_steps=[GenerationStep.SOURCE_ANALYSIS],
            current_content={"source_content": "test", "source_type": "test", "user_preferences": {}},
            user_edits=[],
            errors=[],
            started_at=datetime.utcnow(),
            last_updated=datetime.utcnow()
        )
        
        # Proceed to next step
        progress = await orchestrator.proceed_to_next_step(sample_session_id)
        
        # Verify progression
        assert progress.current_step == GenerationStep.BLUEPRINT_CREATION
        assert progress.status == GenerationStatus.READY_FOR_NEXT
        assert len(progress.completed_steps) == 2
        assert GenerationStep.BLUEPRINT_CREATION in progress.completed_steps
    
    @pytest.mark.asyncio
    async def test_proceed_to_next_step_session_not_found(self, orchestrator):
        """Test error when proceeding with non-existent session."""
        with pytest.raises(ValueError, match="Session nonexistent not found"):
            await orchestrator.proceed_to_next_step("nonexistent")
    
    @pytest.mark.asyncio
    async def test_proceed_to_next_step_not_ready(self, orchestrator, sample_session_id):
        """Test error when proceeding with session not ready for next step."""
        # Setup: Create a session that's not ready
        orchestrator.generation_sessions[sample_session_id] = GenerationProgress(
            session_id=sample_session_id,
            current_step=GenerationStep.SOURCE_ANALYSIS,
            status=GenerationStatus.IN_PROGRESS,
            completed_steps=[],
            current_content={"source_content": "test", "source_type": "test", "user_preferences": {}},
            user_edits=[],
            errors=[],
            started_at=datetime.utcnow(),
            last_updated=datetime.utcnow()
        )
        
        with pytest.raises(ValueError, match="Session test_session_001 not ready for next step"):
            await orchestrator.proceed_to_next_step(sample_session_id)
    
    @pytest.mark.asyncio
    async def test_proceed_to_final_step(self, orchestrator, sample_session_id):
        """Test progression to final step (complete)."""
        # Setup: Create a session at the last step
        orchestrator.generation_sessions[sample_session_id] = GenerationProgress(
            session_id=sample_session_id,
            current_step=GenerationStep.NOTE_GENERATION,
            status=GenerationStatus.READY_FOR_NEXT,
            completed_steps=[
                GenerationStep.SOURCE_ANALYSIS,
                GenerationStep.BLUEPRINT_CREATION,
                GenerationStep.SECTION_GENERATION,
                GenerationStep.PRIMITIVE_EXTRACTION,
                GenerationStep.MASTERY_CRITERIA,
                GenerationStep.QUESTION_GENERATION,
                GenerationStep.NOTE_GENERATION
            ],
            current_content={"source_content": "test", "source_type": "test", "user_preferences": {}},
            user_edits=[],
            errors=[],
            started_at=datetime.utcnow(),
            last_updated=datetime.utcnow()
        )
        
        # Proceed to next step (should be complete)
        progress = await orchestrator.proceed_to_next_step(sample_session_id)
        
        # Verify completion
        assert progress.current_step == GenerationStep.COMPLETE
        assert progress.status == GenerationStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_user_edit_content_success(self, orchestrator, sample_session_id):
        """Test successful user content editing."""
        # Setup: Create a session at section generation step
        orchestrator.generation_sessions[sample_session_id] = GenerationProgress(
            session_id=sample_session_id,
            current_step=GenerationStep.SECTION_GENERATION,
            status=GenerationStatus.READY_FOR_NEXT,
            completed_steps=[GenerationStep.SOURCE_ANALYSIS, GenerationStep.BLUEPRINT_CREATION],
            current_content={
                "source_content": "test",
                "source_type": "test",
                "user_preferences": {},
                "sections": [
                    {"section_id": "sec_001", "title": "Original Title"}
                ]
            },
            user_edits=[],
            errors=[],
            started_at=datetime.utcnow(),
            last_updated=datetime.utcnow()
        )
        
        # Create edit request
        edit_request = UserEditRequest(
            step=GenerationStep.SECTION_GENERATION,
            content_id="sec_001",
            edited_content={"title": "Updated Title"},
            user_notes="Made title more descriptive"
        )
        
        # Apply edit
        progress = await orchestrator.user_edit_content(sample_session_id, edit_request)
        
        # Verify edit was applied
        assert len(progress.user_edits) == 1
        assert progress.user_edits[0].step == GenerationStep.SECTION_GENERATION
        assert progress.user_edits[0].user_notes == "Made title more descriptive"
        assert progress.status == GenerationStatus.READY_FOR_NEXT
        
        # Verify content was updated
        updated_section = progress.current_content["sections"][0]
        assert updated_section["title"] == "Updated Title"
    
    @pytest.mark.asyncio
    async def test_user_edit_content_wrong_step(self, orchestrator, sample_session_id):
        """Test error when editing content for wrong step."""
        # Setup: Create a session at section generation step
        orchestrator.generation_sessions[sample_session_id] = GenerationProgress(
            session_id=sample_session_id,
            current_step=GenerationStep.SECTION_GENERATION,
            status=GenerationStatus.READY_FOR_NEXT,
            completed_steps=[GenerationStep.SOURCE_ANALYSIS, GenerationStep.BLUEPRINT_CREATION],
            current_content={"source_content": "test", "source_type": "test", "user_preferences": {}},
            user_edits=[],
            errors=[],
            started_at=datetime.utcnow(),
            last_updated=datetime.utcnow()
        )
        
        # Try to edit blueprint content while at section generation step
        edit_request = UserEditRequest(
            step=GenerationStep.BLUEPRINT_CREATION,
            content_id="bp_001",
            edited_content={"title": "Updated Title"},
            user_notes="Test edit"
        )
        
        with pytest.raises(ValueError, match="Can only edit content for current step: section_generation"):
            await orchestrator.user_edit_content(sample_session_id, edit_request)
    
    @pytest.mark.asyncio
    async def test_user_edit_content_session_not_found(self, orchestrator):
        """Test error when editing content for non-existent session."""
        edit_request = UserEditRequest(
            step=GenerationStep.SECTION_GENERATION,
            content_id="sec_001",
            edited_content={"title": "Updated Title"},
            user_notes="Test edit"
        )
        
        with pytest.raises(ValueError, match="Session nonexistent not found"):
            await orchestrator.user_edit_content("nonexistent", edit_request)
    
    @pytest.mark.asyncio
    async def test_get_generation_progress_success(self, orchestrator, sample_session_id):
        """Test successful retrieval of generation progress."""
        # Setup: Create a session
        orchestrator.generation_sessions[sample_session_id] = GenerationProgress(
            session_id=sample_session_id,
            current_step=GenerationStep.BLUEPRINT_CREATION,
            status=GenerationStatus.READY_FOR_NEXT,
            completed_steps=[GenerationStep.SOURCE_ANALYSIS],
            current_content={"source_content": "test", "source_type": "test", "user_preferences": {}},
            user_edits=[],
            errors=[],
            started_at=datetime.utcnow(),
            last_updated=datetime.utcnow()
        )
        
        # Get progress
        progress = await orchestrator.get_generation_progress(sample_session_id)
        
        # Verify progress
        assert progress.session_id == sample_session_id
        assert progress.current_step == GenerationStep.BLUEPRINT_CREATION
        assert progress.status == GenerationStatus.READY_FOR_NEXT
    
    @pytest.mark.asyncio
    async def test_get_generation_progress_session_not_found(self, orchestrator):
        """Test error when getting progress for non-existent session."""
        with pytest.raises(ValueError, match="Session nonexistent not found"):
            await orchestrator.get_generation_progress("nonexistent")
    
    @pytest.mark.asyncio
    async def test_get_complete_learning_path_success(self, orchestrator, sample_session_id):
        """Test successful retrieval of complete learning path."""
        # Setup: Create a completed session
        orchestrator.generation_sessions[sample_session_id] = GenerationProgress(
            session_id=sample_session_id,
            current_step=GenerationStep.COMPLETE,
            status=GenerationStatus.COMPLETED,
            completed_steps=[
                GenerationStep.SOURCE_ANALYSIS,
                GenerationStep.BLUEPRINT_CREATION,
                GenerationStep.SECTION_GENERATION,
                GenerationStep.PRIMITIVE_EXTRACTION,
                GenerationStep.MASTERY_CRITERIA,
                GenerationStep.QUESTION_GENERATION,
                GenerationStep.NOTE_GENERATION
            ],
            current_content={
                "source_content": "test",
                "source_type": "test",
                "user_preferences": {},
                "blueprint": {"title": "Test Blueprint"},
                "sections": [{"title": "Test Section"}],
                "primitives": [{"title": "Test Primitive"}],
                "mastery_criteria": [{"title": "Test Criterion"}],
                "questions": [{"title": "Test Question"}],
                "notes": {"title": "Test Notes"}
            },
            user_edits=[],
            errors=[],
            started_at=datetime.utcnow(),
            last_updated=datetime.utcnow()
        )
        
        # Get complete learning path
        complete_path = await orchestrator.get_complete_learning_path(sample_session_id)
        
        # Verify complete path
        assert complete_path["session_id"] == sample_session_id
        assert complete_path["status"] == "complete"
        assert "content" in complete_path
        assert "user_edits" in complete_path
    
    @pytest.mark.asyncio
    async def test_get_complete_learning_path_not_complete(self, orchestrator, sample_session_id):
        """Test error when getting complete path for incomplete session."""
        # Setup: Create an incomplete session
        orchestrator.generation_sessions[sample_session_id] = GenerationProgress(
            session_id=sample_session_id,
            current_step=GenerationStep.BLUEPRINT_CREATION,
            status=GenerationStatus.READY_FOR_NEXT,
            completed_steps=[GenerationStep.SOURCE_ANALYSIS],
            current_content={"source_content": "test", "source_type": "test", "user_preferences": {}},
            user_edits=[],
            errors=[],
            started_at=datetime.utcnow(),
            last_updated=datetime.utcnow()
        )
        
        with pytest.raises(ValueError, match="Session test_session_001 not complete"):
            await orchestrator.get_complete_learning_path(sample_session_id)
    
    def test_get_next_step_success(self, orchestrator):
        """Test successful retrieval of next step."""
        next_step = orchestrator._get_next_step(GenerationStep.SOURCE_ANALYSIS)
        assert next_step == GenerationStep.BLUEPRINT_CREATION
        
        next_step = orchestrator._get_next_step(GenerationStep.BLUEPRINT_CREATION)
        assert next_step == GenerationStep.SECTION_GENERATION
    
    def test_get_next_step_final_step(self, orchestrator):
        """Test that final step returns None."""
        next_step = orchestrator._get_next_step(GenerationStep.COMPLETE)
        assert next_step is None
    
    def test_get_next_step_invalid_step(self, orchestrator):
        """Test that invalid step returns None."""
        next_step = orchestrator._get_next_step("invalid_step")
        assert next_step is None
    
    @pytest.mark.asyncio
    async def test_complete_workflow_simulation(self, orchestrator, sample_source_content, 
                                              sample_user_preferences):
        """Test complete workflow from start to finish."""
        session_id = "workflow_test_session"
        
        # Step 1: Start session
        progress = await orchestrator.start_generation_session(
            session_id=session_id,
            source_content=sample_source_content,
            source_type="textbook_chapter",
            user_preferences=sample_user_preferences
        )
        
        assert progress.current_step == GenerationStep.SOURCE_ANALYSIS
        assert progress.status == GenerationStatus.READY_FOR_NEXT
        
        # Step 2: Proceed through all steps
        steps_to_complete = [
            GenerationStep.BLUEPRINT_CREATION,
            GenerationStep.SECTION_GENERATION,
            GenerationStep.PRIMITIVE_EXTRACTION,
            GenerationStep.MASTERY_CRITERIA,
            GenerationStep.QUESTION_GENERATION,
            GenerationStep.NOTE_GENERATION
        ]
        
        for step in steps_to_complete:
            # Proceed to next step
            progress = await orchestrator.proceed_to_next_step(session_id)
            
            # Verify step was completed
            assert step in progress.completed_steps
            assert progress.status == GenerationStatus.READY_FOR_NEXT
        
        # Final step should be complete
        progress = await orchestrator.proceed_to_next_step(session_id)
        assert progress.current_step == GenerationStep.COMPLETE
        assert progress.status == GenerationStatus.COMPLETED
        
        # Verify all steps were completed
        expected_steps = [GenerationStep.SOURCE_ANALYSIS] + steps_to_complete
        assert len(progress.completed_steps) == len(expected_steps)
        for step in expected_steps:
            assert step in progress.completed_steps
        
        # Get complete learning path
        complete_path = await orchestrator.get_complete_learning_path(session_id)
        assert complete_path["status"] == "complete"
        assert "content" in complete_path
    
    @pytest.mark.asyncio
    async def test_workflow_with_user_edits(self, orchestrator, sample_source_content, 
                                          sample_user_preferences):
        """Test workflow with user edits at multiple steps."""
        session_id = "edit_test_session"
        
        # Start session
        await orchestrator.start_generation_session(
            session_id=session_id,
            source_content=sample_source_content,
            source_type="textbook_chapter",
            user_preferences=sample_user_preferences
        )
        
        # Proceed to blueprint creation
        progress = await orchestrator.proceed_to_next_step(session_id)
        assert progress.current_step == GenerationStep.BLUEPRINT_CREATION
        
        # Edit blueprint
        edit_request = UserEditRequest(
            step=GenerationStep.BLUEPRINT_CREATION,
            content_id="blueprint",
            edited_content={"title": "Updated Blueprint Title"},
            user_notes="Improved blueprint title"
        )
        
        progress = await orchestrator.user_edit_content(session_id, edit_request)
        assert len(progress.user_edits) == 1
        assert progress.user_edits[0].user_notes == "Improved blueprint title"
        
        # Proceed to section generation
        progress = await orchestrator.proceed_to_next_step(session_id)
        assert progress.current_step == GenerationStep.SECTION_GENERATION
        
        # Edit sections
        edit_request = UserEditRequest(
            step=GenerationStep.SECTION_GENERATION,
            content_id="sec_001",
            edited_content={"title": "Updated Section Title"},
            user_notes="Improved section title"
        )
        
        progress = await orchestrator.user_edit_content(session_id, edit_request)
        assert len(progress.user_edits) == 2
        
        # Continue workflow
        progress = await orchestrator.proceed_to_next_step(session_id)
        assert progress.current_step == GenerationStep.PRIMITIVE_EXTRACTION
        
        # Verify edits were preserved
        assert len(progress.user_edits) == 2
        assert progress.user_edits[0].user_notes == "Improved blueprint title"
        assert progress.user_edits[1].user_notes == "Improved section title"


class TestUserEditRequest:
    """Test cases for the UserEditRequest model."""
    
    def test_user_edit_request_creation(self):
        """Test UserEditRequest model creation."""
        edit_request = UserEditRequest(
            step=GenerationStep.SECTION_GENERATION,
            content_id="sec_001",
            edited_content={"title": "Updated Title"},
            user_notes="Test edit"
        )
        
        assert edit_request.step == GenerationStep.SECTION_GENERATION
        assert edit_request.content_id == "sec_001"
        assert edit_request.edited_content == {"title": "Updated Title"}
        assert edit_request.user_notes == "Test edit"
    
    def test_user_edit_request_without_notes(self):
        """Test UserEditRequest creation without user notes."""
        edit_request = UserEditRequest(
            step=GenerationStep.SECTION_GENERATION,
            content_id="sec_001",
            edited_content={"title": "Updated Title"}
        )
        
        assert edit_request.user_notes is None


class TestGenerationProgress:
    """Test cases for the GenerationProgress model."""
    
    def test_generation_progress_creation(self):
        """Test GenerationProgress model creation."""
        now = datetime.utcnow()
        progress = GenerationProgress(
            session_id="test_session",
            current_step=GenerationStep.SOURCE_ANALYSIS,
            status=GenerationStatus.IN_PROGRESS,
            completed_steps=[],
            current_content={"test": "content"},
            user_edits=[],
            errors=[],
            started_at=now,
            last_updated=now
        )
        
        assert progress.current_step == GenerationStep.SOURCE_ANALYSIS
        assert progress.status == GenerationStatus.IN_PROGRESS
        assert progress.current_content == {"test": "content"}
        assert progress.started_at == now
        assert progress.last_updated == now


if __name__ == "__main__":
    pytest.main([__file__])
