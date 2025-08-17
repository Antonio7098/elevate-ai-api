"""
Tests for the Orchestrator API Endpoints

Tests the REST API endpoints for the sequential generation workflow:
- Start generation session
- Proceed to next step
- Edit generated content
- Get generation progress
- Get complete learning path
- Delete session
"""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import HTTPException

from app.api.orchestrator_endpoints import (
    router,
    start_generation_session,
    proceed_to_next_step,
    edit_generated_content,
    get_generation_progress,
    get_complete_learning_path,
    delete_generation_session,
    _get_next_actions
)
from app.core.generation_orchestrator import (
    GenerationStep,
    GenerationStatus,
    UserEditRequest
)


class TestOrchestratorEndpoints:
    """Test cases for the orchestrator API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the API."""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)
    
    @pytest.fixture
    def sample_start_request(self):
        """Sample request data for starting a generation session."""
        return {
            "source_content": "Test source content for photosynthesis",
            "source_type": "textbook_chapter",
            "user_preferences": {
                "learning_style": "visual",
                "difficulty": "intermediate"
            },
            "session_title": "Test Session"
        }
    
    @pytest.fixture
    def sample_edit_request(self):
        """Sample request data for editing content."""
        return {
            "step": "section_generation",
            "content_id": "sec_001",
            "edited_content": {
                "title": "Updated Section Title"
            },
            "user_notes": "Made title more descriptive"
        }
    
    @pytest.fixture
    def mock_progress(self):
        """Mock generation progress for testing."""
        return Mock(
            session_id="test_session_001",
            current_step=GenerationStep.SECTION_GENERATION,
            status=GenerationStatus.READY_FOR_NEXT,
            completed_steps=[GenerationStep.SOURCE_ANALYSIS, GenerationStep.BLUEPRINT_CREATION],
            current_content={
                "source_content": "Test content",
                "source_type": "textbook_chapter",
                "sections": [
                    {"section_id": "sec_001", "title": "Test Section"}
                ]
            },
            user_edits=[],
            errors=[],
            started_at=Mock(isoformat=lambda: "2024-01-01T00:00:00"),
            last_updated=Mock(isoformat=lambda: "2024-01-01T00:00:00")
        )
    
    @pytest.mark.asyncio
    async def test_start_generation_session_success(self, sample_start_request):
        """Test successful start of generation session."""
        # Mock the orchestrator
        mock_orchestrator = Mock()
        mock_orchestrator.start_generation_session = AsyncMock(return_value=Mock(
            current_step=GenerationStep.SOURCE_ANALYSIS,
            status=GenerationStatus.READY_FOR_NEXT
        ))
        
        with patch('app.api.orchestrator_endpoints.generation_orchestrator', mock_orchestrator):
            # Mock get_current_user
            with patch('app.api.orchestrator_endpoints.get_current_user', return_value={"id": "test_user"}):
                # Mock uuid generation
                with patch('app.api.orchestrator_endpoints.uuid.uuid4', return_value="test_session_001"):
                    response = await start_generation_session(
                        request=Mock(**sample_start_request),
                        current_user={"id": "test_user"}
                    )
        
        # Verify response
        assert response.session_id == "test_session_001"
        assert response.current_step == GenerationStep.SOURCE_ANALYSIS
        assert response.status == GenerationStatus.READY_FOR_NEXT
        assert "Generation session started successfully" in response.message
        assert "Proceed to next step" in response.next_actions
    
    @pytest.mark.asyncio
    async def test_start_generation_session_failure(self, sample_start_request):
        """Test failure when starting generation session."""
        # Mock the orchestrator to raise an exception
        mock_orchestrator = Mock()
        mock_orchestrator.start_generation_session = AsyncMock(side_effect=Exception("Test error"))
        
        with patch('app.api.orchestrator_endpoints.generation_orchestrator', mock_orchestrator):
            with patch('app.api.orchestrator_endpoints.get_current_user', return_value={"id": "test_user"}):
                with patch('app.api.orchestrator_endpoints.uuid.uuid4', return_value="test_session_001"):
                    with pytest.raises(HTTPException) as exc_info:
                        await start_generation_session(
                            request=Mock(**sample_start_request),
                            current_user={"id": "test_user"}
                        )
        
        assert exc_info.value.status_code == 500
        assert "Failed to start generation session" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_proceed_to_next_step_success(self):
        """Test successful progression to next step."""
        # Mock the orchestrator
        mock_orchestrator = Mock()
        mock_orchestrator.proceed_to_next_step = AsyncMock(return_value=Mock(
            session_id="test_session_001",
            current_step=GenerationStep.BLUEPRINT_CREATION,
            status=GenerationStatus.READY_FOR_NEXT,
            completed_steps=[GenerationStep.SOURCE_ANALYSIS],
            current_content={"test": "content"},
            user_edits=[],
            errors=[],
            started_at=Mock(isoformat=lambda: "2024-01-01T00:00:00"),
            last_updated=Mock(isoformat=lambda: "2024-01-01T00:00:00")
        ))
        
        with patch('app.api.orchestrator_endpoints.generation_orchestrator', mock_orchestrator):
            with patch('app.api.orchestrator_endpoints.get_current_user', return_value={"id": "test_user"}):
                response = await proceed_to_next_step(
                    request=Mock(session_id="test_session_001"),
                    current_user={"id": "test_user"}
                )
        
        # Verify response
        assert response.session_id == "test_session_001"
        assert response.current_step == GenerationStep.BLUEPRINT_CREATION
        assert response.status == GenerationStatus.READY_FOR_NEXT
        assert "Proceed to next step" in response.next_actions
    
    @pytest.mark.asyncio
    async def test_proceed_to_next_step_validation_error(self):
        """Test validation error when proceeding to next step."""
        # Mock the orchestrator to raise a ValueError
        mock_orchestrator = Mock()
        mock_orchestrator.proceed_to_next_step = AsyncMock(side_effect=ValueError("Session not found"))
        
        with patch('app.api.orchestrator_endpoints.generation_orchestrator', mock_orchestrator):
            with patch('app.api.orchestrator_endpoints.get_current_user', return_value={"id": "test_user"}):
                with pytest.raises(HTTPException) as exc_info:
                    await proceed_to_next_step(
                        request=Mock(session_id="test_session_001"),
                        current_user={"id": "test_user"}
                    )
        
        assert exc_info.value.status_code == 400
        assert "Session not found" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_proceed_to_next_step_server_error(self):
        """Test server error when proceeding to next step."""
        # Mock the orchestrator to raise a generic exception
        mock_orchestrator = Mock()
        mock_orchestrator.proceed_to_next_step = AsyncMock(side_effect=Exception("Server error"))
        
        with patch('app.api.orchestrator_endpoints.generation_orchestrator', mock_orchestrator):
            with patch('app.api.orchestrator_endpoints.get_current_user', return_value={"id": "test_user"}):
                with pytest.raises(HTTPException) as exc_info:
                    await proceed_to_next_step(
                        request=Mock(session_id="test_session_001"),
                        current_user={"id": "test_user"}
                    )
        
        assert exc_info.value.status_code == 500
        assert "Failed to proceed to next step" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_edit_generated_content_success(self, sample_edit_request):
        """Test successful content editing."""
        # Mock the orchestrator
        mock_orchestrator = Mock()
        mock_orchestrator.user_edit_content = AsyncMock(return_value=Mock(
            session_id="test_session_001",
            current_step=GenerationStep.SECTION_GENERATION,
            status=GenerationStatus.READY_FOR_NEXT,
            completed_steps=[GenerationStep.SOURCE_ANALYSIS, GenerationStep.BLUEPRINT_CREATION],
            current_content={"test": "content"},
            user_edits=[Mock(dict=lambda: {"step": "section_generation", "user_notes": "Test edit"})],
            errors=[],
            started_at=Mock(isoformat=lambda: "2024-01-01T00:00:00"),
            last_updated=Mock(isoformat=lambda: "2024-01-01T00:00:00")
        ))
        
        with patch('app.api.orchestrator_endpoints.generation_orchestrator', mock_orchestrator):
            with patch('app.api.orchestrator_endpoints.get_current_user', return_value={"id": "test_user"}):
                response = await edit_generated_content(
                    session_id="test_session_001",
                    edit_request=Mock(**sample_edit_request),
                    current_user={"id": "test_user"}
                )
        
        # Verify response
        assert response.session_id == "test_session_001"
        assert response.current_step == GenerationStep.SECTION_GENERATION
        assert response.status == GenerationStatus.READY_FOR_NEXT
        assert len(response.user_edits) == 1
        assert response.user_edits[0]["user_notes"] == "Test edit"
    
    @pytest.mark.asyncio
    async def test_edit_generated_content_validation_error(self, sample_edit_request):
        """Test validation error when editing content."""
        # Mock the orchestrator to raise a ValueError
        mock_orchestrator = Mock()
        mock_orchestrator.user_edit_content = AsyncMock(side_effect=ValueError("Invalid edit request"))
        
        with patch('app.api.orchestrator_endpoints.generation_orchestrator', mock_orchestrator):
            with patch('app.api.orchestrator_endpoints.get_current_user', return_value={"id": "test_user"}):
                with pytest.raises(HTTPException) as exc_info:
                    await edit_generated_content(
                        session_id="test_session_001",
                        edit_request=Mock(**sample_edit_request),
                        current_user={"id": "test_user"}
                    )
        
        assert exc_info.value.status_code == 400
        assert "Invalid edit request" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_get_generation_progress_success(self):
        """Test successful retrieval of generation progress."""
        # Mock the orchestrator
        mock_orchestrator = Mock()
        mock_orchestrator.get_generation_progress = AsyncMock(return_value=Mock(
            session_id="test_session_001",
            current_step=GenerationStep.SECTION_GENERATION,
            status=GenerationStatus.READY_FOR_NEXT,
            completed_steps=[GenerationStep.SOURCE_ANALYSIS, GenerationStep.BLUEPRINT_CREATION],
            current_content={"test": "content"},
            user_edits=[],
            errors=[],
            started_at=Mock(isoformat=lambda: "2024-01-01T00:00:00"),
            last_updated=Mock(isoformat=lambda: "2024-01-01T00:00:00")
        ))
        
        with patch('app.api.orchestrator_endpoints.generation_orchestrator', mock_orchestrator):
            with patch('app.api.orchestrator_endpoints.get_current_user', return_value={"id": "test_user"}):
                response = await get_generation_progress(
                    session_id="test_session_001",
                    current_user={"id": "test_user"}
                )
        
        # Verify response
        assert response.session_id == "test_session_001"
        assert response.current_step == GenerationStep.SECTION_GENERATION
        assert response.status == GenerationStatus.READY_FOR_NEXT
        assert "Proceed to next step" in response.next_actions
    
    @pytest.mark.asyncio
    async def test_get_generation_progress_not_found(self):
        """Test error when getting progress for non-existent session."""
        # Mock the orchestrator to raise a ValueError
        mock_orchestrator = Mock()
        mock_orchestrator.get_generation_progress = AsyncMock(side_effect=ValueError("Session not found"))
        
        with patch('app.api.orchestrator_endpoints.generation_orchestrator', mock_orchestrator):
            with patch('app.api.orchestrator_endpoints.get_current_user', return_value={"id": "test_user"}):
                with pytest.raises(HTTPException) as exc_info:
                    await get_generation_progress(
                        session_id="nonexistent",
                        current_user={"id": "test_user"}
                    )
        
        assert exc_info.value.status_code == 404
        assert "Session not found" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_get_complete_learning_path_success(self):
        """Test successful retrieval of complete learning path."""
        # Mock the orchestrator
        mock_orchestrator = Mock()
        mock_orchestrator.get_complete_learning_path = AsyncMock(return_value={
            "session_id": "test_session_001",
            "status": "complete",
            "generated_at": "2024-01-01T00:00:00",
            "completed_at": "2024-01-01T01:00:00",
            "content": {"test": "content"},
            "user_edits": []
        })
        
        with patch('app.api.orchestrator_endpoints.generation_orchestrator', mock_orchestrator):
            with patch('app.api.orchestrator_endpoints.get_current_user', return_value={"id": "test_user"}):
                response = await get_complete_learning_path(
                    session_id="test_session_001",
                    current_user={"id": "test_user"}
                )
        
        # Verify response
        assert response.session_id == "test_session_001"
        assert response.status == "complete"
        assert "content" in response.model_dump()
        assert "user_edits" in response.model_dump()
    
    @pytest.mark.asyncio
    async def test_get_complete_learning_path_not_complete(self):
        """Test error when getting complete path for incomplete session."""
        # Mock the orchestrator to raise a ValueError
        mock_orchestrator = Mock()
        mock_orchestrator.get_complete_learning_path = AsyncMock(side_effect=ValueError("Session not complete"))
        
        with patch('app.api.orchestrator_endpoints.generation_orchestrator', mock_orchestrator):
            with patch('app.api.orchestrator_endpoints.get_current_user', return_value={"id": "test_user"}):
                with pytest.raises(HTTPException) as exc_info:
                    await get_complete_learning_path(
                        session_id="test_session_001",
                        current_user={"id": "test_user"}
                    )
        
        assert exc_info.value.status_code == 400
        assert "Session not complete" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_delete_generation_session_success(self):
        """Test successful deletion of generation session."""
        # Mock the orchestrator
        mock_orchestrator = Mock()
        mock_orchestrator.generation_sessions = {"test_session_001": Mock()}
        
        with patch('app.api.orchestrator_endpoints.generation_orchestrator', mock_orchestrator):
            with patch('app.api.orchestrator_endpoints.get_current_user', return_value={"id": "test_user"}):
                response = await delete_generation_session(
                    session_id="test_session_001",
                    current_user={"id": "test_user"}
                )
        
        # Verify response
        assert response["message"] == "Session test_session_001 deleted successfully"
        assert "test_session_001" not in mock_orchestrator.generation_sessions
    
    @pytest.mark.asyncio
    async def test_delete_generation_session_server_error(self):
        """Test server error when deleting session."""
        # Mock the orchestrator to raise an exception during deletion
        mock_orchestrator = Mock()
        mock_orchestrator.generation_sessions = {"test_session_001": Mock()}
        
        with patch('app.api.orchestrator_endpoints.generation_orchestrator', mock_orchestrator):
            with patch('app.api.orchestrator_endpoints.get_current_user', return_value={"id": "test_user"}):
                # Mock the deletion to raise an exception
                def delete_side_effect(key):
                    raise Exception("Simulated deletion error")
                
                # Replace the dict with a mock that raises on deletion
                mock_dict = Mock()
                mock_dict.__contains__ = lambda x: x == "test_session_001"
                mock_dict.__delitem__ = Mock(side_effect=delete_side_effect)
                mock_orchestrator.generation_sessions = mock_dict
                
                with pytest.raises(HTTPException) as exc_info:
                    await delete_generation_session(
                        session_id="test_session_001",
                        current_user={"id": "test_user"}
                    )
        
        assert exc_info.value.status_code == 500
        assert "Failed to delete session" in str(exc_info.value.detail)


class TestHelperFunctions:
    """Test cases for helper functions."""
    
    def test_get_next_actions_ready_for_next(self):
        """Test next actions when status is ready_for_next."""
        progress = Mock(
            status=GenerationStatus.READY_FOR_NEXT,
            current_step=GenerationStep.SECTION_GENERATION
        )
        
        actions = _get_next_actions(progress)
        
        assert "Proceed to next step" in actions
        assert "Edit current content" in actions
        assert "View current progress" in actions
        assert "Delete session" in actions
    
    def test_get_next_actions_complete(self):
        """Test next actions when status is complete."""
        progress = Mock(
            status=GenerationStatus.READY_FOR_NEXT,
            current_step=GenerationStep.COMPLETE
        )
        
        actions = _get_next_actions(progress)
        
        assert "View complete learning path" in actions
        assert "View current progress" in actions
        assert "Delete session" in actions
    
    def test_get_next_actions_user_editing(self):
        """Test next actions when status is user_editing."""
        progress = Mock(
            status=GenerationStatus.USER_EDITING,
            current_step=GenerationStep.SECTION_GENERATION
        )
        
        actions = _get_next_actions(progress)
        
        assert "Continue editing" in actions
        assert "Mark edits complete" in actions
        assert "View current progress" in actions
        assert "Delete session" in actions
    
    def test_get_next_actions_failed(self):
        """Test next actions when status is failed."""
        progress = Mock(
            status=GenerationStatus.FAILED,
            current_step=GenerationStep.SECTION_GENERATION
        )
        
        actions = _get_next_actions(progress)
        
        assert "Review errors" in actions
        assert "Restart session" in actions
        assert "View current progress" in actions
        assert "Delete session" in actions


class TestAPIIntegration:
    """Integration tests for the API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the API."""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)
    
    def test_start_endpoint_integration(self, client):
        """Test the start endpoint integration."""
        # Mock the orchestrator
        with patch('app.api.orchestrator_endpoints.generation_orchestrator') as mock_orchestrator:
            mock_orchestrator.start_generation_session = AsyncMock(return_value=Mock(
                current_step=GenerationStep.SOURCE_ANALYSIS,
                status=GenerationStatus.READY_FOR_NEXT
            ))
            
            # Mock uuid generation
            with patch('app.api.orchestrator_endpoints.uuid.uuid4', return_value="test_session_001"):
                response = client.post(
                    "/api/v1/orchestrator/start",
                    json={
                        "source_content": "Test content",
                        "source_type": "textbook_chapter",
                        "user_preferences": {"learning_style": "visual"}
                    }
                )
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test_session_001"
        assert data["current_step"] == "source_analysis"
        assert data["status"] == "ready_for_next"
    
    def test_proceed_endpoint_integration(self, client):
        """Test the proceed endpoint integration."""
        # Mock the orchestrator
        with patch('app.api.orchestrator_endpoints.generation_orchestrator') as mock_orchestrator:
            mock_orchestrator.proceed_to_next_step = AsyncMock(return_value=Mock(
                session_id="test_session_001",
                current_step=GenerationStep.BLUEPRINT_CREATION,
                status=GenerationStatus.READY_FOR_NEXT,
                completed_steps=[GenerationStep.SOURCE_ANALYSIS],
                current_content={"test": "content"},
                user_edits=[],
                errors=[],
                started_at=Mock(isoformat=lambda: "2024-01-01T00:00:00"),
                last_updated=Mock(isoformat=lambda: "2024-01-01T00:00:00")
            ))
            
            response = client.post(
                "/api/v1/orchestrator/proceed",
                json={"session_id": "test_session_001"}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test_session_001"
        assert data["current_step"] == "blueprint_creation"
        assert data["status"] == "ready_for_next"
    
    def test_edit_endpoint_integration(self, client):
        """Test the edit endpoint integration."""
        # Mock the orchestrator
        with patch('app.api.orchestrator_endpoints.generation_orchestrator') as mock_orchestrator:
            mock_orchestrator.user_edit_content = AsyncMock(return_value=Mock(
                session_id="test_session_001",
                current_step=GenerationStep.SECTION_GENERATION,
                status=GenerationStatus.READY_FOR_NEXT,
                completed_steps=[GenerationStep.SOURCE_ANALYSIS, GenerationStep.BLUEPRINT_CREATION],
                current_content={"test": "content"},
                user_edits=[Mock(dict=lambda: {"step": "section_generation", "user_notes": "Test edit"})],
                errors=[],
                started_at=Mock(isoformat=lambda: "2024-01-01T00:00:00"),
                last_updated=Mock(isoformat=lambda: "2024-01-01T00:00:00")
            ))
            
            response = client.post(
                "/api/v1/orchestrator/edit?session_id=test_session_001",
                json={
                    "step": "section_generation",
                    "content_id": "sec_001",
                    "edited_content": {"title": "Updated Title"},
                    "user_notes": "Test edit"
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test_session_001"
        assert data["current_step"] == "section_generation"
        assert data["status"] == "ready_for_next"
    
    def test_progress_endpoint_integration(self, client):
        """Test the progress endpoint integration."""
        # Mock the orchestrator
        with patch('app.api.orchestrator_endpoints.generation_orchestrator') as mock_orchestrator:
            mock_orchestrator.get_generation_progress = AsyncMock(return_value=Mock(
                session_id="test_session_001",
                current_step=GenerationStep.SECTION_GENERATION,
                status=GenerationStatus.READY_FOR_NEXT,
                completed_steps=[GenerationStep.SOURCE_ANALYSIS, GenerationStep.BLUEPRINT_CREATION],
                current_content={"test": "content"},
                user_edits=[],
                errors=[],
                started_at=Mock(isoformat=lambda: "2024-01-01T00:00:00"),
                last_updated=Mock(isoformat=lambda: "2024-01-01T00:00:00")
            ))
            
            response = client.get("/api/v1/orchestrator/progress/test_session_001")
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test_session_001"
        assert data["current_step"] == "section_generation"
        assert data["status"] == "ready_for_next"
    
    def test_complete_endpoint_integration(self, client):
        """Test the complete endpoint integration."""
        # Mock the orchestrator
        with patch('app.api.orchestrator_endpoints.generation_orchestrator') as mock_orchestrator:
            mock_orchestrator.get_complete_learning_path = AsyncMock(return_value={
                "session_id": "test_session_001",
                "status": "complete",
                "generated_at": "2024-01-01T00:00:00",
                "completed_at": "2024-01-01T01:00:00",
                "content": {"test": "content"},
                "user_edits": []
            })
            
            response = client.get("/api/v1/orchestrator/complete/test_session_001")
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test_session_001"
        assert data["status"] == "complete"
    
    def test_delete_endpoint_integration(self, client):
        """Test the delete endpoint integration."""
        # Mock the orchestrator
        with patch('app.api.orchestrator_endpoints.generation_orchestrator') as mock_orchestrator:
            mock_orchestrator.generation_sessions = {"test_session_001": Mock()}
            
            response = client.delete("/api/v1/orchestrator/session/test_session_001")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Session test_session_001 deleted successfully"


if __name__ == "__main__":
    pytest.main([__file__])
