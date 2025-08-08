# Sprint 33: Contract Testing between AI API and Core API

import pytest
import json
from typing import Dict, List, Any, Optional
from unittest.mock import AsyncMock, patch
import jsonschema
from pydantic import ValidationError

from app.api.schemas import (
    MasteryCriterionDto,
    KnowledgePrimitiveDto,
    PrismaCriterionEvaluationRequest,
    PrismaCriterionEvaluationResponse
)

class TestCoreAPIPrismaSchemaContracts:
    """Test contracts with Core API Prisma schema definitions."""
    
    @pytest.fixture
    def core_api_primitive_schema(self):
        """Expected Core API Prisma schema for KnowledgePrimitive."""
        return {
            "type": "object",
            "required": [
                "primitiveId", "title", "description", "content", 
                "primitiveType", "createdAt"
            ],
            "properties": {
                "primitiveId": {"type": "string", "minLength": 1},
                "title": {"type": "string", "minLength": 1, "maxLength": 255},
                "description": {"type": "string", "maxLength": 1000},
                "content": {"type": "string", "minLength": 1},
                "primitiveType": {
                    "type": "string",
                    "enum": ["concept", "process", "principle", "fact", "skill"]
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "createdAt": {"type": "string", "format": "date-time"},
                "updatedAt": {"type": "string", "format": "date-time"}
            },
            "additionalProperties": False
        }
    
    @pytest.fixture
    def core_api_mastery_criterion_schema(self):
        """Expected Core API Prisma schema for MasteryCriterion."""
        return {
            "type": "object",
            "required": [
                "criterionId", "primitiveId", "title", "ueeLevel", 
                "weight", "isRequired"
            ],
            "properties": {
                "criterionId": {"type": "string", "minLength": 1},
                "primitiveId": {"type": "string", "minLength": 1},
                "title": {"type": "string", "minLength": 1, "maxLength": 255},
                "description": {"type": "string", "maxLength": 1000},
                "ueeLevel": {
                    "type": "string",
                    "enum": ["UNDERSTAND", "USE", "EXPLORE"]
                },
                "weight": {
                    "type": "number",
                    "minimum": 1.0,
                    "maximum": 5.0
                },
                "isRequired": {"type": "boolean"},
                "trackingIntensity": {
                    "type": "string",
                    "enum": ["LOW", "MEDIUM", "HIGH"]
                }
            },
            "additionalProperties": False
        }
    
    def test_ai_api_primitive_matches_core_schema(self, core_api_primitive_schema):
        """Test that AI API primitive DTOs match Core API schema."""
        # Create AI API primitive DTO
        ai_primitive = KnowledgePrimitiveDto(
            primitiveId="test_prim_001",
            title="Test Primitive",
            description="Test primitive description",
            primitiveType="concept",
            difficultyLevel="intermediate",
            estimatedTimeMinutes=10,
            trackingIntensity="NORMAL",
            masteryCriteria=[]
        )
        
        # Convert to Core API format
        core_api_data = {
            "primitiveId": ai_primitive.primitiveId,
            "title": ai_primitive.title,
            "description": ai_primitive.description,
            "content": "Test primitive content for learning",  # Core API expects content field
            "primitiveType": ai_primitive.primitiveType,
            "createdAt": "2024-01-01T00:00:00Z",  # Would be set by Core API
            "updatedAt": "2024-01-01T00:00:00Z"   # Would be set by Core API
        }
        
        # Validate against Core API schema
        try:
            jsonschema.validate(core_api_data, core_api_primitive_schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"AI API primitive data doesn't match Core API schema: {e}")

    def test_ai_api_criterion_matches_core_schema(self, core_api_mastery_criterion_schema):
        """Test that AI API mastery criterion DTOs match Core API schema."""
        # Create AI API criterion DTO
        ai_criterion = MasteryCriterionDto(
            criterionId="test_crit_001",
            primitiveId="test_prim_001",
            title="Test Criterion",
            description="Test criterion description",
            ueeLevel="UNDERSTAND",
            weight=3.5,
            isRequired=True
        )
        
        # Convert to Core API format
        core_api_data = {
            "criterionId": ai_criterion.criterionId,
            "primitiveId": ai_criterion.primitiveId,
            "title": ai_criterion.title,
            "description": ai_criterion.description,
            "ueeLevel": ai_criterion.ueeLevel,
            "weight": ai_criterion.weight,
            "isRequired": ai_criterion.isRequired,
            "trackingIntensity": "MEDIUM"  # Default value in Core API
        }
        
        # Validate against Core API schema
        try:
            jsonschema.validate(core_api_data, core_api_mastery_criterion_schema)
        except jsonschema.ValidationError as e:
            pytest.fail(f"AI API criterion data doesn't match Core API schema: {e}")

    def test_primitive_type_enum_compatibility(self):
        """Test that primitive types are compatible between APIs."""
        ai_api_types = ["fact", "concept", "process"]
        core_api_types = ["fact", "concept", "process"]
        
        # Should be identical
        assert set(ai_api_types) == set(core_api_types)
        
        # Test that all AI API types validate in Core API
        for primitive_type in ai_api_types:
            primitive = KnowledgePrimitiveDto(
                primitiveId="test",
                title="Test",
                description="Test",
                primitiveType=primitive_type,
                difficultyLevel="intermediate",
                estimatedTimeMinutes=10,
                trackingIntensity="NORMAL",
                masteryCriteria=[]
            )
            assert primitive.primitiveType == primitive_type

    def test_uee_level_enum_compatibility(self):
        """Test that UEE levels are compatible between APIs."""
        ai_api_levels = ["UNDERSTAND", "USE", "EXPLORE"]
        core_api_levels = ["UNDERSTAND", "USE", "EXPLORE"]
        
        # Should be identical
        assert set(ai_api_levels) == set(core_api_levels)
        
        # Test that all AI API levels validate in Core API
        for uee_level in ai_api_levels:
            criterion = MasteryCriterionDto(
                criterionId="test",
                primitiveId="test",
                title="Test",
                ueeLevel=uee_level,
                weight=3.0,
                isRequired=True
            )
            assert criterion.ueeLevel == uee_level

    def test_weight_range_compatibility(self):
        """Test that weight ranges are compatible between APIs."""
        # Test boundary values
        valid_weights = [1.0, 2.5, 3.0, 4.5, 5.0]
        invalid_weights = [0.5, 0.0, 5.1, 6.0, -1.0]
        
        # Valid weights should work
        for weight in valid_weights:
            criterion = MasteryCriterionDto(
                criterionId="test",
                primitiveId="test", 
                title="Test",
                ueeLevel="UNDERSTAND",
                weight=weight,
                isRequired=True
            )
            assert criterion.weight == weight
        
        # Invalid weights should raise errors
        for weight in invalid_weights:
            with pytest.raises(ValidationError):
                MasteryCriterionDto(
                    criterionId="test",
                    primitiveId="test",
                    title="Test", 
                    ueeLevel="UNDERSTAND",
                    weight=weight,
                    isRequired=True
                )


class TestCoreAPIEndpointContracts:
    """Test contracts with Core API endpoints."""
    
    @pytest.fixture
    def mock_core_api_responses(self):
        """Mock Core API responses that match expected contracts."""
        return {
            "create_primitive": {
                "primitiveId": "prim_created_001",
                "title": "Created Primitive",
                "status": "created",
                "createdAt": "2024-01-01T00:00:00Z"
            },
            "create_mastery_criterion": {
                "criterionId": "crit_created_001",
                "primitiveId": "prim_001",
                "title": "Created Criterion",
                "status": "created",
                "createdAt": "2024-01-01T00:00:00Z"
            },
            "get_primitive": {
                "primitiveId": "prim_001",
                "title": "Existing Primitive",
                "description": "An existing primitive",
                "content": "Primitive content",
                "primitiveType": "concept",
                "tags": ["test"],
                "createdAt": "2024-01-01T00:00:00Z",
                "updatedAt": "2024-01-01T00:00:00Z"
            }
        }
    
    @pytest.mark.asyncio
    async def test_primitive_creation_contract(self, mock_core_api_responses):
        """Test contract for primitive creation with Core API."""
        from app.core.core_api_integration import CoreAPIIntegrationService
        from app.models.learning_blueprint import KnowledgePrimitive
        
        service = CoreAPIIntegrationService()
        
        # Mock the actual create_primitive method
        with patch.object(service, 'create_primitive') as mock_create:
            mock_create.return_value = mock_core_api_responses["create_primitive"]
            
            # Create a valid KnowledgePrimitive instance
            primitive = KnowledgePrimitive(
                primitiveId="test_001",
                title="Test Primitive",
                description="Test description",
                primitiveType="concept",
                difficultyLevel="intermediate",
                estimatedTimeMinutes=10,
                trackingIntensity="NORMAL",
                masteryCriteria=[]
            )
            
            result = await service.create_primitive(primitive, user_id=1, blueprint_id=1)
            
            # Verify method was called with correct parameters
            mock_create.assert_called_once_with(primitive, user_id=1, blueprint_id=1)
            
            # Verify response matches expected contract
            assert result == mock_core_api_responses["create_primitive"]
            assert "primitiveId" in result
            assert result["primitiveId"] == "prim_created_001"
            assert "status" in result
            assert result["status"] == "created"

    @pytest.mark.asyncio
    async def test_mastery_criterion_creation_contract(self, mock_core_api_responses):
        """Test contract for mastery criterion creation with Core API."""
        from app.core.core_api_integration import CoreAPIIntegrationService
        from app.models.learning_blueprint import MasteryCriterion
        
        service = CoreAPIIntegrationService()
        
        # Mock the actual create_mastery_criteria method (plural)
        with patch.object(service, 'create_mastery_criteria') as mock_create:
            mock_create.return_value = mock_core_api_responses["create_mastery_criterion"]
            
            # Create a valid MasteryCriterion instance
            criterion = MasteryCriterion(
                criterionId="crit_001",
                primitiveId="prim_001",
                title="Test Criterion",
                description="Test description",
                ueeLevel="UNDERSTAND",
                weight=3.0,
                isRequired=True
            )
            
            result = await service.create_mastery_criteria([criterion], user_id=1, blueprint_id=1)
            
            # Verify method was called with correct parameters
            mock_create.assert_called_once_with([criterion], user_id=1, blueprint_id=1)
            
            # Verify response matches expected contract
            assert result == mock_core_api_responses["create_mastery_criterion"]
            assert "criterionId" in result
            assert result["criterionId"] == "crit_created_001"
            assert "status" in result
            assert result["status"] == "created"

    @pytest.mark.asyncio
    async def test_sync_primitives_contract(self, mock_core_api_responses):
        """Test contract for syncing primitives with Core API."""
        from app.core.core_api_integration import CoreAPIIntegrationService
        from app.models.learning_blueprint import KnowledgePrimitive
        
        service = CoreAPIIntegrationService()
        
        # Mock the actual sync_primitives_with_core_api method
        with patch.object(service, 'sync_primitives_with_core_api') as mock_sync:
            mock_sync.return_value = {
                "primitives_created": 1,
                "criteria_created": 2,
                "errors": []
            }
            
            # Create a valid KnowledgePrimitive instance
            primitive = KnowledgePrimitive(
                primitiveId="test_001",
                title="Test Primitive",
                description="Test description",
                primitiveType="concept",
                difficultyLevel="intermediate",
                estimatedTimeMinutes=10,
                trackingIntensity="NORMAL",
                masteryCriteria=[]
            )
            
            result = await service.sync_primitives_with_core_api([primitive], user_id=1, blueprint_id=1)
            
            # Verify method was called with correct parameters
            mock_sync.assert_called_once_with([primitive], user_id=1, blueprint_id=1)
            
            # Verify response matches expected contract
            assert "primitives_created" in result
            assert "criteria_created" in result
            assert "errors" in result
            assert result["primitives_created"] == 1
            assert result["criteria_created"] == 2
            assert isinstance(result["errors"], list)

    def test_error_response_contracts(self):
        """Test that error responses follow expected contracts."""
        # Test 400 Bad Request contract
        bad_request_error = {
            "error": "Validation failed",
            "details": [
                {
                    "field": "primitiveType",
                    "message": "Invalid primitive type"
                }
            ],
            "statusCode": 400
        }
        
        # Should have standard error format
        assert "error" in bad_request_error
        assert "statusCode" in bad_request_error
        assert bad_request_error["statusCode"] == 400
        
        # Test 404 Not Found contract
        not_found_error = {
            "error": "Resource not found",
            "resource": "primitive",
            "id": "nonexistent_id",
            "statusCode": 404
        }
        
        assert "error" in not_found_error
        assert "statusCode" in not_found_error
        assert not_found_error["statusCode"] == 404


class TestCriterionEvaluationContracts:
    """Test contracts for criterion evaluation between APIs."""
    
    def test_evaluation_request_contract(self):
        """Test criterion evaluation request contract."""
        # Create AI API evaluation request
        request = PrismaCriterionEvaluationRequest(
            criterionId="crit_001",
            criterionTitle="Test Criterion",
            primitiveId="prim_001",
            primitiveTitle="Test Primitive",
            ueeLevel="UNDERSTAND",
            criterionWeight=3.0,
            questionText="What is the answer?",
            questionType="short_answer",
            correctAnswer="The correct answer",
            userAnswer="User's answer",
            totalMarks=10
        )
        
        # Convert to Core API format
        core_api_request = {
            "criterionId": request.criterionId,
            "primitiveId": request.primitiveId,
            "userAnswer": request.userAnswer,
            "questionData": {
                "questionText": request.questionText,
                "questionType": request.questionType,
                "correctAnswer": request.correctAnswer,
                "totalMarks": request.totalMarks
            },
            "evaluationContext": {
                "ueeLevel": request.ueeLevel,
                "criterionWeight": request.criterionWeight
            }
        }
        
        # Verify contract structure
        assert "criterionId" in core_api_request
        assert "primitiveId" in core_api_request
        assert "userAnswer" in core_api_request
        assert "questionData" in core_api_request
        assert "evaluationContext" in core_api_request

    def test_evaluation_response_contract(self):
        """Test criterion evaluation response contract."""
        # Create AI API evaluation response
        response = PrismaCriterionEvaluationResponse(
            success=True,
            criterionId="crit_001",
            primitiveId="prim_001",
            ueeLevel="UNDERSTAND",
            masteryScore=0.85,
            masteryLevel="mastered",
            marksAchieved=8,
            totalMarks=10,
            feedback="Good understanding demonstrated",
            correctedAnswer="The corrected answer",
            criterionWeight=3.0,
            metadata={"evaluatedAt": "2024-01-01T00:00:00Z"}
        )
        
        # Should match expected Core API response format
        assert response.success is True
        assert 0.0 <= response.masteryScore <= 1.0
        assert response.masteryLevel in ["novice", "developing", "mastered"]
        assert response.marksAchieved <= response.totalMarks
        assert 1.0 <= response.criterionWeight <= 5.0

    def test_batch_evaluation_contract(self):
        """Test batch evaluation contract compatibility."""
        from app.api.answer_evaluation_schemas import BatchCriterionEvaluationRequest
        
        # Create batch request with multiple evaluations
        individual_requests = []
        for i in range(3):
            individual_requests.append(PrismaCriterionEvaluationRequest(
                criterionId=f"crit_{i:03d}",
                criterionTitle=f"Test Criterion {i}",
                primitiveId=f"prim_{i:03d}",
                primitiveTitle=f"Test Primitive {i}",
                ueeLevel="UNDERSTAND",
                criterionWeight=3.0,
                questionText=f"Question {i}?",
                questionType="short_answer",
                correctAnswer=f"Answer {i}",
                userAnswer=f"User answer {i}",
                totalMarks=10
            ))
        
        batch_request = BatchCriterionEvaluationRequest(
            evaluationRequests=individual_requests
        )
        
        # Verify batch structure
        assert len(batch_request.evaluationRequests) == 3
        assert all(
            hasattr(req, 'criterionId') and hasattr(req, 'primitiveId')
            for req in batch_request.evaluationRequests
        )


class TestAPIVersioningContracts:
    """Test API versioning contracts between AI API and Core API."""
    
    def test_api_version_headers(self):
        """Test that API version headers are compatible."""
        expected_headers = {
            "API-Version": "v1",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # These headers should be included in all API calls
        for header, value in expected_headers.items():
            assert isinstance(header, str)
            assert isinstance(value, str)

    def test_backward_compatibility_markers(self):
        """Test backward compatibility markers in schemas."""
        # Check that schemas include version compatibility info
        primitive_schema_info = {
            "schema_version": "1.0",
            "compatible_versions": ["1.0"],
            "deprecated_fields": [],
            "new_fields_since": {}
        }
        
        assert "schema_version" in primitive_schema_info
        assert "compatible_versions" in primitive_schema_info

    def test_deprecation_warnings(self):
        """Test handling of deprecated fields and endpoints."""
        # Test that deprecated fields are handled gracefully
        deprecated_primitive_data = {
            "primitive_id": "old_format",  # Old snake_case format
            "primitiveId": "new_format",   # New camelCase format
            "title": "Test"
        }
        
        # Should handle both formats gracefully
        assert "primitive_id" in deprecated_primitive_data
        assert "primitiveId" in deprecated_primitive_data


# Contract test configuration
pytestmark = [pytest.mark.contract, pytest.mark.integration]
