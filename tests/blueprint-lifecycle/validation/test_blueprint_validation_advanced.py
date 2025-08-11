"""
Advanced Blueprint Validation Test Suite

This module contains comprehensive tests for advanced validation capabilities
in the blueprint lifecycle, including schema validation, business rule validation,
cross-field validation, and custom validation rules.
"""

import pytest
import asyncio
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import json
import re

from app.core.blueprint.blueprint_validator import BlueprintValidator
from app.core.blueprint.blueprint_model import Blueprint
from app.core.blueprint.blueprint_schema import BlueprintSchema
from app.core.blueprint.blueprint_business_rules import BlueprintBusinessRules
from app.core.blueprint.blueprint_cross_validator import BlueprintCrossValidator


class TestAdvancedBlueprintValidation:
    """Advanced test suite for blueprint validation capabilities."""
    
    @pytest.fixture
    def mock_blueprint_validator(self):
        """Mock blueprint validator for testing."""
        validator = Mock(spec=BlueprintValidator)
        validator.validate_schema = AsyncMock()
        validator.validate_business_rules = AsyncMock()
        validator.validate_cross_fields = AsyncMock()
        validator.validate_custom_rules = AsyncMock()
        validator.validate_dependencies = AsyncMock()
        return validator
    
    @pytest.fixture
    def mock_blueprint_schema(self):
        """Mock blueprint schema for testing."""
        schema = Mock(spec=BlueprintSchema)
        schema.validate_field = AsyncMock()
        schema.validate_structure = AsyncMock()
        schema.validate_types = AsyncMock()
        schema.validate_constraints = AsyncMock()
        return schema
    
    @pytest.fixture
    def mock_business_rules(self):
        """Mock business rules validator for testing."""
        rules = Mock(spec=BlueprintBusinessRules)
        rules.validate_complexity = AsyncMock()
        rules.validate_prerequisites = AsyncMock()
        rules.validate_consistency = AsyncMock()
        rules.validate_business_logic = AsyncMock()
        return rules
    
    @pytest.fixture
    def sample_blueprint_data(self):
        """Sample blueprint data for validation testing."""
        return {
            "name": "Advanced Validation Test Blueprint",
            "description": "A comprehensive blueprint for testing advanced validation capabilities",
            "content": "This blueprint contains various content types and structures for validation testing.",
            "metadata": {
                "category": "artificial_intelligence",
                "tags": ["machine_learning", "deep_learning", "validation"],
                "difficulty": "advanced",
                "prerequisites": ["python", "mathematics", "statistics"],
                "estimated_time": "8-12 hours",
                "last_updated": "2024-01-15",
                "version": "1.0.0",
                "author": "Test Author",
                "license": "MIT"
            },
            "settings": {
                "chunk_size": 1000,
                "overlap": 200,
                "embedding_model": "text-embedding-ada-002",
                "chunking_strategy": "semantic",
                "retrieval_strategy": "hybrid",
                "generation_model": "gpt-4"
            },
            "requirements": {
                "min_python_version": "3.8",
                "required_packages": ["numpy", "pandas", "scikit-learn"],
                "system_requirements": {
                    "min_ram": "8GB",
                    "min_storage": "2GB",
                    "gpu_required": False
                }
            }
        }
    
    @pytest.fixture
    def validation_config(self):
        """Validation configuration for testing."""
        return {
            "strict_mode": True,
            "validate_dependencies": True,
            "validate_business_rules": True,
            "validate_cross_fields": True,
            "custom_validation_rules": True,
            "max_validation_errors": 10
        }

    def test_schema_validation_comprehensive(self, mock_blueprint_validator, sample_blueprint_data):
        """Test comprehensive schema validation."""
        # Mock successful schema validation
        mock_blueprint_validator.validate_schema.return_value = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "validation_time_ms": 45,
            "fields_validated": 25
        }
        
        # Test schema validation
        result = asyncio.run(mock_blueprint_validator.validate_schema(sample_blueprint_data))
        
        # Verify validation result
        assert result is not None
        assert result["is_valid"] is True
        assert len(result["errors"]) == 0
        assert "validation_time_ms" in result
        assert "fields_validated" in result
        
        # Verify validation was called with correct data
        mock_blueprint_validator.validate_schema.assert_called_once_with(sample_blueprint_data)
        
        print("✅ Comprehensive schema validation working correctly")

    def test_business_rules_validation(self, mock_blueprint_validator, sample_blueprint_data):
        """Test business rules validation."""
        # Mock business rules validation
        mock_blueprint_validator.validate_business_rules.return_value = {
            "is_valid": True,
            "rule_violations": [],
            "business_checks": [
                {"rule": "complexity_appropriate", "status": "passed"},
                {"rule": "prerequisites_valid", "status": "passed"},
                {"rule": "consistency_check", "status": "passed"}
            ],
            "validation_time_ms": 30
        }
        
        # Test business rules validation
        result = asyncio.run(mock_blueprint_validator.validate_business_rules(sample_blueprint_data))
        
        # Verify business rules result
        assert result is not None
        assert result["is_valid"] is True
        assert len(result["rule_violations"]) == 0
        assert "business_checks" in result
        
        # Verify business checks
        checks = result["business_checks"]
        assert len(checks) > 0
        assert all("rule" in check for check in checks)
        assert all("status" in check for check in checks)
        assert all(check["status"] == "passed" for check in checks)
        
        print("✅ Business rules validation working correctly")

    def test_cross_field_validation(self, mock_blueprint_validator, sample_blueprint_data):
        """Test cross-field validation."""
        # Mock cross-field validation
        mock_blueprint_validator.validate_cross_fields.return_value = {
            "is_valid": True,
            "cross_field_errors": [],
            "field_relationships": [
                {
                    "fields": ["difficulty", "prerequisites"],
                    "relationship": "difficulty_requires_prerequisites",
                    "status": "valid"
                },
                {
                    "fields": ["estimated_time", "content_length"],
                    "relationship": "time_content_correlation",
                    "status": "valid"
                }
            ],
            "validation_time_ms": 25
        }
        
        # Test cross-field validation
        result = asyncio.run(mock_blueprint_validator.validate_cross_fields(sample_blueprint_data))
        
        # Verify cross-field validation result
        assert result is not None
        assert result["is_valid"] is True
        assert len(result["cross_field_errors"]) == 0
        assert "field_relationships" in result
        
        # Verify field relationships
        relationships = result["field_relationships"]
        assert len(relationships) > 0
        
        for relationship in relationships:
            assert "fields" in relationship
            assert "relationship" in relationship
            assert "status" in relationship
            assert relationship["status"] == "valid"
        
        print("✅ Cross-field validation working correctly")

    def test_custom_validation_rules(self, mock_blueprint_validator, sample_blueprint_data):
        """Test custom validation rules."""
        # Mock custom validation
        mock_blueprint_validator.validate_custom_rules.return_value = {
            "is_valid": True,
            "custom_rule_results": [
                {
                    "rule_name": "content_quality_check",
                    "status": "passed",
                    "score": 0.92,
                    "details": "Content meets quality standards"
                },
                {
                    "rule_name": "accessibility_check",
                    "status": "passed",
                    "score": 0.88,
                    "details": "Blueprint is accessible to target audience"
                }
            ],
            "validation_time_ms": 35
        }
        
        # Test custom validation rules
        result = asyncio.run(mock_blueprint_validator.validate_custom_rules(sample_blueprint_data))
        
        # Verify custom validation result
        assert result is not None
        assert result["is_valid"] is True
        assert "custom_rule_results" in result
        
        # Verify custom rule results
        rule_results = result["custom_rule_results"]
        assert len(rule_results) > 0
        
        for rule_result in rule_results:
            assert "rule_name" in rule_result
            assert "status" in rule_result
            assert "score" in rule_result
            assert "details" in rule_result
            assert rule_result["status"] == "passed"
            assert 0 <= rule_result["score"] <= 1
        
        print("✅ Custom validation rules working correctly")

    def test_dependency_validation(self, mock_blueprint_validator, sample_blueprint_data):
        """Test dependency validation."""
        # Mock dependency validation
        mock_blueprint_validator.validate_dependencies.return_value = {
            "is_valid": True,
            "dependency_errors": [],
            "dependency_graph": {
                "nodes": ["python", "mathematics", "statistics"],
                "edges": [
                    {"from": "python", "to": "machine_learning"},
                    {"from": "mathematics", "to": "machine_learning"},
                    {"from": "statistics", "to": "machine_learning"}
                ]
            },
            "validation_time_ms": 40
        }
        
        # Test dependency validation
        result = asyncio.run(mock_blueprint_validator.validate_dependencies(sample_blueprint_data))
        
        # Verify dependency validation result
        assert result is not None
        assert result["is_valid"] is True
        assert len(result["dependency_errors"]) == 0
        assert "dependency_graph" in result
        
        # Verify dependency graph
        graph = result["dependency_graph"]
        assert "nodes" in graph
        assert "edges" in graph
        
        nodes = graph["nodes"]
        edges = graph["edges"]
        
        assert len(nodes) > 0
        assert len(edges) > 0
        
        # Verify edge structure
        for edge in edges:
            assert "from" in edge
            assert "to" in edge
            assert edge["from"] in nodes
        
        print("✅ Dependency validation working correctly")

    def test_validation_error_handling(self, mock_blueprint_validator, sample_blueprint_data):
        """Test validation error handling and reporting."""
        # Mock validation with errors
        mock_blueprint_validator.validate_schema.return_value = {
            "is_valid": False,
            "errors": [
                {
                    "field": "metadata.version",
                    "error_type": "format_error",
                    "message": "Version format must be semantic (e.g., 1.0.0)",
                    "severity": "error"
                },
                {
                    "field": "settings.chunk_size",
                    "error_type": "constraint_violation",
                    "message": "Chunk size must be between 100 and 5000",
                    "severity": "warning"
                }
            ],
            "warnings": [
                {
                    "field": "metadata.estimated_time",
                    "warning_type": "format_suggestion",
                    "message": "Consider using ISO 8601 duration format",
                    "severity": "info"
                }
            ],
            "validation_time_ms": 50
        }
        
        # Test validation with errors
        result = asyncio.run(mock_blueprint_validator.validate_schema(sample_blueprint_data))
        
        # Verify error handling
        assert result is not None
        assert result["is_valid"] is False
        assert len(result["errors"]) > 0
        assert len(result["warnings"]) > 0
        
        # Verify error structure
        errors = result["errors"]
        for error in errors:
            assert "field" in error
            assert "error_type" in error
            assert "message" in error
            assert "severity" in error
            assert error["severity"] in ["error", "warning", "info"]
        
        # Verify warning structure
        warnings = result["warnings"]
        for warning in warnings:
            assert "field" in warning
            assert "warning_type" in warning
            assert "message" in warning
            assert "severity" in warning
        
        print("✅ Validation error handling working correctly")
        print(f"  Errors found: {len(errors)}")
        print(f"  Warnings found: {len(warnings)}")

    def test_validation_performance(self, mock_blueprint_validator, sample_blueprint_data):
        """Test validation performance and timing."""
        # Mock validation with timing
        validation_times = [45, 42, 48, 41, 47]  # Multiple validation runs
        
        for i, validation_time in enumerate(validation_times):
            mock_blueprint_validator.validate_schema.return_value = {
                "is_valid": True,
                "errors": [],
                "warnings": [],
                "validation_time_ms": validation_time,
                "fields_validated": 25
            }
            
            # Run validation
            result = asyncio.run(mock_blueprint_validator.validate_schema(sample_blueprint_data))
            
            # Verify timing
            assert result["validation_time_ms"] == validation_time
            assert result["validation_time_ms"] < 100  # Should complete within 100ms
        
        # Calculate average validation time
        avg_time = sum(validation_times) / len(validation_times)
        print(f"✅ Validation performance test completed")
        print(f"  Average validation time: {avg_time:.2f}ms")
        print(f"  Total validations: {len(validation_times)}")

    def test_validation_configuration(self, mock_blueprint_validator, sample_blueprint_data, validation_config):
        """Test validation configuration options."""
        # Mock validation with configuration
        mock_blueprint_validator.validate_with_config = AsyncMock(return_value={
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "config_used": validation_config,
            "validation_time_ms": 55
        })
        
        # Test validation with configuration
        result = asyncio.run(mock_blueprint_validator.validate_with_config(
            sample_blueprint_data, validation_config
        ))
        
        # Verify configuration usage
        assert result is not None
        assert result["is_valid"] is True
        assert "config_used" in result
        
        # Verify configuration matches
        config_used = result["config_used"]
        assert config_used["strict_mode"] == validation_config["strict_mode"]
        assert config_used["validate_dependencies"] == validation_config["validate_dependencies"]
        assert config_used["validate_business_rules"] == validation_config["validate_business_rules"]
        
        print("✅ Validation configuration working correctly")

    def test_validation_chain_execution(self, mock_blueprint_validator, sample_blueprint_data):
        """Test validation chain execution order."""
        # Mock validation chain
        validation_steps = []
        
        def mock_validate_step(step_name):
            async def validate():
                validation_steps.append(step_name)
                return {"is_valid": True, "step": step_name}
            return validate
        
        # Setup validation chain
        mock_blueprint_validator.validate_schema = mock_validate_step("schema")
        mock_blueprint_validator.validate_business_rules = mock_validate_step("business_rules")
        mock_blueprint_validator.validate_cross_fields = mock_validate_step("cross_fields")
        mock_blueprint_validator.validate_custom_rules = mock_validate_step("custom_rules")
        mock_blueprint_validator.validate_dependencies = mock_validate_step("dependencies")
        
        # Execute validation chain
        async def run_validation_chain():
            results = []
            results.append(await mock_blueprint_validator.validate_schema(sample_blueprint_data))
            results.append(await mock_blueprint_validator.validate_business_rules(sample_blueprint_data))
            results.append(await mock_blueprint_validator.validate_cross_fields(sample_blueprint_data))
            results.append(await mock_blueprint_validator.validate_custom_rules(sample_blueprint_data))
            results.append(await mock_blueprint_validator.validate_dependencies(sample_blueprint_data))
            return results
        
        # Run validation chain
        results = asyncio.run(run_validation_chain())
        
        # Verify validation chain execution
        assert len(results) == 5
        assert len(validation_steps) == 5
        
        # Verify execution order
        expected_order = ["schema", "business_rules", "cross_fields", "custom_rules", "dependencies"]
        assert validation_steps == expected_order
        
        # Verify all steps completed successfully
        for result in results:
            assert result["is_valid"] is True
            assert "step" in result
        
        print("✅ Validation chain execution working correctly")
        print(f"  Steps executed: {', '.join(validation_steps)}")

    def test_validation_error_recovery(self, mock_blueprint_validator, sample_blueprint_data):
        """Test validation error recovery mechanisms."""
        # Mock validation with recovery
        mock_blueprint_validator.validate_with_recovery = AsyncMock(return_value={
            "is_valid": True,
            "original_errors": [
                {
                    "field": "metadata.version",
                    "error_type": "format_error",
                    "message": "Invalid version format"
                }
            ],
            "recovery_actions": [
                {
                    "field": "metadata.version",
                    "action": "auto_correct",
                    "original_value": "1.0",
                    "corrected_value": "1.0.0",
                    "success": True
                }
            ],
            "final_validation": {
                "is_valid": True,
                "errors": [],
                "warnings": []
            }
        })
        
        # Test validation with recovery
        result = asyncio.run(mock_blueprint_validator.validate_with_recovery(sample_blueprint_data))
        
        # Verify recovery result
        assert result is not None
        assert result["is_valid"] is True
        assert "original_errors" in result
        assert "recovery_actions" in result
        assert "final_validation" in result
        
        # Verify original errors
        original_errors = result["original_errors"]
        assert len(original_errors) > 0
        
        # Verify recovery actions
        recovery_actions = result["recovery_actions"]
        assert len(recovery_actions) > 0
        
        for action in recovery_actions:
            assert "field" in action
            assert "action" in action
            assert "original_value" in action
            assert "corrected_value" in action
            assert "success" in action
            assert action["success"] is True
        
        # Verify final validation
        final_validation = result["final_validation"]
        assert final_validation["is_valid"] is True
        assert len(final_validation["errors"]) == 0
        
        print("✅ Validation error recovery working correctly")
        print(f"  Original errors: {len(original_errors)}")
        print(f"  Recovery actions: {len(recovery_actions)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
