"""
Blueprint validator for validation operations.

This module provides validation logic for blueprint operations.
"""

from typing import Dict, Any
from app.models.blueprint import BlueprintCreateRequest, BlueprintUpdateRequest, BlueprintType


class BlueprintValidationError(Exception):
    """Custom exception for blueprint validation errors."""
    pass


class BlueprintValidator:
    """Validator class for blueprint operations."""
    
    def __init__(self):
        self.max_title_length = 200
        self.max_description_length = 1000
        self.max_tags_count = 20
        self.max_tag_length = 50
        self.max_metadata_keys = 100
        self.max_metadata_value_length = 1000
    
    async def validate_create_request(self, request: BlueprintCreateRequest) -> None:
        """Validate a blueprint creation request."""
        errors = []
        
        # Validate title
        if not request.title or not request.title.strip():
            errors.append("Title is required")
        elif len(request.title) > self.max_title_length:
            errors.append(f"Title must be {self.max_title_length} characters or less")
        
        # Validate description
        if request.description and len(request.description) > self.max_description_length:
            errors.append(f"Description must be {self.max_description_length} characters or less")
        
        # Validate content
        if not request.content:
            errors.append("Content is required")
        elif not isinstance(request.content, dict):
            errors.append("Content must be a dictionary")
        elif len(request.content) == 0:
            errors.append("Content cannot be empty")
        
        # Validate type
        if not request.type:
            errors.append("Blueprint type is required")
        elif not isinstance(request.type, BlueprintType):
            errors.append("Invalid blueprint type")
        
        # Validate tags
        if request.tags:
            if not isinstance(request.tags, list):
                errors.append("Tags must be a list")
            elif len(request.tags) > self.max_tags_count:
                errors.append(f"Maximum {self.max_tags_count} tags allowed")
            else:
                for tag in request.tags:
                    if not isinstance(tag, str):
                        errors.append("All tags must be strings")
                    elif not tag.strip():
                        errors.append("Tags cannot be empty")
                    elif len(tag) > self.max_tag_length:
                        errors.append(f"Tags must be {self.max_tag_length} characters or less")
        
        # Validate metadata
        if request.metadata:
            if not isinstance(request.metadata, dict):
                errors.append("Metadata must be a dictionary")
            elif len(request.metadata) > self.max_metadata_keys:
                errors.append(f"Maximum {self.max_metadata_keys} metadata keys allowed")
            else:
                for key, value in request.metadata.items():
                    if not isinstance(key, str):
                        errors.append("Metadata keys must be strings")
                    elif not key.strip():
                        errors.append("Metadata keys cannot be empty")
                    
                    if isinstance(value, str) and len(value) > self.max_metadata_value_length:
                        errors.append(f"Metadata values must be {self.max_metadata_value_length} characters or less")
        
        # Validate is_public
        if not isinstance(request.is_public, bool):
            errors.append("is_public must be a boolean")
        
        if errors:
            raise BlueprintValidationError(f"Validation failed: {'; '.join(errors)}")
    
    async def validate_update_request(self, request: BlueprintUpdateRequest) -> None:
        """Validate a blueprint update request."""
        errors = []
        
        # Validate title if provided
        if request.title is not None:
            if not request.title.strip():
                errors.append("Title cannot be empty")
            elif len(request.title) > self.max_title_length:
                errors.append(f"Title must be {self.max_title_length} characters or less")
        
        # Validate description if provided
        if request.description is not None and len(request.description) > self.max_description_length:
            errors.append(f"Description must be {self.max_description_length} characters or less")
        
        # Validate content if provided
        if request.content is not None:
            if not isinstance(request.content, dict):
                errors.append("Content must be a dictionary")
            elif len(request.content) == 0:
                errors.append("Content cannot be empty")
        
        # Validate type if provided
        if request.type is not None and not isinstance(request.type, BlueprintType):
            errors.append("Invalid blueprint type")
        
        # Validate tags if provided
        if request.tags is not None:
            if not isinstance(request.tags, list):
                errors.append("Tags must be a list")
            elif len(request.tags) > self.max_tags_count:
                errors.append(f"Maximum {self.max_tags_count} tags allowed")
            else:
                for tag in request.tags:
                    if not isinstance(tag, str):
                        errors.append("All tags must be strings")
                    elif not tag.strip():
                        errors.append("Tags cannot be empty")
                    elif len(tag) > self.max_tag_length:
                        errors.append(f"Tags must be {self.max_tag_length} characters or less")
        
        # Validate metadata if provided
        if request.metadata is not None:
            if not isinstance(request.metadata, dict):
                errors.append("Metadata must be a dictionary")
            elif len(request.metadata) > self.max_metadata_keys:
                errors.append(f"Maximum {self.max_metadata_keys} metadata keys allowed")
            else:
                for key, value in request.metadata.items():
                    if not isinstance(key, str):
                        errors.append("Metadata keys must be strings")
                    elif not key.strip():
                        errors.append("Metadata keys cannot be empty")
                    
                    if isinstance(value, str) and len(value) > self.max_metadata_value_length:
                        errors.append(f"Metadata values must be {self.max_metadata_value_length} characters or less")
        
        # Validate is_public if provided
        if request.is_public is not None and not isinstance(request.is_public, bool):
            errors.append("is_public must be a boolean")
        
        if errors:
            raise BlueprintValidationError(f"Validation failed: {'; '.join(errors)}")
    
    async def validate_blueprint_content(self, content: Dict[str, Any]) -> None:
        """Validate blueprint content structure."""
        errors = []
        
        if not isinstance(content, dict):
            errors.append("Content must be a dictionary")
            raise BlueprintValidationError(f"Validation failed: {'; '.join(errors)}")
        
        # Check for required content fields
        required_fields = ['sections', 'learning_objectives']
        for field in required_fields:
            if field not in content:
                errors.append(f"Content must contain '{field}' field")
        
        # Validate sections if present
        if 'sections' in content:
            sections = content['sections']
            if not isinstance(sections, list):
                errors.append("Sections must be a list")
            elif len(sections) == 0:
                errors.append("Sections cannot be empty")
            else:
                for i, section in enumerate(sections):
                    if not isinstance(section, dict):
                        errors.append(f"Section {i} must be a dictionary")
                    elif 'title' not in section:
                        errors.append(f"Section {i} must have a title")
                    elif 'content' not in section:
                        errors.append(f"Section {i} must have content")
        
        # Validate learning objectives if present
        if 'learning_objectives' in content:
            objectives = content['learning_objectives']
            if not isinstance(objectives, list):
                errors.append("Learning objectives must be a list")
            elif len(objectives) == 0:
                errors.append("Learning objectives cannot be empty")
            else:
                for i, objective in enumerate(objectives):
                    if not isinstance(objective, str):
                        errors.append(f"Learning objective {i} must be a string")
                    elif not objective.strip():
                        errors.append(f"Learning objective {i} cannot be empty")
        
        if errors:
            raise BlueprintValidationError(f"Validation failed: {'; '.join(errors)}")
