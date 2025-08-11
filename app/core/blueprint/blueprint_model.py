"""
Blueprint model for the blueprint lifecycle system.

This module defines the core Blueprint model and related data structures.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class Blueprint(BaseModel):
    """Core Blueprint model for the blueprint lifecycle system."""
    
    id: str = Field(..., description="Unique identifier for the blueprint")
    name: str = Field(..., description="Name of the blueprint")
    description: Optional[str] = Field(None, description="Description of the blueprint")
    content: str = Field(..., description="Content of the blueprint")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    settings: Dict[str, Any] = Field(default_factory=dict, description="Blueprint settings")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    version: str = Field(default="1.0.0", description="Blueprint version")
    status: str = Field(default="draft", description="Blueprint status")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class BlueprintCreateRequest(BaseModel):
    """Request model for creating a new blueprint."""
    
    name: str = Field(..., description="Name of the blueprint")
    description: Optional[str] = Field(None, description="Description of the blueprint")
    content: str = Field(..., description="Content of the blueprint")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    settings: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Blueprint settings")


class BlueprintUpdateRequest(BaseModel):
    """Request model for updating an existing blueprint."""
    
    name: Optional[str] = Field(None, description="Name of the blueprint")
    description: Optional[str] = Field(None, description="Description of the blueprint")
    content: Optional[str] = Field(None, description="Content of the blueprint")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    settings: Optional[Dict[str, Any]] = Field(None, description="Blueprint settings")


class BlueprintResponse(BaseModel):
    """Response model for blueprint operations."""
    
    blueprint: Blueprint = Field(..., description="The blueprint data")
    message: str = Field(..., description="Response message")
    success: bool = Field(..., description="Operation success status")
