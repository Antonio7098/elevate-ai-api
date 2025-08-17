from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_serializer
from enum import Enum


class BlueprintStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


class BlueprintType(str, Enum):
    LEARNING = "learning"
    ASSESSMENT = "assessment"
    PRACTICE = "practice"
    REVIEW = "review"


class Blueprint(BaseModel):
    id: str = Field(..., description="Unique identifier for the blueprint")
    title: str = Field(..., description="Title of the blueprint")
    description: Optional[str] = Field(None, description="Description of the blueprint")
    content: Dict[str, Any] = Field(..., description="Main content of the blueprint")
    type: BlueprintType = Field(..., description="Type of blueprint")
    status: BlueprintStatus = Field(default=BlueprintStatus.DRAFT, description="Current status")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    version: str = Field(default="1.0.0", description="Version of the blueprint")
    author_id: Optional[str] = Field(None, description="ID of the author")
    is_public: bool = Field(default=False, description="Whether the blueprint is public")
    
    @field_serializer('created_at', 'updated_at')
    def serialize_datetime(self, dt: datetime) -> str:
        return dt.isoformat() if dt else None


class BlueprintCreateRequest(BaseModel):
    title: str = Field(..., description="Title of the blueprint")
    description: Optional[str] = Field(None, description="Description of the blueprint")
    content: Dict[str, Any] = Field(..., description="Main content of the blueprint")
    type: BlueprintType = Field(..., description="Type of blueprint")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    is_public: bool = Field(default=False, description="Whether the blueprint is public")


class BlueprintUpdateRequest(BaseModel):
    title: Optional[str] = Field(None, description="Title of the blueprint")
    description: Optional[str] = Field(None, description="Description of the blueprint")
    content: Optional[Dict[str, Any]] = Field(None, description="Main content of the blueprint")
    type: Optional[BlueprintType] = Field(None, description="Type of blueprint")
    status: Optional[BlueprintStatus] = Field(None, description="Current status")
    tags: Optional[List[str]] = Field(None, description="Tags for categorization")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    is_public: Optional[bool] = Field(None, description="Whether the blueprint is public")


class BlueprintResponse(BaseModel):
    id: str
    title: str
    description: Optional[str]
    content: Dict[str, Any]
    type: BlueprintType
    status: BlueprintStatus
    metadata: Dict[str, Any]
    tags: List[str]
    created_at: datetime
    updated_at: datetime
    version: str
    author_id: Optional[str]
    is_public: bool
