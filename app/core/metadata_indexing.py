"""
Metadata indexing utilities for the RAG system.

This module provides utilities for indexing and managing metadata
for vector database operations.
Enhanced with blueprint section hierarchy support.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timezone
from app.models.blueprint_centric import BlueprintSection, DifficultyLevel, UueStage
from app.models.text_node import TextNode, LocusType, UUEStage

logger = logging.getLogger(__name__)


class MetadataIndexingError(Exception):
    """Base exception for metadata indexing operations."""
    pass


class MetadataIndexer:
    """Handles metadata indexing operations for blueprint sections."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)


class MetadataIndexingService:
    """Service for managing metadata indexing operations."""
    
    def __init__(self):
        self.indexer = MetadataIndexer()
        self.logger = logging.getLogger(__name__)
    
    def index_section_metadata(self, section: BlueprintSection) -> Dict[str, Any]:
        """Index metadata for a single section."""
        try:
            return self.indexer.create_section_metadata(section)
        except Exception as e:
            self.logger.error(f"Failed to index section metadata: {e}")
            raise MetadataIndexingError(f"Section metadata indexing failed: {e}")
    
    def index_section_hierarchy(self, sections: List[BlueprintSection]) -> Dict[int, Dict[str, Any]]:
        """Index metadata for a complete section hierarchy."""
        try:
            return self.indexer.create_section_hierarchy_metadata(sections)
        except Exception as e:
            self.logger.error(f"Failed to index section hierarchy: {e}")
            raise MetadataIndexingError(f"Section hierarchy indexing failed: {e}")
    
    def update_section_metadata(self, section: BlueprintSection) -> Dict[str, Any]:
        """Update metadata for an existing section."""
        try:
            metadata = self.indexer.create_section_metadata(section)
            metadata["last_updated"] = datetime.now(timezone.utc).isoformat()
            return metadata
        except Exception as e:
            self.logger.error(f"Failed to update section metadata: {e}")
            raise MetadataIndexingError(f"Section metadata update failed: {e}")
    
    def create_section_metadata(
        self, 
        section: BlueprintSection,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create metadata for a blueprint section.
        
        Args:
            section: BlueprintSection object
            additional_metadata: Additional metadata to include
            
        Returns:
            Dictionary containing section metadata
        """
        try:
            metadata = {
                "section_id": section.id,
                "section_title": section.title,
                "section_description": section.description,
                "blueprint_id": section.blueprint_id,
                "parent_section_id": section.parent_section_id,
                "section_depth": section.depth,
                "section_order": section.order_index,
                "section_difficulty": section.difficulty.value if section.difficulty else DifficultyLevel.BEGINNER.value,
                "section_estimated_time": section.estimated_time_minutes,
                "section_created_at": section.created_at.isoformat() if section.created_at else None,
                "section_updated_at": section.updated_at.isoformat() if section.updated_at else None,
                "metadata_type": "blueprint_section",
                "indexed_at": datetime.now(timezone.utc).isoformat()
            }
            
            if additional_metadata:
                metadata.update(additional_metadata)
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to create section metadata: {e}")
            raise MetadataIndexingError(f"Section metadata creation failed: {e}")
    
    def create_section_hierarchy_metadata(
        self, 
        sections: List[BlueprintSection]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Create metadata for a complete section hierarchy.
        
        Args:
            sections: List of BlueprintSection objects
            
        Returns:
            Dictionary mapping section ID to metadata
        """
        try:
            hierarchy_metadata = {}
            
            # Create section lookup map
            section_map = {section.id: section for section in sections}
            
            for section in sections:
                # Build section path
                section_path = self._build_section_path(section, section_map)
                
                # Create enhanced metadata
                metadata = self.create_section_metadata(section)
                metadata.update({
                    "section_path": section_path,
                    "section_path_string": " > ".join(section_path),
                    "has_children": any(s.parent_section_id == section.id for s in sections),
                    "child_count": len([s for s in sections if s.parent_section_id == section.id]),
                    "is_root": section.parent_section_id is None,
                    "is_leaf": not any(s.parent_section_id == section.id for s in sections)
                })
                
                hierarchy_metadata[section.id] = metadata
            
            return hierarchy_metadata
            
        except Exception as e:
            self.logger.error(f"Failed to create section hierarchy metadata: {e}")
            raise MetadataIndexingError(f"Section hierarchy metadata creation failed: {e}")
    
    def _build_section_path(self, section: BlueprintSection, section_map: Dict[int, BlueprintSection]) -> List[str]:
        """
        Build the full path from root to a section.
        
        Args:
            section: The section to build path for
            section_map: Map of section ID to section object
            
        Returns:
            List of section titles from root to the target section
        """
        path = [section.title]
        current = section
        
        while current.parent_section_id and current.parent_section_id in section_map:
            current = section_map[current.parent_section_id]
            path.insert(0, current.title)
        
        return path
    
    def create_text_node_section_metadata(
        self, 
        node: TextNode,
        section: BlueprintSection,
        section_path: List[str]
    ) -> Dict[str, Any]:
        """
        Create metadata for a TextNode with section information.
        
        Args:
            node: TextNode object
            section: Associated BlueprintSection
            section_path: Path from root to the section
            
        Returns:
            Dictionary containing enhanced node metadata
        """
        try:
            # Start with existing node metadata
            metadata = node.metadata.copy()
            
            # Add section information
            section_metadata = {
                "section_id": section.id,
                "section_title": section.title,
                "section_depth": section.depth,
                "section_path": section_path,
                "section_path_string": " > ".join(section_path),
                "parent_section_id": section.parent_section_id,
                "blueprint_id": section.blueprint_id,
                "section_difficulty": section.difficulty.value if section.difficulty else DifficultyLevel.BEGINNER.value,
                "section_estimated_time": section.estimated_time_minutes,
                "section_order": section.order_index,
                "node_section_mapping": "explicit",
                "metadata_type": "text_node_with_section"
            }
            
            metadata.update(section_metadata)
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to create text node section metadata: {e}")
            raise MetadataIndexingError(f"Text node section metadata creation failed: {e}")
    
    def create_section_search_metadata(
        self, 
        section: BlueprintSection,
        search_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create metadata optimized for section-based search.
        
        Args:
            section: BlueprintSection object
            search_context: Additional search context
            
        Returns:
            Dictionary containing search-optimized metadata
        """
        try:
            metadata = self.create_section_metadata(section)
            
            # Add search-specific fields
            search_metadata = {
                "searchable_content": f"{section.title} {section.description or ''}",
                "search_tags": [
                    section.difficulty.value if section.difficulty else "beginner",
                    f"depth_{section.depth}",
                    f"blueprint_{section.blueprint_id}",
                    "section"
                ],
                "search_priority": self._calculate_search_priority(section),
                "content_type": "blueprint_section",
                "search_indexed": True
            }
            
            metadata.update(search_metadata)
            
            if search_context:
                metadata.update(search_context)
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to create section search metadata: {e}")
            raise MetadataIndexingError(f"Section search metadata creation failed: {e}")
    
    def _calculate_search_priority(self, section: BlueprintSection) -> float:
        """
        Calculate search priority for a section based on various factors.
        
        Args:
            section: BlueprintSection object
            
        Returns:
            Priority score (higher = more important)
        """
        priority = 1.0
        
        # Depth factor (deeper sections might be more specific)
        if section.depth > 0:
            priority += section.depth * 0.1
        
        # Difficulty factor (advanced content might be more valuable)
        if section.difficulty == DifficultyLevel.ADVANCED:
            priority += 0.5
        elif section.difficulty == DifficultyLevel.INTERMEDIATE:
            priority += 0.2
        
        # Time factor (longer sections might be more comprehensive)
        if section.estimated_time_minutes:
            priority += min(section.estimated_time_minutes / 60.0, 1.0) * 0.3
        
        return round(priority, 2)
    
    def merge_section_metadata(
        self, 
        base_metadata: Dict[str, Any],
        section_metadata: Dict[str, Any],
        merge_strategy: str = "override"
    ) -> Dict[str, Any]:
        """
        Merge base metadata with section metadata.
        
        Args:
            base_metadata: Base metadata dictionary
            section_metadata: Section-specific metadata
            merge_strategy: How to handle conflicts ("override", "merge", "preserve")
            
        Returns:
            Merged metadata dictionary
        """
        try:
            if merge_strategy == "override":
                # Section metadata takes precedence
                merged = base_metadata.copy()
                merged.update(section_metadata)
            elif merge_strategy == "merge":
                # Merge arrays and combine other fields
                merged = base_metadata.copy()
                for key, value in section_metadata.items():
                    if key in merged and isinstance(merged[key], list) and isinstance(value, list):
                        merged[key] = merged[key] + value
                    else:
                        merged[key] = value
            elif merge_strategy == "preserve":
                # Base metadata takes precedence
                merged = section_metadata.copy()
                merged.update(base_metadata)
            else:
                raise ValueError(f"Unknown merge strategy: {merge_strategy}")
            
            return merged
            
        except Exception as e:
            self.logger.error(f"Failed to merge section metadata: {e}")
            raise MetadataIndexingError(f"Metadata merge failed: {e}")
    
    def validate_section_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        Validate section metadata for required fields and data types.
        
        Args:
            metadata: Metadata dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            required_fields = [
                "section_id", "section_title", "blueprint_id", 
                "section_depth", "metadata_type"
            ]
            
            for field in required_fields:
                if field not in metadata:
                    self.logger.warning(f"Missing required field: {field}")
                    return False
            
            # Validate data types
            if not isinstance(metadata["section_id"], int):
                self.logger.warning(f"Invalid section_id type: {type(metadata['section_id'])}")
                return False
            
            if not isinstance(metadata["section_title"], str) or not metadata["section_title"].strip():
                self.logger.warning("Invalid section_title")
                return False
            
            if not isinstance(metadata["blueprint_id"], int):
                self.logger.warning(f"Invalid blueprint_id type: {type(metadata['blueprint_id'])}")
                return False
            
            if not isinstance(metadata["section_depth"], int) or metadata["section_depth"] < 0:
                self.logger.warning(f"Invalid section_depth: {metadata['section_depth']}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Metadata validation failed: {e}")
            return False
