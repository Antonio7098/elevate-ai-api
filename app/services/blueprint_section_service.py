"""
Blueprint Section Service for AI API

This service provides section CRUD operations and hierarchy management,
replacing the folder-based system with blueprint-centric section management.
"""

from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timezone
import logging
import uuid

from ..models.blueprint_centric import (
    BlueprintSection, MasteryCriterion, KnowledgePrimitive,
    UueStage, DifficultyLevel, AssessmentType
)
from ..models.mastery_tracking import (
    UserMasteryPreferences, MasteryThreshold
)


logger = logging.getLogger(__name__)


class BlueprintSectionService:
    """
    Service for managing blueprint sections and their hierarchy.
    
    This service provides CRUD operations for blueprint sections,
    manages hierarchical relationships, and handles section-specific
    content aggregation and mastery tracking.
    """
    
    def __init__(self):
        """Initialize the blueprint section service."""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing BlueprintSectionService")
        
        # In-memory storage for sections (in production, this would be a database)
        self.sections: Dict[str, BlueprintSection] = {}
        self.section_counter = 0
    
    async def create_section(self, data: Dict[str, Any]) -> BlueprintSection:
        """
        Create a new blueprint section.
        
        Args:
            data: Section creation data including title, description, blueprint_id, etc.
            
        Returns:
            Created BlueprintSection object
        """
        try:
            self.logger.info(f"Creating section: {data.get('title', 'Unknown')}")
            
            # Generate unique ID
            section_id = self.section_counter + 1
            self.section_counter += 1
            
            # Calculate depth and order
            parent_section_id = data.get('parent_section_id')
            depth = 0
            if parent_section_id:
                parent = self.sections.get(parent_section_id)
                if parent:
                    depth = parent.depth + 1
                else:
                    self.logger.warning(f"Parent section {parent_section_id} not found, using depth 0")
            
            # Get next order index for this level
            order_index = self._get_next_order_index(data.get('blueprint_id'), parent_section_id)
            
            # Create section
            section = BlueprintSection(
                id=section_id,
                title=data['title'],
                description=data.get('description'),
                blueprint_id=data['blueprint_id'],
                parent_section_id=parent_section_id,
                depth=depth,
                order_index=order_index,
                difficulty=data.get('difficulty', DifficultyLevel.BEGINNER),
                estimated_time_minutes=data.get('estimated_time_minutes'),
                user_id=data['user_id'],
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
            
            # Store section
            self.sections[section_id] = section
            
            # Update parent's children list
            if parent_section_id:
                parent = self.sections.get(parent_section_id)
                if parent:
                    parent.children.append(section)
            
            self.logger.info(f"Created section {section_id} with depth {depth}")
            return section
            
        except Exception as e:
            self.logger.error(f"Error creating section: {e}")
            raise
    
    async def get_section_tree(self, blueprint_id: str) -> Dict[str, Any]:
        """
        Build complete section tree from flat section array.
        
        Args:
            blueprint_id: ID of the blueprint
            
        Returns:
            Hierarchical section tree structure
        """
        try:
            self.logger.info(f"Building section tree for blueprint {blueprint_id}")
            
            # Get all sections for this blueprint
            blueprint_sections = [
                section for section in self.sections.values()
                if section.blueprint_id == int(blueprint_id)
            ]
            
            if not blueprint_sections:
                return {
                    "blueprint_id": blueprint_id,
                    "sections": [],
                    "total_sections": 0,
                    "max_depth": 0
                }
            
            # Build tree structure
            root_sections = []
            section_map = {}
            
            # Create map of sections by ID
            for section in blueprint_sections:
                section_map[section.id] = section
                section.children = []  # Reset children for tree building
            
            # Build parent-child relationships
            for section in blueprint_sections:
                if section.parent_section_id:
                    parent = section_map.get(section.parent_section_id)
                    if parent:
                        parent.children.append(section)
                else:
                    root_sections.append(section)
            
            # Sort sections by order_index
            root_sections.sort(key=lambda x: x.order_index)
            for section in blueprint_sections:
                section.children.sort(key=lambda x: x.order_index)
            
            # Calculate tree statistics
            max_depth = max(section.depth for section in blueprint_sections)
            
            tree = {
                "blueprint_id": blueprint_id,
                "sections": root_sections,
                "total_sections": len(blueprint_sections),
                "max_depth": max_depth,
                "depth_distribution": self._calculate_depth_distribution(blueprint_sections)
            }
            
            self.logger.info(f"Built section tree with {len(blueprint_sections)} sections, max depth {max_depth}")
            return tree
            
        except Exception as e:
            self.logger.error(f"Error building section tree: {e}")
            raise
    
    async def move_section(self, section_id: str, new_parent_id: str | None) -> BlueprintSection:
        """
        Move section to new parent with depth recalculation.
        
        Args:
            section_id: ID of section to move
            new_parent_id: ID of new parent section (None for root level)
            
        Returns:
            Updated BlueprintSection object
        """
        try:
            self.logger.info(f"Moving section {section_id} to parent {new_parent_id}")
            
            section = self.sections.get(section_id)
            if not section:
                raise ValueError(f"Section {section_id} not found")
            
            old_parent_id = section.parent_section_id
            
            # Remove from old parent's children
            if old_parent_id:
                old_parent = self.sections.get(str(old_parent_id))
                if old_parent:
                    old_parent.children = [c for c in old_parent.children if c.id != section_id]
            
            # Update section
            section.parent_section_id = new_parent_id
            section.updated_at = datetime.now(timezone.utc)
            
            # Recalculate depth and order
            if new_parent_id:
                new_parent = self.sections.get(new_parent_id)
                if new_parent:
                    section.depth = new_parent.depth + 1
                else:
                    self.logger.warning(f"New parent {new_parent_id} not found")
                    section.depth = 0
            else:
                section.depth = 0
            
            # Get new order index
            section.order_index = self._get_next_order_index(section.blueprint_id, new_parent_id)
            
            # Add to new parent's children
            if new_parent_id:
                new_parent = self.sections.get(new_parent_id)
                if new_parent:
                    new_parent.children.append(section)
            
            # Update depths of all descendants
            await self._update_descendant_depths(section_id, section.depth)
            
            self.logger.info(f"Moved section {section_id} to depth {section.depth}")
            return section
            
        except Exception as e:
            self.logger.error(f"Error moving section: {e}")
            raise
    
    async def reorder_sections(self, blueprint_id: str, order_data: List[Dict[str, Any]]) -> None:
        """
        Reorder sections within blueprint.
        
        Args:
            blueprint_id: ID of the blueprint
            order_data: List of section order updates
        """
        try:
            self.logger.info(f"Reordering sections for blueprint {blueprint_id}")
            
            for order_item in order_data:
                section_id = order_item['section_id']
                new_order = order_item['order_index']
                
                section = self.sections.get(section_id)
                if section and section.blueprint_id == int(blueprint_id):
                                    section.order_index = new_order
                section.updated_at = datetime.now(timezone.utc)
            
            # Sort children by new order
            for section in self.sections.values():
                if section.blueprint_id == int(blueprint_id):
                    section.children.sort(key=lambda x: x.order_index)
            
            self.logger.info(f"Reordered {len(order_data)} sections")
            
        except Exception as e:
            self.logger.error(f"Error reordering sections: {e}")
            raise
    
    async def get_section_content(self, section_id: str) -> Dict[str, Any]:
        """
        Get aggregated content for a section and its descendants.
        
        Args:
            section_id: ID of the section
            
        Returns:
            Aggregated content information
        """
        try:
            self.logger.info(f"Getting content for section {section_id}")
            
            section = self.sections.get(section_id)
            if not section:
                raise ValueError(f"Section {section_id} not found")
            
            # TODO: Integrate with actual content services
            # This would aggregate primitives, questions, and mastery criteria
            
            content = {
                "section_id": section_id,
                "title": section.title,
                "description": section.description,
                "content_count": 0,
                "mastery_criteria_count": 0,
                "questions_count": 0,
                "primitives_count": 0,
                "children_content": []
            }
            
            # Aggregate children content
            for child in section.children:
                child_content = await self.get_section_content(child.id)
                content["children_content"].append(child_content)
                content["content_count"] += child_content["content_count"]
                content["mastery_criteria_count"] += child_content["mastery_criteria_count"]
                content["questions_count"] += child_content["questions_count"]
                content["primitives_count"] += child_content["primitives_count"]
            
            self.logger.info(f"Retrieved content for section {section_id}")
            return content
            
        except Exception as e:
            self.logger.error(f"Error getting section content: {e}")
            raise
    
    async def get_section_stats(self, section_id: str) -> Dict[str, Any]:
        """
        Get statistics for a section including mastery progress.
        
        Args:
            section_id: ID of the section
            
        Returns:
            Section statistics
        """
        try:
            self.logger.info(f"Getting stats for section {section_id}")
            
            section = self.sections.get(section_id)
            if not section:
                raise ValueError(f"Section {section_id} not found")
            
            # TODO: Integrate with actual mastery tracking services
            # This would calculate real mastery progress and statistics
            
            stats = {
                "section_id": section_id,
                "title": section.title,
                "total_criteria": 0,
                "mastered_criteria": 0,
                "mastery_progress": 0.0,
                "average_difficulty": str(section.difficulty),
                "estimated_completion_time": section.estimated_time_minutes or 0,
                "children_count": len(section.children),
                "depth": section.depth,
                "last_activity": section.updated_at.isoformat() if section.updated_at else None
            }
            
            # Calculate mastery progress
            if stats["total_criteria"] > 0:
                stats["mastery_progress"] = stats["mastered_criteria"] / stats["total_criteria"]
            
            self.logger.info(f"Retrieved stats for section {section_id}")
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting section stats: {e}")
            raise
    
    def _get_next_order_index(self, blueprint_id: str, parent_section_id: Optional[str]) -> int:
        """Get the next available order index for a section."""
        sections_at_level = [
            section for section in self.sections.values()
            if section.blueprint_id == int(blueprint_id) and 
               section.parent_section_id == parent_section_id
        ]
        
        if not sections_at_level:
            return 0
        
        return max(section.order_index for section in sections_at_level) + 1
    
    async def _update_descendant_depths(self, section_id: str, new_depth: int) -> None:
        """Update depths of all descendant sections."""
        section = self.sections.get(section_id)
        if not section:
            return
        
        for child in section.children:
            child.depth = new_depth + 1
            child.updated_at = datetime.now(timezone.utc)
            await self._update_descendant_depths(child.id, child.depth)
    
    def _calculate_depth_distribution(self, sections: List[BlueprintSection]) -> Dict[int, int]:
        """Calculate distribution of sections across depths."""
        distribution = {}
        for section in sections:
            depth = section.depth
            distribution[depth] = distribution.get(depth, 0) + 1
        return distribution

