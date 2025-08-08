"""
Blueprint Translation Utility

Converts arbitrary blueprint JSON formats into the strict LearningBlueprint Pydantic model.
This allows the API to accept various blueprint formats from different sources while 
maintaining a consistent internal data structure.
"""

import logging
from typing import Dict, Any, List, Optional
from app.models.learning_blueprint import (
    LearningBlueprint, 
    Section, 
    KnowledgePrimitives,
    Proposition,
    Entity,
    Process,
    Relationship,
    Question
)

logger = logging.getLogger(__name__)


class BlueprintTranslationError(Exception):
    """Raised when blueprint translation fails."""
    pass


class BlueprintTranslator:
    """Translates arbitrary blueprint JSON to LearningBlueprint model."""
    
    @staticmethod
    def _is_learning_blueprint_format(blueprint_json: Dict[str, Any]) -> bool:
        """
        Check if the blueprint JSON is already in LearningBlueprint format.
        
        Returns:
            True if already in LearningBlueprint format, False if raw blueprint format
        """
        # Check for LearningBlueprint-specific fields
        learning_blueprint_fields = ['source_id', 'source_title', 'source_type', 'source_summary']
        raw_blueprint_fields = ['id', 'title', 'description', 'user_id']
        
        learning_score = sum(1 for field in learning_blueprint_fields if field in blueprint_json)
        raw_score = sum(1 for field in raw_blueprint_fields if field in blueprint_json)
        
        # If more LearningBlueprint fields than raw fields, assume it's already structured
        return learning_score >= raw_score and learning_score >= 2
    
    @staticmethod
    def translate(blueprint_json: Dict[str, Any], user_id: Optional[str] = None) -> LearningBlueprint:
        """
        Translate arbitrary blueprint JSON to LearningBlueprint model.
        Handles both raw blueprint JSON and already-structured LearningBlueprint format.
        
        Args:
            blueprint_json: The blueprint data in various possible formats
            user_id: Optional user ID to associate with the blueprint
            
        Returns:
            LearningBlueprint object ready for indexing
            
        Raises:
            ValueError: If required data cannot be extracted
        """
        try:
            # Check if already in LearningBlueprint format
            if BlueprintTranslator._is_learning_blueprint_format(blueprint_json):
                logger.info(f"Detected LearningBlueprint format for {blueprint_json.get('source_id', 'unknown')}")
                return BlueprintTranslator._translate_learning_blueprint_format(blueprint_json, user_id)
            else:
                logger.info(f"Detected raw blueprint format for {blueprint_json.get('id', 'unknown')}")
                return BlueprintTranslator._translate_raw_blueprint_format(blueprint_json, user_id)
                
        except Exception as e:
            logger.error(f"Blueprint translation failed: {e}")
            raise BlueprintTranslationError(f"Failed to translate blueprint: {e}")
    
    @staticmethod
    def _translate_learning_blueprint_format(blueprint_json: Dict[str, Any], user_id: Optional[str] = None) -> LearningBlueprint:
        """
        Translate Core API blueprint format to LearningBlueprint model.
        Properly transforms field values to match expected schema.
        
        Args:
            blueprint_json: The blueprint data from Core API
            user_id: Optional user ID to associate with the blueprint
            
        Returns:
            LearningBlueprint object ready for indexing
            
        Raises:
            ValueError: If required data cannot be extracted
        """
        try:
            logger.info(f"Translating Core API blueprint: {blueprint_json.get('source_id', 'unknown')}")
            
            # Create a copy to avoid modifying the original
            transformed = blueprint_json.copy()
            
            # Ensure required top-level fields exist with sensible defaults
            if not transformed.get('source_type'):
                transformed['source_type'] = 'article'
            
            if not isinstance(transformed.get('source_summary'), dict):
                transformed['source_summary'] = {
                    'core_thesis_or_main_argument': transformed.get('description') or transformed.get('source_title', 'Learning material'),
                    'inferred_purpose': f"Educational content: {transformed.get('source_title', 'Unknown')}"
                }
            
            # Transform source_summary from string to dict format
            if isinstance(transformed.get('source_summary'), str):
                transformed['source_summary'] = {
                    'core_thesis_or_main_argument': transformed['source_summary'],
                    'inferred_purpose': f"Learning material: {transformed.get('source_title', 'Unknown')}"
                }
            
            # Transform sections from Core API format to Section objects
            if 'sections' in transformed and isinstance(transformed['sections'], list):
                section_objects = []
                for i, section in enumerate(transformed['sections']):
                    if isinstance(section, dict):
                        section_obj = {
                            'section_id': section.get('id', f"section_{i+1}"),
                            'section_name': section.get('title', section.get('section_name', f"Section {i+1}")),
                            'description': section.get('content', section.get('description', f"Content for section {i+1}")),
                            'parent_section_id': section.get('parent_section_id')
                        }
                        section_objects.append(section_obj)
                transformed['sections'] = section_objects
            elif 'sections' not in transformed or not isinstance(transformed['sections'], list):
                # Provide a minimal default section
                transformed['sections'] = [{
                    'section_id': 'main_content',
                    'section_name': 'Main Content',
                    'description': 'Primary content section',
                    'parent_section_id': None
                }]
            
            # Transform knowledge_primitives from array to KnowledgePrimitives object
            if 'knowledge_primitives' in transformed:
                if isinstance(transformed['knowledge_primitives'], list):
                    # Convert empty array or list to proper KnowledgePrimitives structure
                    transformed['knowledge_primitives'] = {
                        'key_propositions_and_facts': [],
                        'key_entities_and_definitions': [],
                        'described_processes_and_steps': [],
                        'identified_relationships': [],
                        'implicit_and_open_questions': []
                    }
                elif not isinstance(transformed['knowledge_primitives'], dict):
                    # Ensure it's a dict structure
                    transformed['knowledge_primitives'] = {
                        'key_propositions_and_facts': [],
                        'key_entities_and_definitions': [],
                        'described_processes_and_steps': [],
                        'identified_relationships': [],
                        'implicit_and_open_questions': []
                    }
            else:
                # Provide a default empty structure
                transformed['knowledge_primitives'] = {
                    'key_propositions_and_facts': [],
                    'key_entities_and_definitions': [],
                    'described_processes_and_steps': [],
                    'identified_relationships': [],
                    'implicit_and_open_questions': []
                }
            
            # Map user ID if provided
            if user_id:
                transformed["user_id"] = user_id
            
            # Create and validate LearningBlueprint
            return LearningBlueprint(**transformed)
            
        except Exception as e:
            logger.error(f"Blueprint translation failed: {e}")
            raise BlueprintTranslationError(f"Failed to translate blueprint: {e}")
    
    @staticmethod
    def _translate_raw_blueprint_format(blueprint_json: Dict[str, Any], user_id: Optional[str] = None) -> LearningBlueprint:
        """
        Translate raw blueprint JSON to LearningBlueprint model.
        
        Args:
            blueprint_json: The raw blueprint data
            user_id: Optional user ID to associate with the blueprint
            
        Returns:
            LearningBlueprint object ready for indexing
            
        Raises:
            ValueError: If required data cannot be extracted
        """
        try:
            # Step 1: Extract and map basic fields
            translated = BlueprintTranslator._extract_basic_fields(blueprint_json)
            
            # Step 2: Transform sections
            translated["sections"] = BlueprintTranslator._transform_sections(blueprint_json)
            
            # Step 3: Transform knowledge primitives
            translated["knowledge_primitives"] = BlueprintTranslator._transform_knowledge_primitives(blueprint_json)
            
            # Step 4: Validate and create LearningBlueprint
            logger.info(f"Translating blueprint: {translated.get('source_id', 'unknown')}")
            return LearningBlueprint(**translated)
            
        except Exception as e:
            logger.error(f"Blueprint translation failed: {e}")
            raise BlueprintTranslationError(f"Failed to translate blueprint: {e}")
    
    @staticmethod
    def _extract_basic_fields(blueprint_json: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and map basic required fields."""
        
        # Map common field variations to LearningBlueprint schema
        field_mappings = {
            # source_id variations
            "source_id": ["source_id", "id", "blueprint_id", "sourceId"],
            # source_title variations  
            "source_title": ["source_title", "title", "name", "blueprint_title", "sourceTitle"],
            # source_type variations
            "source_type": ["source_type", "type", "blueprint_type", "sourceType", "category"],
        }
        
        translated = {}
        
        # Map basic fields
        for target_field, possible_sources in field_mappings.items():
            value = None
            for source_field in possible_sources:
                if source_field in blueprint_json and blueprint_json[source_field]:
                    value = str(blueprint_json[source_field])
                    break
            
            if value:
                translated[target_field] = value
            else:
                # Provide defaults for required fields
                if target_field == "source_id":
                    translated[target_field] = blueprint_json.get("id", "unknown_blueprint")
                elif target_field == "source_title":
                    translated[target_field] = blueprint_json.get("title", "Untitled Blueprint")
                elif target_field == "source_type":
                    translated[target_field] = "blueprint"
        
        # Handle source_summary (required dict field)
        if "source_summary" in blueprint_json and isinstance(blueprint_json["source_summary"], dict):
            translated["source_summary"] = blueprint_json["source_summary"]
        else:
            # Create minimal source_summary from available fields
            description = blueprint_json.get("description", "")
            summary = blueprint_json.get("summary", "")
            objectives = blueprint_json.get("learning_objectives", [])
            
            translated["source_summary"] = {
                "core_thesis_or_main_argument": description or summary or "Learning material covering various topics.",
                "inferred_purpose": f"Educational content with {len(objectives)} learning objectives." if objectives else "Educational learning material."
            }
        
        return translated
    
    @staticmethod
    def _transform_sections(blueprint_json: Dict[str, Any]) -> List[Section]:
        """Transform sections from various formats to Section models."""
        sections = []
        
        # Handle different section formats
        if "sections" in blueprint_json and isinstance(blueprint_json["sections"], list):
            # Already in sections format
            for i, section_data in enumerate(blueprint_json["sections"]):
                sections.append(BlueprintTranslator._create_section(section_data, i))
                
        elif "content_sections" in blueprint_json and isinstance(blueprint_json["content_sections"], list):
            # Common format: content_sections
            for i, section_data in enumerate(blueprint_json["content_sections"]):
                sections.append(BlueprintTranslator._create_section(section_data, i))
                
        elif "chapters" in blueprint_json and isinstance(blueprint_json["chapters"], list):
            # Book format: chapters
            for i, chapter_data in enumerate(blueprint_json["chapters"]):
                sections.append(BlueprintTranslator._create_section(chapter_data, i))
        
        # If no structured sections found, create a single section from content
        if not sections:
            content = blueprint_json.get("content", blueprint_json.get("text", ""))
            if content:
                sections.append(Section(
                    section_id="main_content",
                    section_name="Main Content", 
                    description="Primary content section",
                    parent_section_id=None
                ))
        
        return sections
    
    @staticmethod
    def _create_section(section_data: Dict[str, Any], index: int) -> Section:
        """Create a Section model from section data."""
        
        # Extract section fields with fallbacks
        section_id = section_data.get("section_id", section_data.get("id", f"section_{index}"))
        section_name = section_data.get("section_name", section_data.get("title", section_data.get("name", f"Section {index + 1}")))
        description = section_data.get("description", section_data.get("content", "")[:100] + "..." if section_data.get("content", "") else "Section content")
        parent_section_id = section_data.get("parent_section_id", section_data.get("parent_id"))
        
        return Section(
            section_id=str(section_id),
            section_name=str(section_name),
            description=str(description),
            parent_section_id=str(parent_section_id) if parent_section_id else None
        )
    
    @staticmethod
    def _transform_knowledge_primitives(blueprint_json: Dict[str, Any]) -> KnowledgePrimitives:
        """Transform knowledge primitives from various formats."""
        
        # If knowledge_primitives already exists and is properly structured
        if "knowledge_primitives" in blueprint_json and isinstance(blueprint_json["knowledge_primitives"], dict):
            kp_data = blueprint_json["knowledge_primitives"]
            return KnowledgePrimitives(
                key_propositions_and_facts=BlueprintTranslator._extract_propositions(kp_data),
                key_entities_and_definitions=BlueprintTranslator._extract_entities(kp_data),
                described_processes_and_steps=BlueprintTranslator._extract_processes(kp_data),
                identified_relationships=BlueprintTranslator._extract_relationships(kp_data),
                implicit_and_open_questions=BlueprintTranslator._extract_questions(kp_data)
            )
        
        # Otherwise, extract from common blueprint fields
        return KnowledgePrimitives(
            key_propositions_and_facts=BlueprintTranslator._extract_propositions_from_content(blueprint_json),
            key_entities_and_definitions=BlueprintTranslator._extract_entities_from_content(blueprint_json),
            described_processes_and_steps=BlueprintTranslator._extract_processes_from_content(blueprint_json),
            identified_relationships=[],  # Default empty
            implicit_and_open_questions=BlueprintTranslator._extract_questions_from_content(blueprint_json)
        )
    
    @staticmethod
    def _extract_propositions(kp_data: Dict[str, Any]) -> List[Proposition]:
        """Extract propositions from knowledge primitives data."""
        propositions = []
        prop_list = kp_data.get("key_propositions_and_facts", [])
        
        for i, prop_data in enumerate(prop_list):
            if isinstance(prop_data, dict):
                propositions.append(Proposition(
                    id=prop_data.get("id", f"prop_{i}"),
                    statement=prop_data.get("statement", ""),
                    supporting_evidence=prop_data.get("supporting_evidence", []),
                    sections=prop_data.get("sections", [])
                ))
        
        return propositions
    
    @staticmethod
    def _extract_propositions_from_content(blueprint_json: Dict[str, Any]) -> List[Proposition]:
        """Extract propositions from general content fields."""
        propositions = []
        
        # Extract from learning objectives
        objectives = blueprint_json.get("learning_objectives", [])
        for i, objective in enumerate(objectives):
            if isinstance(objective, str) and objective.strip():
                propositions.append(Proposition(
                    id=f"objective_{i}",
                    statement=objective,
                    supporting_evidence=[],
                    sections=["main_content"]
                ))
        
        return propositions
    
    @staticmethod
    def _extract_entities(kp_data: Dict[str, Any]) -> List[Entity]:
        """Extract entities from knowledge primitives data."""
        entities = []
        entity_list = kp_data.get("key_entities_and_definitions", [])
        
        for i, entity_data in enumerate(entity_list):
            if isinstance(entity_data, dict):
                # Ensure category is a valid literal value
                category = entity_data.get("category", "Concept")
                if category not in ["Person", "Organization", "Concept", "Place", "Object"]:
                    category = "Concept"  # Default to Concept if invalid
                
                entities.append(Entity(
                    id=entity_data.get("id", f"entity_{i}"),
                    entity=entity_data.get("entity", ""),
                    definition=entity_data.get("definition", ""),
                    category=category,
                    sections=entity_data.get("sections", [])
                ))
        
        return entities
    
    @staticmethod
    def _extract_entities_from_content(blueprint_json: Dict[str, Any]) -> List[Entity]:
        """Extract entities from general content fields."""
        # For now, return empty list - could be enhanced to extract key terms
        return []
    
    @staticmethod
    def _extract_processes(kp_data: Dict[str, Any]) -> List[Process]:
        """Extract processes from knowledge primitives data."""
        processes = []
        process_list = kp_data.get("described_processes_and_steps", [])
        
        for i, process_data in enumerate(process_list):
            if isinstance(process_data, dict):
                processes.append(Process(
                    id=process_data.get("id", f"process_{i}"),
                    process_name=process_data.get("process_name", ""),
                    steps=process_data.get("steps", []),
                    sections=process_data.get("sections", [])
                ))
        
        return processes
    
    @staticmethod
    def _extract_processes_from_content(blueprint_json: Dict[str, Any]) -> List[Process]:
        """Extract processes from general content fields."""
        # For now, return empty list - could be enhanced to extract step-by-step content
        return []
    
    @staticmethod
    def _extract_relationships(kp_data: Dict[str, Any]) -> List[Relationship]:
        """Extract relationships from knowledge primitives data."""
        relationships = []
        rel_list = kp_data.get("identified_relationships", [])
        
        for i, rel_data in enumerate(rel_list):
            if isinstance(rel_data, dict):
                relationships.append(Relationship(
                    id=rel_data.get("id", f"rel_{i}"),
                    relationship_type=rel_data.get("relationship_type", "related"),
                    source_primitive_id=rel_data.get("source_primitive_id", ""),
                    target_primitive_id=rel_data.get("target_primitive_id", ""),
                    description=rel_data.get("description", ""),
                    sections=rel_data.get("sections", [])
                ))
        
        return relationships
    
    @staticmethod
    def _extract_questions(kp_data: Dict[str, Any]) -> List[Question]:
        """Extract questions from knowledge primitives data."""
        questions = []
        question_list = kp_data.get("implicit_and_open_questions", [])
        
        for i, question_data in enumerate(question_list):
            if isinstance(question_data, dict):
                questions.append(Question(
                    id=question_data.get("id", f"question_{i}"),
                    question=question_data.get("question", ""),
                    sections=question_data.get("sections", [])
                ))
        
        return questions
    
    @staticmethod
    def _extract_questions_from_content(blueprint_json: Dict[str, Any]) -> List[Question]:
        """Extract questions from general content fields."""
        # For now, return empty list - could be enhanced to extract questions from content
        return []


# Convenience function
def translate_blueprint(blueprint_json: Dict[str, Any]) -> LearningBlueprint:
    """
    Convenience function to translate blueprint JSON to LearningBlueprint model.
    
    Args:
        blueprint_json: Raw blueprint JSON from integration layer
        
    Returns:
        LearningBlueprint: Validated Pydantic model
    """
    return BlueprintTranslator.translate(blueprint_json)
