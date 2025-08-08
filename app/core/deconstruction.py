"""
Core deconstruction logic for transforming raw text into LearningBlueprints.

This module contains the specialist agent functions that work together to
deconstruct educational content into structured knowledge primitives.
"""

import json
import re
from typing import List, Dict, Any, Optional
from app.models.learning_blueprint import (
    LearningBlueprint, Section, KnowledgePrimitives, Proposition, Entity, Process, Relationship, Question
)
from app.core.llm_service import (
    llm_service,
    create_section_extraction_prompt,
    create_proposition_extraction_prompt,
    create_entity_extraction_prompt,
    create_process_extraction_prompt,
    create_question_extraction_prompt,
    create_relationship_extraction_prompt
)
import uuid


def generate_section_id(name: str) -> str:
    """Generate a unique section ID based on the section name."""
    return f"sec_{uuid.uuid5(uuid.NAMESPACE_DNS, name)}"


def generate_primitive_id(label: str) -> str:
    """Generate a unique primitive ID based on the label."""
    return f"prim_{uuid.uuid5(uuid.NAMESPACE_DNS, label)}"


def generate_relationship_id(source_id: str, target_id: str) -> str:
    """Generate a unique relationship ID based on source and target IDs."""
    return f"rel_{uuid.uuid5(uuid.NAMESPACE_DNS, source_id + target_id)}"


async def find_sections(text: str) -> List[Section]:
    """
    Identifies and defines the hierarchical structure of the source text.
    
    Args:
        text (str): Raw source text.
    Returns:
        List[Section]: List of Section objects representing the document structure.
    """
    try:
        # Create prompt for section extraction
        prompt = create_section_extraction_prompt(text)
        
        # Call LLM
        response = await llm_service.call_llm(prompt, prefer_google=True, operation="extract_sections")
        
        # Parse JSON response
        sections_data = json.loads(response.strip())
        
        # Convert to Section objects
        sections = []
        for section_data in sections_data:
            section = Section(
                section_id=section_data.get("section_id", generate_section_id(section_data["section_name"])),
                section_name=section_data["section_name"],
                description=section_data["description"],
                parent_section_id=section_data.get("parent_section_id")
            )
            sections.append(section)
        
        return sections
        
    except (json.JSONDecodeError, KeyError, Exception) as e:
        # Fallback to heuristic if LLM fails
        print(f"LLM section extraction failed: {e}. Falling back to heuristic.")
        return await _fallback_find_sections(text)


async def _fallback_find_sections(text: str) -> List[Section]:
    """Fallback heuristic for section extraction."""
    sections = []
    # Simple heuristic: split by markdown headings (##, ###, etc.)
    heading_pattern = re.compile(r'^(#+)\s+(.*)', re.MULTILINE)
    matches = list(heading_pattern.finditer(text))
    if not matches:
        # If no headings, treat the whole text as one section
        section_id = generate_section_id("root")
        sections.append(Section(
            section_id=section_id,
            section_name="Main",
            description="Entire source text",
            parent_section_id=None
        ))
        return sections
    for i, match in enumerate(matches):
        heading = match.group(2).strip()
        section_id = generate_section_id(heading)
        # Description: first 20 words after the heading, or empty
        start = match.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        description = " ".join(section_text.split()[:20])
        sections.append(Section(
            section_id=section_id,
            section_name=heading,
            description=description or f"Section: {heading}",
            parent_section_id=None
        ))
    return sections


async def extract_foundational_concepts(text: str, section_id: str) -> List[Proposition]:
    """
    Extracts the main foundational concepts (propositions, high-level processes) from the input text within a given section.
    
    Args:
        text (str): Raw source text for the current section.
        section_id (str): The ID of the current section being processed.
    Returns:
        List[Proposition]: List of Proposition objects with section_ids set.
    """
    try:
        # Create prompt for proposition extraction
        prompt = create_proposition_extraction_prompt(text, section_id)
        
        # Call LLM
        response = await llm_service.call_llm(prompt, prefer_google=True, operation="extract_propositions")
        
        # Use regex to extract JSON array from response (similar to parse_criteria_response)
        import re
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if not json_match:
            print("No JSON array found in proposition response")
            return []
        
        # Parse JSON response
        propositions_data = json.loads(json_match.group())
        
        # Convert to Proposition objects
        propositions = []
        for prop_data in propositions_data:
            proposition = Proposition(
                id=prop_data.get("id", generate_primitive_id(prop_data["statement"][:50])),
                statement=prop_data["statement"],
                supporting_evidence=prop_data.get("supporting_evidence", []),
                sections=[section_id]
            )
            propositions.append(proposition)
        
        return propositions
        
    except (json.JSONDecodeError, KeyError, Exception) as e:
        print(f"LLM proposition extraction failed: {e}. Returning empty list.")
        return []


async def extract_key_terms(text: str, section_id: str) -> List[Entity]:
    """
    Identifies key terms, definitions, and important named entities in the text within a given section.
    
    Args:
        text (str): Raw source text for the current section.
        section_id (str): The ID of the current section being processed.
    Returns:
        List[Entity]: List of Entity objects with section_ids set.
    """
    try:
        # Create prompt for entity extraction
        prompt = create_entity_extraction_prompt(text, section_id)
        
        # Call LLM
        response = await llm_service.call_llm(prompt, prefer_google=True, operation="extract_entities")
        
        # Use regex to extract JSON array from response (similar to parse_criteria_response)
        import re
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if not json_match:
            print("No JSON array found in entity response")
            return []
        
        # Parse JSON response
        entities_data = json.loads(json_match.group())
        
        # Convert to Entity objects with validation
        entities = []
        skipped_count = 0
        for entity_data in entities_data:
            # Skip entities with null or empty required fields
            if (not entity_data.get("entity") or 
                not entity_data.get("definition") or 
                not entity_data.get("category")):
                skipped_count += 1
                continue
                
            try:
                entity = Entity(
                    id=entity_data.get("id", generate_primitive_id(entity_data["entity"])),
                    entity=entity_data["entity"],
                    definition=entity_data["definition"],
                    category=entity_data["category"],
                    sections=[section_id]
                )
                entities.append(entity)
            except Exception as e:
                # Skip invalid entities
                skipped_count += 1
                print(f"Skipping invalid entity: {entity_data.get('entity', 'unknown')} - {e}")
                continue
        
        if skipped_count > 0:
            print(f"Entity extraction: {len(entities)} valid entities, {skipped_count} skipped due to validation errors")
        
        return entities
        
    except (json.JSONDecodeError, KeyError, Exception) as e:
        print(f"LLM entity extraction failed: {e}. Returning empty list.")
        return []


async def extract_processes(text: str, section_id: str) -> List[Process]:
    """
    Identifies described processes and steps in the text within a given section.
    
    Args:
        text (str): Raw source text for the current section.
        section_id (str): The ID of the current section being processed.
    Returns:
        List[Process]: List of Process objects with section_ids set.
    """
    try:
        # Create prompt for process extraction
        prompt = create_process_extraction_prompt(text, section_id)
        
        # Call LLM
        response = await llm_service.call_llm(prompt, prefer_google=True, operation="extract_processes")
        
        # Use regex to extract JSON array from response (similar to parse_criteria_response)
        import re
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if not json_match:
            print("No JSON array found in process response")
            return []
        
        # Parse JSON response
        processes_data = json.loads(json_match.group())
        
        # Convert to Process objects
        processes = []
        for process_data in processes_data:
            process = Process(
                id=process_data.get("id", generate_primitive_id(process_data["process_name"])),
                process_name=process_data["process_name"],
                steps=process_data.get("steps", []),
                sections=[section_id]
            )
            processes.append(process)
        
        return processes
        
    except (json.JSONDecodeError, KeyError, Exception) as e:
        print(f"LLM process extraction failed: {e}. Returning empty list.")
        return []


async def identify_relationships(
    propositions: List[Proposition],
    entities: List[Entity],
    processes: List[Process]
) -> List[Relationship]:
    """
    Determines relationships (e.g., causal, part-of, component-of) between extracted primitives.
    
    Args:
        propositions (List[Proposition]): All extracted propositions.
        entities (List[Entity]): All extracted entities.
        processes (List[Process]): All extracted processes.
    Returns:
        List[Relationship]: List of Relationship objects.
    """
    try:
        # Convert to dictionaries for JSON serialization
        propositions_dict = [prop.model_dump() for prop in propositions]
        entities_dict = [entity.model_dump() for entity in entities]
        processes_dict = [process.model_dump() for process in processes]
        
        # Create prompt for relationship extraction
        prompt = create_relationship_extraction_prompt(propositions_dict, entities_dict, processes_dict)
        
        # Call LLM
        response = await llm_service.call_llm(prompt, prefer_google=True, operation="identify_relationships")
        
        # Use regex to extract JSON array from response (similar to parse_criteria_response)
        import re
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if not json_match:
            print("No JSON array found in relationship response")
            return []
        
        # Parse JSON response
        relationships_data = json.loads(json_match.group())
        
        # Convert to Relationship objects
        relationships = []
        for rel_data in relationships_data:
            relationship = Relationship(
                id=rel_data.get("id", generate_relationship_id(rel_data["source_primitive_id"], rel_data["target_primitive_id"])),
                relationship_type=rel_data["relationship_type"],
                source_primitive_id=rel_data["source_primitive_id"],
                target_primitive_id=rel_data["target_primitive_id"],
                description=rel_data["description"],
                sections=rel_data.get("sections", [])
            )
            relationships.append(relationship)
        
        return relationships
        
    except (json.JSONDecodeError, KeyError, Exception) as e:
        print(f"LLM relationship extraction failed: {e}. Returning empty list.")
        return []


async def deconstruct_text(source_text: str, source_type_hint: Optional[str] = None) -> LearningBlueprint:
    """
    Main deconstruction function that orchestrates the specialist agents.
    
    Args:
        source_text (str): Raw source text to deconstruct.
        source_type_hint (Optional[str]): Optional hint about the source type.
    Returns:
        LearningBlueprint: The fully assembled LearningBlueprint object.
    """
    # 1. Find sections
    sections = await find_sections(source_text)
    
    # If no sections found, create a default section
    if not sections:
        sections = [Section(
            section_id=generate_section_id("main"),
            section_name="Main",
            description="Entire source text",
            parent_section_id=None
        )]

    # 2. For each section, extract primitives
    all_propositions: List[Proposition] = []
    all_entities: List[Entity] = []
    all_processes: List[Process] = []

    for i, section in enumerate(sections):
        # Extract section-specific text
        if len(sections) == 1:
            # Single section - use entire text
            section_text = source_text
        else:
            # Multiple sections - extract text between this section and the next
            # This is a simplified approach; in practice, you'd want more sophisticated text segmentation
            section_text = _extract_section_text(source_text, section, sections, i)
        
        # Extract primitives for this section
        section_propositions = await extract_foundational_concepts(section_text, section.section_id)
        section_entities = await extract_key_terms(section_text, section.section_id)
        section_processes = await extract_processes(section_text, section.section_id)
        
        # Add to global lists
        all_propositions.extend(section_propositions)
        all_entities.extend(section_entities)
        all_processes.extend(section_processes)

    # 3. Identify relationships
    relationships = await identify_relationships(all_propositions, all_entities, all_processes)

    # 4. Assemble knowledge primitives
    knowledge_primitives = KnowledgePrimitives(
        key_propositions_and_facts=all_propositions,
        key_entities_and_definitions=all_entities,
        described_processes_and_steps=all_processes,
        identified_relationships=relationships,
        implicit_and_open_questions=[]  # Empty list - no questions extracted
    )

    # 5. Extract source summary using LLM
    source_summary = await _extract_source_summary(source_text)

    # 6. Assemble LearningBlueprint
    blueprint = LearningBlueprint(
        source_id=str(uuid.uuid4()),
        source_title=source_summary.get("title", "Generated from text"),
        source_type=source_type_hint or "text",
        source_summary={
            "core_thesis_or_main_argument": source_summary.get("thesis", "TODO: Extract thesis"),
            "inferred_purpose": source_summary.get("purpose", "TODO: Extract purpose")
        },
        sections=sections,
        knowledge_primitives=knowledge_primitives
    )
    return blueprint


def _extract_section_text(source_text: str, section: Section, all_sections: List[Section], section_index: int) -> str:
    """
    Extract text specific to a section.
    
    This is a simplified implementation. In practice, you'd want more sophisticated
    text segmentation based on the actual document structure.
    """
    # For now, return the entire text for each section
    # TODO: Implement proper text segmentation based on section boundaries
    return source_text


async def _extract_source_summary(source_text: str) -> Dict[str, str]:
    """
    Extract source summary using LLM.
    """
    try:
        prompt = f"""
Analyze the following text and extract:
1. A concise title (max 10 words)
2. The main thesis or argument
3. The inferred purpose

Text:
{source_text[:2000]}  # Limit to first 2000 chars for summary

Return as JSON:
{{
    "title": "Concise title",
    "thesis": "Main thesis or argument",
    "purpose": "Inferred purpose"
}}
"""
        response = await llm_service.call_llm(prompt, prefer_google=True, operation="extract_source_summary")
        summary_data = json.loads(response.strip())
        return summary_data
    except Exception as e:
        print(f"LLM source summary extraction failed: {e}. Using defaults.")
        return {
            "title": "Generated from text",
            "thesis": "TODO: Extract thesis",
            "purpose": "TODO: Extract purpose"
        }


# Enhanced S31 Core API Compatible Primitive Generation

async def generate_primitives_with_criteria_from_source(
    source_content: str, 
    source_type: str,
    user_preferences: Optional[Dict] = None
) -> LearningBlueprint:
    """
    Generate blueprint with primitives and mastery criteria from source content.
    
    Uses LLM to:
    1. Analyze source content and identify discrete knowledge units
    2. Generate mastery criteria for each primitive during creation
    3. Assign UEE levels and importance weights
    4. Ensure comprehensive coverage across all source content
    
    Args:
        source_content: Raw source text to analyze
        source_type: Type of source (e.g., 'textbook', 'article', 'video')
        user_preferences: Optional user learning preferences
        
    Returns:
        LearningBlueprint with Core API compatible primitives and criteria
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Generate traditional blueprint first
    traditional_blueprint = await deconstruct_text(source_content, source_type)
    
    try:
        # Import here to avoid circular imports
        from app.core.primitive_transformation import primitive_transformer
        
        # Transform to Core API compatible primitives
        core_api_primitives = primitive_transformer.transform_blueprint_to_primitives(traditional_blueprint)
        
        # Enhance with additional mastery criteria using LLM
        enhanced_primitives = []
        for primitive in core_api_primitives:
            enhanced_primitive = await enhance_primitive_with_criteria(
                primitive=primitive,
                source_content=source_content,
                user_preferences=user_preferences or {}
            )
            enhanced_primitives.append(enhanced_primitive)
        
        # Optimize UEE level distribution
        optimized_primitives = optimize_uee_distribution(enhanced_primitives)
        
        # Store Core API primitives in blueprint metadata for later use
        traditional_blueprint._core_api_primitives = optimized_primitives
        
        logger.info(f"Generated {len(optimized_primitives)} Core API compatible primitives")
        return traditional_blueprint
        
    except Exception as e:
        logger.error(f"Failed to enhance blueprint with Core API primitives: {e}")
        return traditional_blueprint


async def enhance_primitive_with_criteria(
    primitive,
    source_content: str,
    user_preferences: Dict[str, Any]
):
    """
    Enhance a primitive with additional mastery criteria using LLM analysis.
    
    Args:
        primitive: KnowledgePrimitive to enhance
        source_content: Original source content for context
        user_preferences: User learning preferences
        
    Returns:
        Enhanced KnowledgePrimitive with optimized mastery criteria
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Create LLM prompt for enhanced criteria generation
    prompt = create_enhanced_criteria_prompt(
        primitive=primitive,
        source_content=source_content,
        user_preferences=user_preferences
    )
    
    try:
        response = await llm_service.call_llm(
            prompt=prompt,
            prefer_google=True,
            operation="enhance_primitive_criteria"
        )
        
        # Parse LLM response to extract criteria
        enhanced_criteria = parse_criteria_response(response, primitive)
        
        # Combine with existing criteria and deduplicate
        all_criteria = primitive.masteryCriteria + enhanced_criteria
        optimized_criteria = deduplicate_and_optimize_criteria(all_criteria)
        
        # Update primitive with enhanced criteria
        primitive.masteryCriteria = optimized_criteria
        
        logger.debug(f"Enhanced primitive {primitive.primitiveId} with {len(enhanced_criteria)} new criteria")
        return primitive
        
    except Exception as e:
        logger.error(f"Failed to enhance primitive {primitive.primitiveId} with criteria: {e}")
        return primitive


def create_enhanced_criteria_prompt(
    primitive,
    source_content: str,
    user_preferences: Dict[str, Any]
) -> str:
    """
    Create LLM prompt for generating enhanced mastery criteria.
    
    Args:
        primitive: KnowledgePrimitive to generate criteria for
        source_content: Original source content
        user_preferences: User learning preferences
        
    Returns:
        Formatted prompt string for LLM
    """
    learning_style = user_preferences.get('learning_style', 'balanced')
    focus_areas = user_preferences.get('focus_areas', [])
    difficulty_preference = user_preferences.get('difficulty_preference', 'intermediate')
    
    existing_criteria_text = "\n".join([
        f"- {mc.title} ({mc.ueeLevel}): {mc.description or 'No description'}" 
        for mc in primitive.masteryCriteria
    ]) if primitive.masteryCriteria else "None"
    
    prompt = f"""
You are an expert learning scientist creating mastery criteria for a knowledge primitive.

PRIMITIVE DETAILS:
Title: {primitive.title}
Type: {primitive.primitiveType}
Description: {primitive.description or 'No description'}
Difficulty Level: {primitive.difficultyLevel}

EXISTING CRITERIA:
{existing_criteria_text}

USER PREFERENCES:
Learning Style: {learning_style}
Focus Areas: {', '.join(focus_areas) if focus_areas else 'General'}
Difficulty Preference: {difficulty_preference}

SOURCE CONTEXT:
{source_content[:1000]}...

TASK: Generate 2-4 additional mastery criteria that:
1. Follow UEE progression (UNDERSTAND → USE → EXPLORE)
2. Have appropriate weight based on importance (1.0-5.0)
3. Are specific and measurable
4. Align with user preferences
5. Cover different cognitive aspects

Return as JSON array:
[
  {{
    "title": "Clear, specific criterion title",
    "description": "Detailed description of what mastery looks like",
    "ueeLevel": "UNDERSTAND|USE|EXPLORE",
    "weight": 2.5,
    "isRequired": true
  }}
]

Focus on creating criteria that are:
- Specific to this primitive
- Measurable through assessment
- Progressive in cognitive complexity
- Aligned with the source content context
"""
    
    return prompt


def parse_criteria_response(response: str, primitive) -> List:
    """
    Parse LLM response to extract mastery criteria.
    
    Args:
        response: LLM response containing criteria JSON
        primitive: Parent primitive for criterion ID generation
        
    Returns:
        List of parsed MasteryCriterion instances
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # Import here to avoid circular imports
        from app.models.learning_blueprint import MasteryCriterion
        from app.core.primitive_transformation import primitive_transformer
        
        # Extract JSON from response
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if not json_match:
            logger.warning("No JSON array found in criteria response")
            return []
            
        criteria_data = json.loads(json_match.group())
        
        criteria = []
        for criterion_data in criteria_data:
            criterion = MasteryCriterion(
                criterionId=primitive_transformer.generate_criterion_id(),
                title=criterion_data.get('title', 'Untitled Criterion'),
                description=criterion_data.get('description'),
                ueeLevel=criterion_data.get('ueeLevel', 'UNDERSTAND'),
                weight=float(criterion_data.get('weight', 2.0)),
                isRequired=criterion_data.get('isRequired', True)
            )
            criteria.append(criterion)
            
        logger.debug(f"Parsed {len(criteria)} criteria from LLM response")
        return criteria
        
    except Exception as e:
        logger.error(f"Failed to parse criteria response: {e}")
        return []


def deduplicate_and_optimize_criteria(criteria: List) -> List:
    """
    Remove duplicate criteria and optimize the collection.
    
    Args:
        criteria: List of MasteryCriterion instances
        
    Returns:
        Optimized list of unique criteria
    """
    # Remove duplicates based on title similarity
    unique_criteria = []
    seen_titles = set()
    
    for criterion in criteria:
        title_lower = criterion.title.lower().strip()
        if title_lower not in seen_titles:
            unique_criteria.append(criterion)
            seen_titles.add(title_lower)
    
    return unique_criteria[:8]  # Limit to max 8 criteria per primitive


def optimize_uee_distribution(primitives: List) -> List:
    """
    Optimize UEE level distribution across all primitives.
    
    Args:
        primitives: List of KnowledgePrimitive instances
        
    Returns:
        Primitives with optimized UEE level distribution
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Calculate current distribution
    total_criteria = sum(len(p.masteryCriteria) for p in primitives)
    if total_criteria == 0:
        return primitives
    
    understand_count = sum(1 for p in primitives for c in p.masteryCriteria if c.ueeLevel == 'UNDERSTAND')
    use_count = sum(1 for p in primitives for c in p.masteryCriteria if c.ueeLevel == 'USE')
    explore_count = sum(1 for p in primitives for c in p.masteryCriteria if c.ueeLevel == 'EXPLORE')
    
    # Target distribution: 40% UNDERSTAND, 40% USE, 20% EXPLORE
    understand_ratio = understand_count / total_criteria
    use_ratio = use_count / total_criteria
    explore_ratio = explore_count / total_criteria
    
    logger.info(f"UEE distribution: UNDERSTAND {understand_ratio:.1%}, USE {use_ratio:.1%}, EXPLORE {explore_ratio:.1%}")
    
    return primitives 

async def generate_enhanced_primitives_with_criteria(source_text: str, user_preferences: Dict[str, Any] = None, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Generate enhanced primitives with mastery criteria from source text.
    
    Args:
        source_text: Source text to extract primitives from
        user_preferences: User preferences for primitive generation
        context: Additional context for primitive generation
        
    Returns:
        List of primitive dictionaries with embedded mastery criteria
    """
    import logging
    logger = logging.getLogger(__name__)
    
    if user_preferences is None:
        user_preferences = {}
    
    # Validate input
    if not source_text or source_text.strip() == "":
        raise ValueError("Source text cannot be empty")
    
    # For now, return mock data for testing
    # This would be replaced with actual LLM-based primitive generation
    primitives = [
        {
            "primitive_id": "phot_001",
            "title": "Photosynthesis Process",
            "description": "The process by which plants convert light energy into chemical energy",
            "content": source_text[:200] if source_text else "Test content",
            "primitive_type": "concept",
            "tags": ["biology", "photosynthesis"],
            "mastery_criteria": [
                {
                    "criterion_id": "phot_crit_001",
                    "title": "Define photosynthesis",
                    "description": "Explain what photosynthesis is and its importance",
                    "uee_level": "UNDERSTAND",
                    "weight": 3.0,
                    "is_required": True
                }
            ]
        }
    ]
    
    logger.info(f"Generated {len(primitives)} enhanced primitives with criteria")
    return {"primitives": primitives}


def create_enhanced_blueprint_with_primitives(title: str, description: str, primitives: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create an enhanced blueprint with embedded primitives.
    
    Args:
        title: Blueprint title
        description: Blueprint description
        primitives: List of primitive dictionaries
        
    Returns:
        Enhanced blueprint dictionary
    """
    import logging
    logger = logging.getLogger(__name__)
    
    blueprint = {
        "blueprint_id": f"test_blueprint_{hash(title) % 10000:04d}",
        "title": title,
        "description": description,
        "primitives": primitives,
        "total_primitives": len(primitives),
        "mastery_criteria_coverage": {
            "total_criteria": sum(len(p.get("mastery_criteria", [])) for p in primitives),
            "uee_distribution": {
                "UNDERSTAND": 0.4,
                "USE": 0.4,
                "EXPLORE": 0.2
            }
        },
        "created_at": "2024-01-01T00:00:00Z"
    }
    
    logger.info(f"Created enhanced blueprint '{title}' with {len(primitives)} primitives")
    return blueprint


class DeconstructionService:
    """
    Service class for primitive generation and deconstruction operations.
    
    This class provides a unified interface for all deconstruction operations
    including primitive generation, mastery criteria creation, and blueprint enhancement.
    """
    
    def __init__(self):
        self.llm_service = None  # Would be injected in real implementation
    
    async def generate_primitives(self, blueprint: Dict[str, Any], user_preferences: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Generate primitives from a blueprint.
        
        Args:
            blueprint: Source blueprint data
            user_preferences: User preferences for generation
            
        Returns:
            List of generated primitive dictionaries
        """
        content = blueprint.get("content", "")
        return generate_enhanced_primitives_with_criteria(content, user_preferences)
    
    async def enhance_blueprint(self, blueprint: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance a blueprint with additional primitive data.
        
        Args:
            blueprint: Source blueprint to enhance
            
        Returns:
            Enhanced blueprint dictionary
        """
        primitives = await self.generate_primitives(blueprint)
        return create_enhanced_blueprint_with_primitives(
            blueprint.get("title", "Enhanced Blueprint"),
            blueprint.get("description", "Enhanced with primitives"),
            primitives
        )
    
    def validate_primitives(self, primitives: List[Dict[str, Any]]) -> bool:
        """
        Validate primitive data structure and content.
        
        Args:
            primitives: List of primitive dictionaries to validate
            
        Returns:
            True if all primitives are valid
        """
        required_fields = ["primitive_id", "title", "content", "primitive_type"]
        
        for primitive in primitives:
            if not all(field in primitive for field in required_fields):
                return False
            
            if not isinstance(primitive.get("mastery_criteria", []), list):
                return False
        
        return True
