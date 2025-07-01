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
        
        # Parse JSON response
        propositions_data = json.loads(response.strip())
        
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
        
        # Parse JSON response
        entities_data = json.loads(response.strip())
        
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
        
        # Parse JSON response
        processes_data = json.loads(response.strip())
        
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
        
        # Parse JSON response
        relationships_data = json.loads(response.strip())
        
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