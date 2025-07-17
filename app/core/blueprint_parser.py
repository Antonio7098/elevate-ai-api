"""
Blueprint parser for transforming LearningBlueprints into TextNodes.

This module handles the parsing and transformation of LearningBlueprint structures
into searchable TextNodes for vector database indexing.
"""

import logging
import hashlib
from typing import List, Dict, Any, Optional
from app.models.learning_blueprint import (
    LearningBlueprint, 
    Section, 
    Proposition, 
    Entity, 
    Process, 
    Relationship, 
    Question
)
from app.models.text_node import (
    TextNode, 
    LocusType, 
    UUEStage, 
    create_text_node_id, 
    calculate_word_count
)

logger = logging.getLogger(__name__)


class BlueprintParserError(Exception):
    """Base exception for blueprint parsing operations."""
    pass


class BlueprintParser:
    """Parser for transforming LearningBlueprints into TextNodes."""
    
    def __init__(self):
        self.chunk_size = 1000  # Maximum words per chunk
        self.overlap_size = 100  # Word overlap between chunks
    
    def parse_blueprint(self, blueprint: LearningBlueprint) -> List[TextNode]:
        """
        Parse a LearningBlueprint and extract all TextNodes.
        
        Args:
            blueprint: The LearningBlueprint to parse
            
        Returns:
            List of TextNodes ready for vector indexing
        """
        try:
            nodes = []
            
            # Generate source text hash
            source_text_hash = self._generate_source_hash(blueprint)
            
            # Parse sections
            nodes.extend(self._parse_sections(blueprint, source_text_hash))
            
            # Parse knowledge primitives
            nodes.extend(self._parse_knowledge_primitives(blueprint, source_text_hash))
            
            # Parse relationships
            nodes.extend(self._parse_relationships(blueprint, source_text_hash))
            
            logger.info(f"Parsed blueprint {blueprint.source_id} into {len(nodes)} TextNodes")
            return nodes
            
        except Exception as e:
            logger.error(f"Failed to parse blueprint {blueprint.source_id}: {e}")
            raise BlueprintParserError(f"Blueprint parsing failed: {e}")
    
    def _generate_source_hash(self, blueprint: LearningBlueprint) -> str:
        """Generate a hash of the source content for deduplication."""
        content = f"{blueprint.source_title}{blueprint.source_type}"
        for section in blueprint.sections:
            content += f"{section.section_name}{section.description}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _parse_sections(self, blueprint: LearningBlueprint, source_text_hash: str) -> List[TextNode]:
        """Parse sections into TextNodes."""
        nodes = []
        
        for section in blueprint.sections:
            # Create section content
            content = f"Section: {section.section_name}\n\n{section.description}"
            
            # Create TextNode for section
            node = TextNode(
                id=create_text_node_id(blueprint.source_id, section.section_id, 0),
                content=content,
                blueprint_id=blueprint.source_id,
                source_text_hash=source_text_hash,
                locus_id=section.section_id,
                locus_type=LocusType.FOUNDATIONAL_CONCEPT,
                locus_title=section.section_name,
                uue_stage=UUEStage.UNDERSTAND,
                chunk_index=0,
                total_chunks=1,
                word_count=calculate_word_count(content),
                pathway_ids=[],
                related_locus_ids=[section.parent_section_id] if section.parent_section_id else [],
                metadata={
                    "section_type": "section",
                    "parent_section_id": section.parent_section_id,
                    "source_title": blueprint.source_title,
                    "source_type": blueprint.source_type
                }
            )
            nodes.append(node)
        
        return nodes
    
    def _parse_knowledge_primitives(self, blueprint: LearningBlueprint, source_text_hash: str) -> List[TextNode]:
        """Parse knowledge primitives into TextNodes."""
        nodes = []
        
        # Parse propositions
        nodes.extend(self._parse_propositions(blueprint, source_text_hash))
        
        # Parse entities
        nodes.extend(self._parse_entities(blueprint, source_text_hash))
        
        # Parse processes
        nodes.extend(self._parse_processes(blueprint, source_text_hash))
        
        # Parse questions
        nodes.extend(self._parse_questions(blueprint, source_text_hash))
        
        return nodes
    
    def _parse_propositions(self, blueprint: LearningBlueprint, source_text_hash: str) -> List[TextNode]:
        """Parse propositions into TextNodes."""
        nodes = []
        
        for prop in blueprint.knowledge_primitives.key_propositions_and_facts:
            # Create proposition content
            content = f"Proposition: {prop.statement}"
            if prop.supporting_evidence:
                content += f"\n\nSupporting Evidence:\n" + "\n".join(f"- {evidence}" for evidence in prop.supporting_evidence)
            
            # Split into chunks if needed
            chunks = self._chunk_content(content, prop.id)
            
            for i, chunk in enumerate(chunks):
                node = TextNode(
                    id=create_text_node_id(blueprint.source_id, prop.id, i),
                    content=chunk,
                    blueprint_id=blueprint.source_id,
                    source_text_hash=source_text_hash,
                    locus_id=prop.id,
                    locus_type=LocusType.FOUNDATIONAL_CONCEPT,
                    locus_title=f"Proposition: {prop.statement[:50]}...",
                    uue_stage=UUEStage.UNDERSTAND,
                    chunk_index=i,
                    total_chunks=len(chunks),
                    word_count=calculate_word_count(chunk),
                    pathway_ids=[],
                    related_locus_ids=prop.sections,
                    metadata={
                        "primitive_type": "proposition",
                        "sections": prop.sections,
                        "source_title": blueprint.source_title,
                        "source_type": blueprint.source_type
                    }
                )
                nodes.append(node)
        
        return nodes
    
    def _parse_entities(self, blueprint: LearningBlueprint, source_text_hash: str) -> List[TextNode]:
        """Parse entities into TextNodes."""
        nodes = []
        
        for entity in blueprint.knowledge_primitives.key_entities_and_definitions:
            # Create entity content
            content = f"Entity: {entity.entity}\nCategory: {entity.category}\n\nDefinition: {entity.definition}"
            
            # Split into chunks if needed
            chunks = self._chunk_content(content, entity.id)
            
            for i, chunk in enumerate(chunks):
                node = TextNode(
                    id=create_text_node_id(blueprint.source_id, entity.id, i),
                    content=chunk,
                    blueprint_id=blueprint.source_id,
                    source_text_hash=source_text_hash,
                    locus_id=entity.id,
                    locus_type=LocusType.KEY_TERM,
                    locus_title=f"Entity: {entity.entity}",
                    uue_stage=UUEStage.UNDERSTAND,
                    chunk_index=i,
                    total_chunks=len(chunks),
                    word_count=calculate_word_count(chunk),
                    pathway_ids=[],
                    related_locus_ids=entity.sections,
                    metadata={
                        "primitive_type": "entity",
                        "entity_category": entity.category,
                        "sections": entity.sections,
                        "source_title": blueprint.source_title,
                        "source_type": blueprint.source_type
                    }
                )
                nodes.append(node)
        
        return nodes
    
    def _parse_processes(self, blueprint: LearningBlueprint, source_text_hash: str) -> List[TextNode]:
        """Parse processes into TextNodes."""
        nodes = []
        
        for process in blueprint.knowledge_primitives.described_processes_and_steps:
            # Create process content
            content = f"Process: {process.process_name}"
            if process.steps:
                content += f"\n\nSteps:\n" + "\n".join(f"{i+1}. {step}" for i, step in enumerate(process.steps))
            
            # Split into chunks if needed
            chunks = self._chunk_content(content, process.id)
            
            for i, chunk in enumerate(chunks):
                node = TextNode(
                    id=create_text_node_id(blueprint.source_id, process.id, i),
                    content=chunk,
                    blueprint_id=blueprint.source_id,
                    source_text_hash=source_text_hash,
                    locus_id=process.id,
                    locus_type=LocusType.USE_CASE,
                    locus_title=f"Process: {process.process_name}",
                    uue_stage=UUEStage.USE,
                    chunk_index=i,
                    total_chunks=len(chunks),
                    word_count=calculate_word_count(chunk),
                    pathway_ids=[],
                    related_locus_ids=process.sections,
                    metadata={
                        "primitive_type": "process",
                        "sections": process.sections,
                        "source_title": blueprint.source_title,
                        "source_type": blueprint.source_type
                    }
                )
                nodes.append(node)
        
        return nodes
    
    def _parse_questions(self, blueprint: LearningBlueprint, source_text_hash: str) -> List[TextNode]:
        """Parse questions into TextNodes."""
        nodes = []
        
        for question in blueprint.knowledge_primitives.implicit_and_open_questions:
            # Create question content
            content = f"Question: {question.question}"
            
            # Split into chunks if needed
            chunks = self._chunk_content(content, question.id)
            
            for i, chunk in enumerate(chunks):
                node = TextNode(
                    id=create_text_node_id(blueprint.source_id, question.id, i),
                    content=chunk,
                    blueprint_id=blueprint.source_id,
                    source_text_hash=source_text_hash,
                    locus_id=question.id,
                    locus_type=LocusType.EXPLORATION,
                    locus_title=f"Question: {question.question[:50]}...",
                    uue_stage=UUEStage.EXPLORE,
                    chunk_index=i,
                    total_chunks=len(chunks),
                    word_count=calculate_word_count(chunk),
                    pathway_ids=[],
                    related_locus_ids=question.sections,
                    metadata={
                        "primitive_type": "question",
                        "sections": question.sections,
                        "source_title": blueprint.source_title,
                        "source_type": blueprint.source_type
                    }
                )
                nodes.append(node)
        
        return nodes
    
    def _parse_relationships(self, blueprint: LearningBlueprint, source_text_hash: str) -> List[TextNode]:
        """Parse relationships into TextNodes."""
        nodes = []
        
        for rel in blueprint.knowledge_primitives.identified_relationships:
            # Create relationship content
            content = f"Relationship: {rel.relationship_type}\n\nFrom: {rel.source_primitive_id}\nTo: {rel.target_primitive_id}\n\nDescription: {rel.description}"
            
            # Split into chunks if needed
            chunks = self._chunk_content(content, rel.id)
            
            for i, chunk in enumerate(chunks):
                node = TextNode(
                    id=create_text_node_id(blueprint.source_id, rel.id, i),
                    content=chunk,
                    blueprint_id=blueprint.source_id,
                    source_text_hash=source_text_hash,
                    locus_id=rel.id,
                    locus_type=LocusType.FOUNDATIONAL_CONCEPT,
                    locus_title=f"Relationship: {rel.relationship_type}",
                    uue_stage=UUEStage.UNDERSTAND,
                    chunk_index=i,
                    total_chunks=len(chunks),
                    word_count=calculate_word_count(chunk),
                    pathway_ids=[rel.id],
                    related_locus_ids=[rel.source_primitive_id, rel.target_primitive_id] + rel.sections,
                    metadata={
                        "primitive_type": "relationship",
                        "relationship_type": rel.relationship_type,
                        "source_primitive_id": rel.source_primitive_id,
                        "target_primitive_id": rel.target_primitive_id,
                        "sections": rel.sections,
                        "source_title": blueprint.source_title,
                        "source_type": blueprint.source_type
                    }
                )
                nodes.append(node)
        
        return nodes
    
    def _chunk_content(self, content: str, locus_id: str) -> List[str]:
        """
        Split content into chunks if it exceeds the maximum size.
        
        Args:
            content: The content to chunk
            locus_id: The locus ID for logging
            
        Returns:
            List of content chunks
        """
        words = content.split()
        
        if len(words) <= self.chunk_size:
            return [content]
        
        chunks = []
        start = 0
        
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            chunk = " ".join(chunk_words)
            chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.overlap_size
            if start >= len(words):
                break
        
        logger.debug(f"Chunked content for {locus_id} into {len(chunks)} chunks")
        return chunks
    
    def get_blueprint_stats(self, blueprint: LearningBlueprint) -> Dict[str, Any]:
        """
        Get statistics about a LearningBlueprint.
        
        Args:
            blueprint: The LearningBlueprint to analyze
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            "source_id": blueprint.source_id,
            "source_title": blueprint.source_title,
            "source_type": blueprint.source_type,
            "sections_count": len(blueprint.sections),
            "propositions_count": len(blueprint.knowledge_primitives.key_propositions_and_facts),
            "entities_count": len(blueprint.knowledge_primitives.key_entities_and_definitions),
            "processes_count": len(blueprint.knowledge_primitives.described_processes_and_steps),
            "relationships_count": len(blueprint.knowledge_primitives.identified_relationships),
            "questions_count": len(blueprint.knowledge_primitives.implicit_and_open_questions),
            "total_primitives": (
                len(blueprint.knowledge_primitives.key_propositions_and_facts) +
                len(blueprint.knowledge_primitives.key_entities_and_definitions) +
                len(blueprint.knowledge_primitives.described_processes_and_steps) +
                len(blueprint.knowledge_primitives.identified_relationships) +
                len(blueprint.knowledge_primitives.implicit_and_open_questions)
            )
        }
        
        return stats 