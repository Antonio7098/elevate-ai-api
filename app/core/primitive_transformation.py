"""
Primitive Transformation Service for Core API Compatibility.

This service transforms AI-generated blueprint data into Core API compatible
KnowledgePrimitive and MasteryCriterion formats.
"""

import logging
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.models.learning_blueprint import (
    KnowledgePrimitive, 
    MasteryCriterion,
    LearningBlueprint,
    Proposition,
    Entity,
    Process,
    Question,
    Relationship
)
from app.api.schemas import KnowledgePrimitiveDto, MasteryCriterionDto

logger = logging.getLogger(__name__)


class PrimitiveTransformationService:
    """Service for transforming legacy blueprint data to Core API compatible primitives."""
    
    def __init__(self):
        self.primitive_counter = 0
        self.criterion_counter = 0
    
    def generate_primitive_id(self) -> str:
        """Generate a unique primitive ID."""
        self.primitive_counter += 1
        timestamp = int(datetime.now().timestamp())
        return f"primitive_{timestamp}_{self.primitive_counter}_{str(uuid.uuid4())[:8]}"
    
    def generate_criterion_id(self) -> str:
        """Generate a unique criterion ID."""
        self.criterion_counter += 1
        timestamp = int(datetime.now().timestamp())
        return f"criterion_{timestamp}_{self.criterion_counter}_{str(uuid.uuid4())[:8]}"
    
    def transform_proposition_to_primitive(self, proposition: Proposition) -> KnowledgePrimitive:
        """Transform a Proposition to a Core API compatible KnowledgePrimitive."""
        # Generate mastery criteria for the proposition
        criteria = []
        if proposition.mastery_criteria:
            for mc in proposition.mastery_criteria:
                criterion = MasteryCriterion(
                    criterionId=self.generate_criterion_id(),
                    title=f"Master: {mc.description[:50]}..." if len(mc.description) > 50 else mc.description,
                    description=mc.description,
                    ueeLevel=mc.uee_level.upper(),
                    weight=float(mc.weight),
                    isRequired=True
                )
                criteria.append(criterion)
        else:
            # Default mastery criteria for propositions
            criteria.append(MasteryCriterion(
                criterionId=self.generate_criterion_id(),
                title="Understand the key proposition",
                description="Demonstrate understanding of this factual statement",
                ueeLevel="UNDERSTAND",
                weight=3.0,
                isRequired=True
            ))
        
        return KnowledgePrimitive(
            primitiveId=self.generate_primitive_id(),
            title=proposition.statement[:100] + "..." if len(proposition.statement) > 100 else proposition.statement,
            description=proposition.statement,
            primitiveType="fact",
            difficultyLevel="intermediate",
            estimatedTimeMinutes=5,
            trackingIntensity="NORMAL",
            masteryCriteria=criteria
        )
    
    def transform_entity_to_primitive(self, entity: Entity) -> KnowledgePrimitive:
        """Transform an Entity to a Core API compatible KnowledgePrimitive."""
        # Generate mastery criteria for the entity
        criteria = []
        if entity.mastery_criteria:
            for mc in entity.mastery_criteria:
                criterion = MasteryCriterion(
                    criterionId=self.generate_criterion_id(),
                    title=f"Master: {mc.description[:50]}..." if len(mc.description) > 50 else mc.description,
                    description=mc.description,
                    ueeLevel=mc.uee_level.upper(),
                    weight=float(mc.weight),
                    isRequired=True
                )
                criteria.append(criterion)
        else:
            # Default mastery criteria for entities
            criteria.extend([
                MasteryCriterion(
                    criterionId=self.generate_criterion_id(),
                    title="Define the entity",
                    description=f"Provide a clear definition of {entity.entity}",
                    ueeLevel="UNDERSTAND",
                    weight=3.0,
                    isRequired=True
                ),
                MasteryCriterion(
                    criterionId=self.generate_criterion_id(),
                    title="Apply the concept",
                    description=f"Use {entity.entity} appropriately in context",
                    ueeLevel="USE",
                    weight=2.0,
                    isRequired=True
                )
            ])
        
        return KnowledgePrimitive(
            primitiveId=self.generate_primitive_id(),
            title=entity.entity,
            description=entity.definition,
            primitiveType="concept",
            difficultyLevel="intermediate",
            estimatedTimeMinutes=8,
            trackingIntensity="NORMAL",
            masteryCriteria=criteria
        )
    
    def transform_process_to_primitive(self, process: Process) -> KnowledgePrimitive:
        """Transform a Process to a Core API compatible KnowledgePrimitive."""
        # Generate mastery criteria for the process
        criteria = []
        if process.mastery_criteria:
            for mc in process.mastery_criteria:
                criterion = MasteryCriterion(
                    criterionId=self.generate_criterion_id(),
                    title=f"Master: {mc.description[:50]}..." if len(mc.description) > 50 else mc.description,
                    description=mc.description,
                    ueeLevel=mc.uee_level.upper(),
                    weight=float(mc.weight),
                    isRequired=True
                )
                criteria.append(criterion)
        else:
            # Default mastery criteria for processes
            criteria.extend([
                MasteryCriterion(
                    criterionId=self.generate_criterion_id(),
                    title="Understand the process",
                    description=f"Explain the purpose and flow of {process.process_name}",
                    ueeLevel="UNDERSTAND",
                    weight=2.0,
                    isRequired=True
                ),
                MasteryCriterion(
                    criterionId=self.generate_criterion_id(),
                    title="Execute the process",
                    description=f"Perform the steps of {process.process_name} correctly",
                    ueeLevel="USE",
                    weight=3.0,
                    isRequired=True
                ),
                MasteryCriterion(
                    criterionId=self.generate_criterion_id(),
                    title="Adapt the process",
                    description=f"Modify {process.process_name} for different contexts",
                    ueeLevel="EXPLORE",
                    weight=1.0,
                    isRequired=False
                )
            ])
        
        # Create description from steps
        steps_description = "; ".join(process.steps[:3])  # First 3 steps
        if len(process.steps) > 3:
            steps_description += "..."
        
        return KnowledgePrimitive(
            primitiveId=self.generate_primitive_id(),
            title=process.process_name,
            description=f"Process involving: {steps_description}",
            primitiveType="process",
            difficultyLevel="intermediate",
            estimatedTimeMinutes=12,
            trackingIntensity="NORMAL",
            masteryCriteria=criteria
        )
    
    def transform_blueprint_to_primitives(self, blueprint: LearningBlueprint) -> List[KnowledgePrimitive]:
        """
        Transform a complete LearningBlueprint to Core API compatible KnowledgePrimitive list.
        
        Args:
            blueprint: LearningBlueprint instance with legacy structure
            
        Returns:
            List of Core API compatible KnowledgePrimitive instances
        """
        primitives = []
        
        # Transform propositions (facts)
        for proposition in blueprint.knowledge_primitives.key_propositions_and_facts:
            primitive = self.transform_proposition_to_primitive(proposition)
            primitives.append(primitive)
            logger.debug(f"Transformed proposition to primitive: {primitive.primitiveId}")
        
        # Transform entities (concepts)
        for entity in blueprint.knowledge_primitives.key_entities_and_definitions:
            primitive = self.transform_entity_to_primitive(entity)
            primitives.append(primitive)
            logger.debug(f"Transformed entity to primitive: {primitive.primitiveId}")
        
        # Transform processes
        for process in blueprint.knowledge_primitives.described_processes_and_steps:
            primitive = self.transform_process_to_primitive(process)
            primitives.append(primitive)
            logger.debug(f"Transformed process to primitive: {primitive.primitiveId}")
        
        logger.info(f"Transformed blueprint to {len(primitives)} Core API compatible primitives")
        return primitives
    
    def primitive_to_dto(self, primitive: KnowledgePrimitive) -> KnowledgePrimitiveDto:
        """Convert KnowledgePrimitive to KnowledgePrimitiveDto for API responses."""
        criteria_dtos = [
            MasteryCriterionDto(
                criterionId=mc.criterionId,
                title=mc.title,
                description=mc.description,
                ueeLevel=mc.ueeLevel,
                weight=mc.weight,
                isRequired=mc.isRequired
            )
            for mc in primitive.masteryCriteria
        ]
        
        return KnowledgePrimitiveDto(
            primitiveId=primitive.primitiveId,
            title=primitive.title,
            description=primitive.description,
            primitiveType=primitive.primitiveType,
            difficultyLevel=primitive.difficultyLevel,
            estimatedTimeMinutes=primitive.estimatedTimeMinutes,
            trackingIntensity=primitive.trackingIntensity,
            masteryCriteria=criteria_dtos
        )
    
    def primitives_to_dtos(self, primitives: List[KnowledgePrimitive]) -> List[KnowledgePrimitiveDto]:
        """Convert list of KnowledgePrimitive to list of KnowledgePrimitiveDto."""
        return [self.primitive_to_dto(primitive) for primitive in primitives]
    
    def _create_mastery_criterion_from_dict(self, criterion_dict: dict) -> MasteryCriterion:
        """Create a MasteryCriterion from a dictionary."""
        return MasteryCriterion(
            criterionId=criterion_dict.get('criterionId', self.generate_criterion_id()),
            title=criterion_dict.get('title', 'Untitled Criterion'),
            description=criterion_dict.get('description', ''),
            ueeLevel=criterion_dict.get('ueeLevel', 'UNDERSTAND').upper(),
            weight=float(criterion_dict.get('weight', 3.0)),
            isRequired=criterion_dict.get('isRequired', True),
            difficultyLevel=criterion_dict.get('difficultyLevel', 'intermediate'),
            estimatedTimeMinutes=criterion_dict.get('estimatedTimeMinutes', 5),
            trackingIntensity=criterion_dict.get('trackingIntensity', 'NORMAL')
        )
    
    def _create_primitive_from_dict(self, primitive_dict: dict) -> KnowledgePrimitive:
        """Create a KnowledgePrimitive from a dictionary."""
        # Create mastery criteria if provided
        criteria = []
        if 'masteryCriteria' in primitive_dict:
            for criterion_dict in primitive_dict['masteryCriteria']:
                # Accept either dicts or already-constructed MasteryCriterion objects
                try:
                    if isinstance(criterion_dict, MasteryCriterion):
                        criteria.append(criterion_dict)
                    elif isinstance(criterion_dict, dict):
                        criterion = self._create_mastery_criterion_from_dict(criterion_dict)
                        criteria.append(criterion)
                    else:
                        logger.warning("Unsupported masteryCriteria item type: %s", type(criterion_dict))
                except Exception as e:
                    logger.error("Failed to process mastery criterion item: %s", e)
        
        return KnowledgePrimitive(
            primitiveId=primitive_dict.get('primitiveId', self.generate_primitive_id()),
            title=primitive_dict.get('title', 'Untitled Primitive'),
            description=primitive_dict.get('description', ''),
            primitiveType=primitive_dict.get('primitiveType', 'fact'),
            difficultyLevel=primitive_dict.get('difficultyLevel', 'intermediate'),
            estimatedTimeMinutes=primitive_dict.get('estimatedTimeMinutes', 5),
            trackingIntensity=primitive_dict.get('trackingIntensity', 'NORMAL'),
            masteryCriteria=criteria
        )


# Global transformer instance
primitive_transformer = PrimitiveTransformationService()
