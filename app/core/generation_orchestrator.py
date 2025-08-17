"""
Generation Orchestrator - Sequential Learning Path Generation with User Editing

This service orchestrates the complete generation workflow:
source → blueprint → sections → primitives → mastery criteria → questions
with user editing capabilities between each step.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class GenerationStep(str, Enum):
    """Enumeration of generation steps."""
    SOURCE_ANALYSIS = "source_analysis"
    BLUEPRINT_AND_SECTIONS = "blueprint_and_sections"  # Parallel generation
    PRIMITIVE_EXTRACTION = "primitive_extraction"
    MASTERY_CRITERIA = "mastery_criteria"
    QUESTION_GENERATION = "question_generation"
    NOTE_GENERATION = "note_generation"
    COMPLETE = "complete"


class GenerationStatus(str, Enum):
    """Status of the generation process."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    USER_EDITING = "user_editing"
    READY_FOR_NEXT = "ready_for_next"


class UserEditRequest(BaseModel):
    """Request for user to edit generated content."""
    step: GenerationStep
    content_id: str
    edited_content: Dict[str, Any]
    user_notes: Optional[str] = None


class GenerationProgress(BaseModel):
    """Progress tracking for the generation workflow."""
    session_id: str
    current_step: GenerationStep
    status: GenerationStatus
    completed_steps: List[GenerationStep]
    current_content: Optional[Dict[str, Any]] = None
    user_edits: List[UserEditRequest] = []
    errors: List[str] = []
    started_at: datetime
    last_updated: datetime


class GenerationOrchestrator:
    """
    Orchestrates sequential learning path generation with user editing capabilities.
    
    Implements the workflow:
    source → blueprint → sections → primitives → mastery criteria → questions
    with user editing between each step.
    """
    
    def __init__(self):
        # Track generation progress
        self.generation_sessions: Dict[str, GenerationProgress] = {}
    
    async def start_generation_session(
        self, 
        session_id: str,
        source_content: str,
        source_type: str,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> GenerationProgress:
        """
        Start a new generation session.
        
        Args:
            session_id: Unique identifier for the session
            source_content: Raw source content to process
            source_type: Type of source (e.g., 'textbook', 'article')
            user_preferences: User learning preferences
            
        Returns:
            GenerationProgress tracking the session
        """
        logger.info(f"Starting generation session {session_id}")
        
        progress = GenerationProgress(
            session_id=session_id,
            current_step=GenerationStep.SOURCE_ANALYSIS,
            status=GenerationStatus.READY_FOR_NEXT,
            completed_steps=[GenerationStep.SOURCE_ANALYSIS],
            started_at=datetime.utcnow(),
            last_updated=datetime.utcnow()
        )
        
        # Store source content and preferences
        progress.current_content = {
            "source_content": source_content,
            "source_type": source_type,
            "user_preferences": user_preferences or {},
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
        
        self.generation_sessions[session_id] = progress
        
        logger.info(f"Source analysis complete for session {session_id}")
        
        return progress
    
    async def proceed_to_next_step(self, session_id: str) -> GenerationProgress:
        """
        Move to the next step in the generation workflow.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Updated GenerationProgress
        """
        if session_id not in self.generation_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        progress = self.generation_sessions[session_id]
        
        if progress.status != GenerationStatus.READY_FOR_NEXT:
            raise ValueError(f"Session {session_id} not ready for next step")
        
        # Determine next step
        next_step = self._get_next_step(progress.current_step)
        if next_step is None:
            progress.status = GenerationStatus.COMPLETED
            progress.current_step = GenerationStep.COMPLETE
            return progress
        
        # Execute next step
        if next_step == GenerationStep.BLUEPRINT_AND_SECTIONS:
            # Execute blueprint and sections in parallel
            await self._execute_parallel_step(session_id, next_step)
        else:
            await self._execute_step(session_id, next_step)
        
        return self.generation_sessions[session_id]
    
    async def user_edit_content(
        self, 
        session_id: str, 
        edit_request: UserEditRequest
    ) -> GenerationProgress:
        """
        Process user edits to generated content.
        
        Args:
            session_id: Session identifier
            edit_request: User's edit request
            
        Returns:
            Updated GenerationProgress
        """
        if session_id not in self.generation_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        progress = self.generation_sessions[session_id]
        
        # Validate edit request
        if edit_request.step != progress.current_step:
            raise ValueError(f"Can only edit content for current step: {progress.current_step.value}")
        
        # Apply user edits
        progress.user_edits.append(edit_request)
        progress.status = GenerationStatus.USER_EDITING
        
        # Update the content with user edits
        await self._apply_user_edits(session_id, edit_request)
        
        # Mark as ready for next step
        progress.status = GenerationStatus.READY_FOR_NEXT
        progress.last_updated = datetime.utcnow()
        
        logger.info(f"User edits applied for session {session_id}, step {edit_request.step}")
        
        return progress
    
    async def get_generation_progress(self, session_id: str) -> GenerationProgress:
        """Get current progress for a generation session."""
        if session_id not in self.generation_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        return self.generation_sessions[session_id]
    
    async def _execute_step(self, session_id: str, step: GenerationStep):
        """Execute a specific generation step."""
        progress = self.generation_sessions[session_id]
        progress.current_step = step
        progress.status = GenerationStatus.IN_PROGRESS
        progress.last_updated = datetime.utcnow()
        
        try:
            if step == GenerationStep.BLUEPRINT_AND_SECTIONS:
                await self._execute_parallel_step(session_id, step)
            elif step == GenerationStep.PRIMITIVE_EXTRACTION:
                await self._extract_primitives(session_id)
            elif step == GenerationStep.MASTERY_CRITERIA:
                await self._generate_mastery_criteria(session_id)
            elif step == GenerationStep.QUESTION_GENERATION:
                await self._generate_questions(session_id)
            elif step == GenerationStep.NOTE_GENERATION:
                await self._generate_notes(session_id)
            elif step == GenerationStep.COMPLETE:
                # Mark as completed
                progress.status = GenerationStatus.COMPLETED
                progress.last_updated = datetime.utcnow()
                return
            else:
                raise ValueError(f"Unknown step: {step}")
                
        except Exception as e:
            logger.error(f"Step {step} failed for session {session_id}: {e}")
            progress.status = GenerationStatus.FAILED
            progress.errors.append(f"Step {step} failed: {str(e)}")
            raise
    
    async def _execute_parallel_step(self, session_id: str, step: GenerationStep):
        """Execute blueprint and section generation in parallel."""
        progress = self.generation_sessions[session_id]
        progress.current_step = step
        progress.status = GenerationStatus.IN_PROGRESS
        progress.last_updated = datetime.utcnow()
        
        try:
            # Execute blueprint and sections in parallel
            source_content = progress.current_content["source_content"]
            source_type = progress.current_content["source_type"]
            
            # Use the deconstruction service to create blueprint and sections
            from app.core.deconstruction import deconstruct_text
            blueprint = await deconstruct_text(source_content, source_type)
            
            # Store the blueprint
            if hasattr(blueprint, 'dict'):
                progress.current_content["blueprint"] = blueprint.dict()
            else:
                progress.current_content["blueprint"] = blueprint
            
            # Extract sections from the blueprint
            if hasattr(blueprint, 'sections'):
                sections = blueprint.sections
            elif isinstance(blueprint, dict) and 'sections' in blueprint:
                sections = blueprint['sections']
            else:
                sections = []
            
            # Store sections
            progress.current_content["sections"] = sections
            
            logger.info(f"Parallel blueprint and section generation completed for session {session_id}")
            
            # Mark step as completed
            progress.completed_steps.append(step)
            progress.status = GenerationStatus.READY_FOR_NEXT
            progress.last_updated = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Parallel step {step} failed for session {session_id}: {e}")
            progress.status = GenerationStatus.FAILED
            progress.errors.append(f"Parallel step {step} failed: {str(e)}")
            raise

    async def _create_blueprint(self, session_id: str):
        """Step 2: Create blueprint from analyzed source."""
        progress = self.generation_sessions[session_id]
        
        logger.info(f"Creating blueprint for session {session_id}")
        
        source_content = progress.current_content["source_content"]
        source_type = progress.current_content["source_type"]
        user_preferences = progress.current_content["user_preferences"]
        
        # Generate blueprint using existing deconstruction service
        from app.core.deconstruction import deconstruct_text
        
        blueprint = await deconstruct_text(source_content, source_type)
        
        # Store blueprint in progress (convert to dict if it's a Pydantic model)
        if hasattr(blueprint, 'dict'):
            progress.current_content["blueprint"] = blueprint.dict()
        else:
            progress.current_content["blueprint"] = blueprint
        
        # Mark blueprint creation complete
        progress.completed_steps.append(GenerationStep.BLUEPRINT_CREATION)
        progress.current_step = GenerationStep.BLUEPRINT_CREATION
        progress.status = GenerationStatus.READY_FOR_NEXT
        progress.last_updated = datetime.utcnow()
        
        logger.info(f"Blueprint creation complete for session {session_id}")
    
    async def _generate_sections(self, session_id: str):
        """Step 3: Generate sections from blueprint."""
        progress = self.generation_sessions[session_id]
        
        logger.info(f"Generating sections for session {session_id}")
        
        blueprint_data = progress.current_content["blueprint"]
        
        # Extract sections from blueprint
        sections = blueprint_data.get("sections", [])
        
        # Store sections in progress
        progress.current_content["sections"] = sections
        
        # Mark section generation complete
        progress.completed_steps.append(GenerationStep.SECTION_GENERATION)
        progress.current_step = GenerationStep.SECTION_GENERATION
        progress.status = GenerationStatus.READY_FOR_NEXT
        progress.last_updated = datetime.utcnow()
        
        logger.info(f"Section generation complete for session {session_id}")
    
    async def _extract_primitives(self, session_id: str):
        """Step 4: Extract knowledge primitives from sections."""
        progress = self.generation_sessions[session_id]
        
        logger.info(f"Extracting primitives for session {session_id}")
        
        sections = progress.current_content["sections"]
        blueprint_data = progress.current_content["blueprint"]
        
        # Extract primitives from sections
        primitives = []
        for section in sections:
            section_primitives = await self._extract_primitives_from_section(section, blueprint_data)
            primitives.extend(section_primitives)
        
        # Store primitives in progress
        progress.current_content["primitives"] = primitives
        
        # Mark primitive extraction complete
        progress.completed_steps.append(GenerationStep.PRIMITIVE_EXTRACTION)
        progress.current_step = GenerationStep.PRIMITIVE_EXTRACTION
        progress.status = GenerationStatus.READY_FOR_NEXT
        progress.last_updated = datetime.utcnow()
        
        logger.info(f"Primitive extraction complete for session {session_id}")
    
    async def _extract_primitives_from_section(self, section: Dict, blueprint_data: Dict) -> List[Dict]:
        """Extract primitives from a specific section."""
        # Get the actual knowledge primitive from the blueprint
        knowledge_primitives = blueprint_data.get("knowledge_primitives", {}).get("key_propositions_and_facts", [])
        
        # Find the primitive that matches this section
        section_id = section.section_id if hasattr(section, 'section_id') else 'unknown'
        actual_primitives = []
        
        for kp in knowledge_primitives:
            if section_id in kp.get("sections", []):
                actual_primitives.append({
                    "primitive_id": f"prim_{section_id}_{len(actual_primitives) + 1:03d}",
                    "title": kp.get("statement", f"Key concept from {section.section_name if hasattr(section, 'section_name') else 'section'}"),
                    "description": "Extracted knowledge primitive from source content",
                    "section_id": section_id,
                    "primitive_type": "concept"
                })
        
        # If no actual primitives found, fall back to generic
        if not actual_primitives:
            actual_primitives.append({
                "primitive_id": f"prim_{section_id}_001",
                "title": f"Key concept from {section.section_name if hasattr(section, 'section_name') else 'section'}",
                "description": "Extracted from section content",
                "section_id": section_id,
                "primitive_type": "concept"
            })
        
        return actual_primitives
    
    async def _generate_mastery_criteria(self, session_id: str):
        """Step 5: Generate mastery criteria for primitives."""
        progress = self.generation_sessions[session_id]
        
        logger.info(f"Generating mastery criteria for session {session_id}")
        
        primitives = progress.current_content["primitives"]
        
        # Generate mastery criteria for each primitive
        criteria = []
        for primitive in primitives:
            primitive_criteria = await self._generate_criteria_for_primitive(primitive)
            criteria.extend(primitive_criteria)
        
        # Store mastery criteria in progress
        progress.current_content["mastery_criteria"] = criteria
        
        # Mark mastery criteria generation complete
        progress.completed_steps.append(GenerationStep.MASTERY_CRITERIA)
        progress.current_step = GenerationStep.MASTERY_CRITERIA
        progress.status = GenerationStatus.READY_FOR_NEXT
        progress.last_updated = datetime.utcnow()
        
        logger.info(f"Mastery criteria generation complete for session {session_id}")
    
    async def _generate_criteria_for_primitive(self, primitive: Dict) -> List[Dict]:
        """Generate mastery criteria for a specific primitive."""
        # This would integrate with your existing mastery criteria service
        # For now, return mock data
        return [
            {
                "criterion_id": f"crit_{primitive.get('primitive_id', 'unknown')}_001",
                "title": f"Understand {primitive.get('title', 'concept')}",
                "description": f"Demonstrate understanding of {primitive.get('title', 'concept')}",
                "uee_level": "UNDERSTAND",
                "weight": 2.0,
                "primitive_id": primitive.get('primitive_id')
            },
            {
                "criterion_id": f"crit_{primitive.get('primitive_id', 'unknown')}_002",
                "title": f"Apply {primitive.get('title', 'concept')}",
                "description": f"Apply {primitive.get('title', 'concept')} in practice",
                "uee_level": "USE",
                "weight": 3.0,
                "primitive_id": primitive.get('primitive_id')
            }
        ]
    
    async def _generate_questions(self, session_id: str):
        """Step 6: Generate questions from mastery criteria."""
        progress = self.generation_sessions[session_id]
        
        logger.info(f"Generating questions for session {session_id}")
        
        mastery_criteria = progress.current_content["mastery_criteria"]
        
        # Generate questions for each mastery criterion
        questions = []
        for criterion in mastery_criteria:
            criterion_questions = await self._generate_questions_for_criterion(criterion)
            questions.extend(criterion_questions)
        
        # Store questions in progress
        progress.current_content["questions"] = questions
        
        # Mark question generation complete
        progress.completed_steps.append(GenerationStep.QUESTION_GENERATION)
        progress.current_step = GenerationStep.QUESTION_GENERATION
        progress.status = GenerationStatus.READY_FOR_NEXT
        progress.last_updated = datetime.utcnow()
        
        logger.info(f"Question generation complete for session {session_id}")
    
    async def _generate_questions_for_criterion(self, criterion: Dict) -> List[Dict]:
        """Generate questions for a specific mastery criterion."""
        # This would integrate with your existing question generation service
        # For now, return mock data
        return [
            {
                "question_id": f"q_{criterion.get('criterion_id', 'unknown')}_001",
                "question_text": f"What is {criterion.get('title', 'this concept')}?",
                "answer": "Sample answer",
                "explanation": "Sample explanation",
                "criterion_id": criterion.get('criterion_id'),
                "difficulty": "BEGINNER"
            }
        ]
    
    async def _generate_notes(self, session_id: str):
        """Step 7: Generate notes from the complete learning path."""
        progress = self.generation_sessions[session_id]
        
        logger.info(f"Generating notes for session {session_id}")
        
        # Generate comprehensive notes using all generated content
        notes = await self._create_comprehensive_notes(progress.current_content)
        
        # Store notes in progress
        progress.current_content["notes"] = notes
        
        # Mark note generation complete
        progress.completed_steps.append(GenerationStep.NOTE_GENERATION)
        progress.current_step = GenerationStep.NOTE_GENERATION
        progress.status = GenerationStatus.READY_FOR_NEXT
        progress.last_updated = datetime.utcnow()
        
        logger.info(f"Note generation complete for session {session_id}")
    
    async def _create_comprehensive_notes(self, content: Dict) -> Dict:
        """Create comprehensive notes from all generated content."""
        # This would integrate with your existing note generation service
        # For now, return mock data
        return {
            "note_id": "note_comprehensive_001",
            "title": "Comprehensive Learning Notes",
            "content": "Generated notes covering all sections and concepts",
            "sections_covered": len(content.get("sections", [])),
            "primitives_covered": len(content.get("primitives", [])),
            "criteria_covered": len(content.get("mastery_criteria", []))
        }
    
    async def _apply_user_edits(self, session_id: str, edit_request: UserEditRequest):
        """Apply user edits to the current content."""
        progress = self.generation_sessions[session_id]
        
        # Apply edits based on the step
        if edit_request.step == GenerationStep.BLUEPRINT_CREATION:
            blueprint = progress.current_content["blueprint"]
            if isinstance(blueprint, dict):
                blueprint.update(edit_request.edited_content)
            else:
                # Convert to dict, update, and store back
                blueprint_dict = blueprint.dict() if hasattr(blueprint, 'dict') else dict(blueprint)
                blueprint_dict.update(edit_request.edited_content)
                progress.current_content["blueprint"] = blueprint_dict
        elif edit_request.step == GenerationStep.SECTION_GENERATION:
            # Find and update the specific section
            for i, section in enumerate(progress.current_content["sections"]):
                section_id = section.section_id if hasattr(section, 'section_id') else section.get("section_id", None)
                if section_id == edit_request.content_id:
                    progress.current_content["sections"][i].update(edit_request.edited_content)
                    break
        elif edit_request.step == GenerationStep.PRIMITIVE_EXTRACTION:
            # Find and update the specific primitive
            for i, primitive in enumerate(progress.current_content["primitives"]):
                if primitive.get("primitive_id") == edit_request.content_id:
                    progress.current_content["primitives"][i].update(edit_request.edited_content)
                    break
        elif edit_request.step == GenerationStep.MASTERY_CRITERIA:
            # Find and update the specific criterion
            for i, criterion in enumerate(progress.current_content["mastery_criteria"]):
                if criterion.get("criterion_id") == edit_request.content_id:
                    progress.current_content["mastery_criteria"][i].update(edit_request.edited_content)
                    break
        elif edit_request.step == GenerationStep.QUESTION_GENERATION:
            # Find and update the specific question
            for i, question in enumerate(progress.current_content["questions"]):
                if question.get("question_id") == edit_request.content_id:
                    progress.current_content["questions"][i].update(edit_request.edited_content)
                    break
        
        logger.info(f"Applied user edits for {edit_request.step} in session {session_id}")
    
    def _get_next_step(self, current_step: GenerationStep) -> Optional[GenerationStep]:
        """Get the next step in the generation workflow."""
        step_order = [
            GenerationStep.SOURCE_ANALYSIS,
            GenerationStep.BLUEPRINT_AND_SECTIONS,  # Parallel generation
            GenerationStep.PRIMITIVE_EXTRACTION,
            GenerationStep.MASTERY_CRITERIA,
            GenerationStep.QUESTION_GENERATION,
            GenerationStep.NOTE_GENERATION,
            GenerationStep.COMPLETE
        ]
        
        try:
            current_index = step_order.index(current_step)
            if current_index + 1 < len(step_order):
                return step_order[current_index + 1]
        except ValueError:
            pass
        
        return None
    
    async def get_complete_learning_path(self, session_id: str) -> Dict[str, Any]:
        """Get the complete generated learning path."""
        if session_id not in self.generation_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        progress = self.generation_sessions[session_id]
        
        if progress.status != GenerationStatus.COMPLETED:
            raise ValueError(f"Session {session_id} not complete")
        
        return {
            "session_id": session_id,
            "status": "complete",
            "generated_at": progress.started_at.isoformat(),
            "completed_at": progress.last_updated.isoformat(),
            "content": progress.current_content,
            "user_edits": [edit.dict() for edit in progress.user_edits]
        }


# Global instance
generation_orchestrator = GenerationOrchestrator()
