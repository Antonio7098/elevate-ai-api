"""
Sequential Generation Workflow using LangGraph.
Implements the workflow: source -> blueprint -> sections -> primitives -> mastery criteria -> questions
With Notes branching off from sections and mastery criteria.
"""

from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from datetime import datetime
import json
import asyncio

from ..langgraph_setup import LangGraphSetup
from ...deconstruction import deconstruct_text, generate_primitives_with_criteria_from_source

class SequentialGenerationState(TypedDict):
    """State for sequential generation workflow"""
    # Core workflow state
    workflow_id: Annotated[str, "Unique workflow identifier"]
    current_step: Annotated[str, "Current step in the workflow"]
    status: Annotated[str, "Current workflow status"]
    started_at: Annotated[str, "Workflow start timestamp"]
    last_updated: Annotated[str, "Last update timestamp"]
    
    # Source content
    source_content: Annotated[str, "Original source material"]
    source_type: Annotated[str, "Type of source material"]
    user_preferences: Annotated[Dict[str, Any], "User preferences and settings"]
    
    # Generated content
    blueprint: Annotated[Dict[str, Any], "Generated learning blueprint"]
    sections: Annotated[List[Dict[str, Any]], "Generated blueprint sections"]
    primitives: Annotated[List[Dict[str, Any]], "Extracted knowledge primitives"]
    mastery_criteria: Annotated[List[Dict[str, Any]], "Generated mastery criteria"]
    questions: Annotated[List[Dict[str, Any]], "Generated assessment questions"]
    notes: Annotated[List[Dict[str, Any]], "Generated study notes"]
    
    # User edits and workflow control
    user_edits: Annotated[List[Dict[str, Any]], "User modifications to content"]
    pending_user_review: Annotated[bool, "Whether waiting for user review"]
    errors: Annotated[List[str], "Any errors encountered"]
    metadata: Annotated[Dict[str, Any], "Additional workflow metadata"]

class SequentialGenerationWorkflow:
    """LangGraph-based sequential generation workflow using functional API"""
    
    def __init__(self):
        self.langgraph_setup = LangGraphSetup()
        self.checkpointer = InMemorySaver()
        
        # Create the workflow using functional API
        self.workflow = self.create_workflow()
    
    def create_workflow(self):
        """Create the sequential generation workflow using functional API"""
        
        @task
        def initialize_workflow(source_content: str, source_type: str, user_preferences: Dict[str, Any] = None) -> SequentialGenerationState:
            """Initialize the workflow with source content"""
            workflow_id = f"seq_gen_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            initial_state = SequentialGenerationState(
                workflow_id=workflow_id,
                current_step="blueprint_generation",
                status="initialized",
                started_at=datetime.utcnow().isoformat(),
                last_updated=datetime.utcnow().isoformat(),
                source_content=source_content,
                source_type=source_type,
                user_preferences=user_preferences or {},
                blueprint={},
                sections=[],
                primitives=[],
                mastery_criteria=[],
                questions=[],
                notes=[],
                user_edits=[],
                pending_user_review=False,
                errors=[],
                metadata={
                    "workflow_type": "sequential_generation",
                    "version": "1.0"
                }
            )
            
            print(f"🚀 Initialized workflow {workflow_id}")
            return initial_state
        
        @task
        async def generate_blueprint(state: SequentialGenerationState) -> SequentialGenerationState:
            """Generate learning blueprint from source content"""
            try:
                print(f"📋 Generating blueprint from source...")
                
                # Use existing deconstruction service
                blueprint_data = await deconstruct_text(
                    state["source_content"],
                    state["source_type"]
                )
                
                state["blueprint"] = blueprint_data
                state["current_step"] = "section_generation"
                state["status"] = "blueprint_generated"
                state["last_updated"] = datetime.utcnow().isoformat()
                
                print(f"✅ Blueprint generated successfully")
                return state
                
            except Exception as e:
                state["errors"].append(f"Blueprint generation error: {str(e)}")
                state["status"] = "error"
                return state
        
        @task
        def generate_sections(state: SequentialGenerationState) -> SequentialGenerationState:
            """Generate blueprint sections"""
            try:
                print(f"📚 Generating blueprint sections...")
                
                # Extract sections from blueprint - handle both dict and Pydantic models
                blueprint_data = state["blueprint"]
                if hasattr(blueprint_data, 'sections'):
                    # Pydantic model
                    sections = blueprint_data.sections
                else:
                    # Dictionary
                    sections = blueprint_data.get("sections", [])
                
                # Process each section
                processed_sections = []
                for section in sections:
                    if hasattr(section, 'section_id'):
                        # Pydantic model
                        processed_section = {
                            "section_id": section.section_id,
                            "title": section.section_name,
                            "content": getattr(section, 'description', ''),
                            "hierarchy_level": 1,
                            "metadata": {}
                        }
                    else:
                        # Dictionary
                        processed_section = {
                            "section_id": section.get("section_id"),
                            "title": section.get("title"),
                            "content": section.get("content"),
                            "hierarchy_level": section.get("hierarchy_level", 1),
                            "metadata": section.get("metadata", {})
                        }
                    processed_sections.append(processed_section)
                
                state["sections"] = processed_sections
                state["current_step"] = "primitive_extraction"
                state["status"] = "sections_generated"
                state["last_updated"] = datetime.utcnow().isoformat()
                
                print(f"✅ Generated {len(processed_sections)} sections")
                return state
                
            except Exception as e:
                state["errors"].append(f"Section generation error: {str(e)}")
                state["status"] = "error"
                return state
        
        @task
        async def extract_primitives(state: SequentialGenerationState) -> SequentialGenerationState:
            """Extract knowledge primitives from sections"""
            try:
                print(f"🧠 Extracting knowledge primitives...")
                
                # Use existing primitive extraction service
                primitives_data = await generate_primitives_with_criteria_from_source(
                    state["source_content"],
                    state["source_type"]
                )
                
                # Extract primitives from the results - handle Pydantic model structure
                if hasattr(primitives_data, 'knowledge_primitives'):
                    # Pydantic model
                    knowledge_primitives = primitives_data.knowledge_primitives
                    if hasattr(knowledge_primitives, 'key_propositions_and_facts'):
                        primitives = knowledge_primitives.key_propositions_and_facts
                    else:
                        primitives = []
                else:
                    # Dictionary
                    primitives = primitives_data.get("knowledge_primitives", {}).get("key_propositions_and_facts", [])
                
                # Process primitives
                processed_primitives = []
                for primitive in primitives:
                    if hasattr(primitive, 'statement'):
                        # Pydantic model
                        processed_primitive = {
                            "primitive_id": getattr(primitive, 'id', ''),
                            "title": primitive.statement,
                            "description": getattr(primitive, 'explanation', ''),
                            "section_id": getattr(primitive, 'sections', [''])[0] if hasattr(primitive, 'sections') and primitive.sections else "",
                            "primitive_type": "concept"
                        }
                    else:
                        # Dictionary
                        processed_primitive = {
                            "primitive_id": primitive.get("primitive_id"),
                            "title": primitive.get("statement", "Knowledge primitive"),
                            "description": primitive.get("explanation", ""),
                            "section_id": primitive.get("sections", [""])[0] if primitive.get("sections") else "",
                            "primitive_type": "concept"
                        }
                    processed_primitives.append(processed_primitive)
                
                state["primitives"] = processed_primitives
                state["current_step"] = "mastery_criteria_generation"
                state["status"] = "primitives_extracted"
                state["last_updated"] = datetime.utcnow().isoformat()
                
                print(f"✅ Extracted {len(processed_primitives)} primitives")
                return state
                
            except Exception as e:
                state["errors"].append(f"Primitive extraction error: {str(e)}")
                state["status"] = "error"
                return state
        
        @task
        def generate_mastery_criteria(state: SequentialGenerationState) -> SequentialGenerationState:
            """Generate mastery criteria for primitives"""
            try:
                print(f"🎯 Generating mastery criteria...")
                
                # Generate mastery criteria for each primitive
                mastery_criteria = []
                for primitive in state["primitives"]:
                    criteria = {
                        "criterion_id": f"criteria_{primitive['primitive_id']}",
                        "primitive_id": primitive["primitive_id"],
                        "title": f"Master {primitive['title']}",
                        "description": f"Demonstrate understanding of {primitive['title']}",
                        "uue_stage": "understand",  # Default to understand stage
                        "difficulty_level": "beginner",
                        "success_criteria": [
                            "Can explain the concept clearly",
                            "Can provide examples",
                            "Can apply in different contexts"
                        ]
                    }
                    mastery_criteria.append(criteria)
                
                state["mastery_criteria"] = mastery_criteria
                state["current_step"] = "question_generation"
                state["status"] = "mastery_criteria_generated"
                state["last_updated"] = datetime.utcnow().isoformat()
                
                print(f"✅ Generated {len(mastery_criteria)} mastery criteria")
                return state
                
            except Exception as e:
                state["errors"].append(f"Mastery criteria generation error: {str(e)}")
                state["status"] = "error"
                return state
        
        @task
        def generate_questions(state: SequentialGenerationState) -> SequentialGenerationState:
            """Generate assessment questions from mastery criteria"""
            try:
                print(f"❓ Generating assessment questions...")
                
                # Generate questions for each mastery criterion
                questions = []
                for criterion in state["mastery_criteria"]:
                    # Generate multiple questions per criterion
                    for i in range(3):  # Generate 3 questions per criterion
                        question = {
                            "question_id": f"q_{criterion['criterion_id']}_{i+1}",
                            "criterion_id": criterion["criterion_id"],
                            "question_text": f"Question about {criterion['title']}",
                            "question_type": "multiple_choice",
                            "difficulty": criterion["difficulty_level"],
                            "options": [
                                "Option A",
                                "Option B", 
                                "Option C",
                                "Option D"
                            ],
                            "correct_answer": 0,
                            "explanation": f"Explanation for question about {criterion['title']}"
                        }
                        questions.append(question)
                
                state["questions"] = questions
                state["current_step"] = "note_generation"
                state["status"] = "questions_generated"
                state["last_updated"] = datetime.utcnow().isoformat()
                
                print(f"✅ Generated {len(questions)} questions")
                return state
                
            except Exception as e:
                state["errors"].append(f"Question generation error: {str(e)}")
                state["status"] = "error"
                return state
        
        @task
        def generate_notes(state: SequentialGenerationState) -> SequentialGenerationState:
            """Generate comprehensive study notes"""
            try:
                print(f"📝 Generating study notes...")
                
                notes = []
                
                # Generate notes for sections
                for section in state["sections"]:
                    section_note = {
                        "note_id": f"note_section_{section['section_id']}",
                        "content_type": "section_summary",
                        "title": f"Summary: {section['title']}",
                        "content": f"Comprehensive summary of {section['title']}",
                        "related_content": {
                            "section_id": section["section_id"],
                            "primitives": [p for p in state["primitives"] if p.get("section_id") == section["section_id"]]
                        }
                    }
                    notes.append(section_note)
                
                # Generate notes for mastery criteria
                for criterion in state["mastery_criteria"]:
                    criterion_note = {
                        "note_id": f"note_criterion_{criterion['criterion_id']}",
                        "content_type": "mastery_guide",
                        "title": f"Mastery Guide: {criterion['title']}",
                        "description": f"Detailed guide to master {criterion['title']}",
                        "related_content": {
                            "criterion_id": criterion["criterion_id"],
                            "primitive_id": criterion["primitive_id"]
                        }
                    }
                    notes.append(criterion_note)
                
                state["notes"] = notes
                state["current_step"] = "complete"
                state["status"] = "notes_generated"
                state["last_updated"] = datetime.utcnow().isoformat()
                
                print(f"✅ Generated {len(notes)} study notes")
                return state
                
            except Exception as e:
                state["errors"].append(f"Note generation error: {str(e)}")
                state["status"] = "error"
                return state
        
        @task
        def check_user_review_needed(state: SequentialGenerationState) -> bool:
            """Check if user review is needed"""
            # For now, always return False (no review needed)
            # In a real implementation, this could check user preferences or workflow settings
            return False
        
        @task
        def wait_for_user_review(state: SequentialGenerationState) -> SequentialGenerationState:
            """Wait for user to review and potentially edit content"""
            try:
                print(f"⏳ Waiting for user review...")
                
                # Set flag to indicate waiting for user input
                state["pending_user_review"] = True
                state["status"] = "waiting_for_user_review"
                state["last_updated"] = datetime.utcnow().isoformat()
                
                # Use interrupt to pause for user input
                user_feedback = interrupt({
                    "action": "Please review the generated content",
                    "current_step": state["current_step"],
                    "content_summary": {
                        "sections": len(state["sections"]),
                        "primitives": len(state["primitives"]),
                        "mastery_criteria": len(state["mastery_criteria"]),
                        "questions": len(state["questions"]),
                        "notes": len(state["notes"])
                    }
                })
                
                # Process user feedback and apply any edits
                if user_feedback and "edits" in user_feedback:
                    state = apply_user_edits(state, user_feedback["edits"])
                
                state["pending_user_review"] = False
                state["status"] = "user_review_completed"
                state["last_updated"] = datetime.utcnow().isoformat()
                
                return state
                
            except Exception as e:
                state["errors"].append(f"User review error: {str(e)}")
                state["status"] = "error"
                return state
        
        @task
        def apply_user_edits(state: SequentialGenerationState, edits: Dict[str, Any]) -> SequentialGenerationState:
            """Apply user edits to the state"""
            try:
                # Apply edits to specific sections
                if "blueprint" in edits:
                    state["blueprint"].update(edits["blueprint"])
                if "sections" in edits:
                    state["sections"] = edits["sections"]
                if "primitives" in edits:
                    state["primitives"] = edits["primitives"]
                if "mastery_criteria" in edits:
                    state["mastery_criteria"] = edits["mastery_criteria"]
                if "questions" in edits:
                    state["questions"] = edits["questions"]
                if "notes" in edits:
                    state["notes"] = edits["notes"]
                
                # Record the edit
                state["user_edits"].append({
                    "edits": edits,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                state["last_updated"] = datetime.utcnow().isoformat()
                return state
                
            except Exception as e:
                state["errors"].append(f"Error applying user edits: {str(e)}")
                return state
        
        @task
        def complete_workflow(state: SequentialGenerationState) -> SequentialGenerationState:
            """Complete the workflow"""
            try:
                print(f"🎉 Completing workflow...")
                
                state["status"] = "completed"
                state["current_step"] = "complete"
                state["last_updated"] = datetime.utcnow().isoformat()
                state["pending_user_review"] = False
                
                # Add completion metadata
                state["metadata"]["completed_at"] = datetime.utcnow().isoformat()
                state["metadata"]["total_sections"] = len(state["sections"])
                state["metadata"]["total_primitives"] = len(state["primitives"])
                state["metadata"]["total_criteria"] = len(state["mastery_criteria"])
                state["metadata"]["total_questions"] = len(state["questions"])
                state["metadata"]["total_notes"] = len(state["notes"])
                
                print(f"✅ Workflow completed successfully!")
                return state
                
            except Exception as e:
                state["errors"].append(f"Completion error: {str(e)}")
                state["status"] = "error"
                return state
        
        # Create the main workflow using entrypoint
        @entrypoint(checkpointer=self.checkpointer)
        async def sequential_generation_workflow(
            inputs: Dict[str, Any],
            *, 
            previous: Optional[SequentialGenerationState] = None
        ) -> SequentialGenerationState:
            """
            Main sequential generation workflow.
            
            Args:
                inputs: Dictionary containing source_content, source_type, and user_preferences
                previous: Previous workflow state (for resumption)
            
            Returns:
                Completed workflow state with all generated content
            """
            
            # Extract inputs
            source_content = inputs.get("source_content", "")
            source_type = inputs.get("source_type", "text")
            user_preferences = inputs.get("user_preferences", {})
            
            # Initialize or resume workflow
            if previous:
                print(f"🔄 Resuming workflow {previous['workflow_id']}")
                state = previous
            else:
                state = await initialize_workflow(source_content, source_type, user_preferences)
            
            # Execute the sequential workflow
            state = await generate_blueprint(state)
            state = await generate_sections(state)
            state = await extract_primitives(state)
            state = await generate_mastery_criteria(state)
            state = await generate_questions(state)
            state = await generate_notes(state)
            
            # Check if user review is needed
            if await check_user_review_needed(state):
                state = await wait_for_user_review(state)
            
            # Complete the workflow
            state = await complete_workflow(state)
            
            # Return final state
            return entrypoint.final(value=state, save=state)
        
        return sequential_generation_workflow
    
    async def start_workflow(self, source_content: str, source_type: str, user_preferences: Dict[str, Any] = None) -> str:
        """Start a new sequential generation workflow"""
        try:
            # Generate unique workflow ID
            workflow_id = f"seq_gen_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Execute the workflow
            config = {"configurable": {"thread_id": workflow_id}}
            result = await self.workflow.ainvoke(
                {
                    "source_content": source_content,
                    "source_type": source_type,
                    "user_preferences": user_preferences or {}
                },
                config=config
            )
            
            print(f"🚀 Started workflow {workflow_id}")
            return workflow_id
            
        except Exception as e:
            print(f"❌ Error starting workflow: {e}")
            raise
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[SequentialGenerationState]:
        """Get the current status of a workflow"""
        try:
            # Retrieve from checkpointer
            config = {"configurable": {"thread_id": workflow_id}}
            
            # Get the current state from the checkpointer (no await needed)
            state_snapshot = self.workflow.get_state(config)
            
            if state_snapshot and hasattr(state_snapshot, 'values'):
                # The state is stored as a dictionary, not a list
                state_values = state_snapshot.values
                if state_values:
                    # Return the state directly (it's a dict, not a list)
                    return state_values
            
            return None
            
        except Exception as e:
            print(f"❌ Error getting workflow status: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def resume_workflow(self, workflow_id: str, user_input: Dict[str, Any] = None) -> SequentialGenerationState:
        """Resume a paused workflow with user input"""
        try:
            config = {"configurable": {"thread_id": workflow_id}}
            
            if user_input:
                # Resume with user input
                result = await self.workflow.ainvoke(user_input, config=config)
            else:
                # Just get current state
                result = await self.workflow.ainvoke({}, config=config)
            
            return result
        except Exception as e:
            print(f"❌ Error resuming workflow: {e}")
            raise
    
    async def apply_user_edit(self, workflow_id: str, edits: Dict[str, Any]) -> bool:
        """Apply user edits to a workflow"""
        try:
            # Resume workflow with edits
            await self.resume_workflow(workflow_id, {"edits": edits})
            return True
        except Exception as e:
            print(f"❌ Error applying user edit: {e}")
            return False
