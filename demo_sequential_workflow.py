#!/usr/bin/env python3
"""
Demo: Sequential Generation Workflow with User Editing

This script demonstrates the new sequential generation workflow:
source → blueprint → sections → primitives → mastery criteria → questions

Each step builds on the previous one, and users can edit content between steps.
"""

import asyncio
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock the orchestrator for demo purposes
class MockGenerationOrchestrator:
    """Mock orchestrator for demonstration purposes."""
    
    def __init__(self):
        self.sessions = {}
        self.step_order = [
            "source_analysis",
            "blueprint_creation", 
            "section_generation",
            "primitive_extraction",
            "mastery_criteria",
            "question_generation",
            "note_generation",
            "complete"
        ]
    
    async def start_generation_session(self, session_id, source_content, source_type, user_preferences):
        """Start a new generation session."""
        logger.info(f"🚀 Starting generation session: {session_id}")
        
        session = {
            "session_id": session_id,
            "current_step": "source_analysis",
            "status": "in_progress",
            "completed_steps": [],
            "current_content": {
                "source_content": source_content,
                "source_type": source_type,
                "user_preferences": user_preferences or {},
                "analysis_timestamp": datetime.utcnow().isoformat()
            },
            "user_edits": [],
            "errors": [],
            "started_at": datetime.utcnow(),
            "last_updated": datetime.utcnow()
        }
        
        self.sessions[session_id] = session
        
        # Complete source analysis
        await self._complete_step(session_id, "source_analysis")
        
        return session
    
    async def proceed_to_next_step(self, session_id):
        """Move to the next step in the workflow."""
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        current_step = session["current_step"]
        next_step = self._get_next_step(current_step)
        
        if next_step:
            logger.info(f"⏭️  Moving from {current_step} to {next_step}")
            await self._complete_step(session_id, next_step)
        
        return self.sessions[session_id]
    
    async def user_edit_content(self, session_id, edit_request):
        """Process user edits to generated content."""
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        logger.info(f"✏️  User editing {edit_request['step']} content")
        
        # Apply edits
        session["user_edits"].append(edit_request)
        session["status"] = "ready_for_next"
        session["last_updated"] = datetime.utcnow()
        
        return session
    
    async def _complete_step(self, session_id, step):
        """Complete a generation step."""
        session = self.sessions[session_id]
        
        logger.info(f"✅ Completing step: {step}")
        
        # Generate content for this step
        if step == "blueprint_creation":
            session["current_content"]["blueprint"] = self._generate_mock_blueprint()
        elif step == "section_generation":
            session["current_content"]["sections"] = self._generate_mock_sections()
        elif step == "primitive_extraction":
            session["current_content"]["primitives"] = self._generate_mock_primitives()
        elif step == "mastery_criteria":
            session["current_content"]["mastery_criteria"] = self._generate_mock_criteria()
        elif step == "question_generation":
            session["current_content"]["questions"] = self._generate_mock_questions()
        elif step == "note_generation":
            session["current_content"]["notes"] = self._generate_mock_notes()
        
        # Update session state
        session["completed_steps"].append(step)
        session["current_step"] = step
        session["status"] = "ready_for_next"
        session["last_updated"] = datetime.utcnow()
        
        if step == "complete":
            session["status"] = "completed"
    
    def _get_next_step(self, current_step):
        """Get the next step in the workflow."""
        try:
            current_index = self.step_order.index(current_step)
            if current_index + 1 < len(self.step_order):
                return self.step_order[current_index + 1]
        except ValueError:
            pass
        return None
    
    def _generate_mock_blueprint(self):
        """Generate mock blueprint content."""
        return {
            "blueprint_id": "bp_001",
            "title": "Photosynthesis Fundamentals",
            "description": "Core concepts of photosynthesis",
            "content": "Mock blueprint content...",
            "type": "learning"
        }
    
    def _generate_mock_sections(self):
        """Generate mock sections."""
        return [
            {
                "section_id": "sec_001",
                "title": "Introduction to Photosynthesis",
                "description": "Basic overview of the process"
            },
            {
                "section_id": "sec_002", 
                "title": "Light Reactions",
                "description": "How light energy is captured"
            }
        ]
    
    def _generate_mock_primitives(self):
        """Generate mock knowledge primitives."""
        return [
            {
                "primitive_id": "prim_001",
                "title": "Photosynthesis Definition",
                "description": "Process of converting light to chemical energy",
                "section_id": "sec_001",
                "primitive_type": "concept"
            },
            {
                "primitive_id": "prim_002",
                "title": "Chlorophyll Function",
                "description": "Role of chlorophyll in light absorption",
                "section_id": "sec_002",
                "primitive_type": "fact"
            }
        ]
    
    def _generate_mock_criteria(self):
        """Generate mock mastery criteria."""
        return [
            {
                "criterion_id": "crit_001",
                "title": "Define Photosynthesis",
                "description": "Explain what photosynthesis is",
                "uee_level": "UNDERSTAND",
                "weight": 2.0,
                "primitive_id": "prim_001"
            },
            {
                "criterion_id": "crit_002",
                "title": "Apply Photosynthesis Knowledge",
                "description": "Use photosynthesis concepts in examples",
                "uee_level": "USE",
                "weight": 3.0,
                "primitive_id": "prim_001"
            }
        ]
    
    def _generate_mock_questions(self):
        """Generate mock questions."""
        return [
            {
                "question_id": "q_001",
                "question_text": "What is photosynthesis?",
                "answer": "Process of converting light to chemical energy",
                "criterion_id": "crit_001"
            },
            {
                "question_id": "q_002",
                "question_text": "How do plants use photosynthesis?",
                "answer": "To produce food and oxygen",
                "criterion_id": "crit_002"
            }
        ]
    
    def _generate_mock_notes(self):
        """Generate mock notes."""
        return {
            "note_id": "note_001",
            "title": "Photosynthesis Study Notes",
            "content": "Comprehensive notes covering all concepts...",
            "sections_covered": 2,
            "primitives_covered": 2,
            "criteria_covered": 2
        }


async def demo_sequential_workflow():
    """Demonstrate the sequential generation workflow."""
    print("🎯 DEMO: Sequential Generation Workflow with User Editing")
    print("=" * 60)
    
    # Initialize mock orchestrator
    orchestrator = MockGenerationOrchestrator()
    
    # Step 1: Start generation session
    print("\n📝 Step 1: Starting Generation Session")
    print("-" * 40)
    
    source_content = """
    Photosynthesis is the process by which plants convert light energy into chemical energy.
    This process occurs in the chloroplasts and involves two main stages: light reactions
    and the Calvin cycle. Chlorophyll, the green pigment in plants, plays a crucial role
    in capturing light energy.
    """
    
    session = await orchestrator.start_generation_session(
        session_id="demo_session_001",
        source_content=source_content,
        source_type="textbook_chapter",
        user_preferences={"learning_style": "visual", "difficulty": "intermediate"}
    )
    
    print(f"✅ Session started: {session['session_id']}")
    print(f"📊 Current step: {session['current_step']}")
    print(f"📋 Status: {session['status']}")
    
    # Step 2: User reviews and proceeds through each step
    print("\n🔄 Step 2: Sequential Generation with User Review")
    print("-" * 40)
    
    while session["current_step"] != "complete":
        current_step = session["current_step"]
        print(f"\n🎯 Current Step: {current_step}")
        
        # Show generated content
        if current_step == "blueprint_creation":
            print("📋 Generated Blueprint:")
            print(json.dumps(session["current_content"]["blueprint"], indent=2))
        
        elif current_step == "section_generation":
            print("📚 Generated Sections:")
            print(json.dumps(session["current_content"]["sections"], indent=2))
            
            # Simulate user editing sections
            print("\n✏️  User editing sections...")
            edit_request = {
                "step": "section_generation",
                "content_id": "sec_001",
                "edited_content": {"title": "Introduction to Photosynthesis (Updated)"},
                "user_notes": "Made title more descriptive"
            }
            session = await orchestrator.user_edit_content("demo_session_001", edit_request)
        
        elif current_step == "primitive_extraction":
            print("🧠 Generated Primitives:")
            print(json.dumps(session["current_content"]["primitives"], indent=2))
        
        elif current_step == "mastery_criteria":
            print("🎯 Generated Mastery Criteria:")
            print(json.dumps(session["current_content"]["mastery_criteria"], indent=2))
        
        elif current_step == "question_generation":
            print("❓ Generated Questions:")
            print(json.dumps(session["current_content"]["questions"], indent=2))
        
        # Proceed to next step
        print(f"\n⏭️  Proceeding to next step...")
        session = await orchestrator.proceed_to_next_step("demo_session_001")
    
    # Step 3: Show complete learning path
    print("\n🎉 Step 3: Complete Learning Path Generated!")
    print("-" * 40)
    
    print("📊 Final Session Status:")
    print(f"   Session ID: {session['session_id']}")
    print(f"   Status: {session['status']}")
    print(f"   Completed Steps: {len(session['completed_steps'])}")
    print(f"   User Edits: {len(session['user_edits'])}")
    
    print("\n📋 Complete Content Summary:")
    content = session["current_content"]
    print(f"   📋 Blueprint: {content['blueprint']['title']}")
    print(f"   📚 Sections: {len(content['sections'])}")
    print(f"   🧠 Primitives: {len(content['primitives'])}")
    print(f"   🎯 Mastery Criteria: {len(content['mastery_criteria'])}")
    print(f"   ❓ Questions: {len(content['questions'])}")
    print(f"   📝 Notes: {content['notes']['title']}")
    
    print("\n✏️  User Edits Applied:")
    for edit in session["user_edits"]:
        print(f"   - {edit['step']}: {edit['user_notes']}")
    
    print("\n🎯 Workflow Benefits Demonstrated:")
    print("   ✅ Sequential generation ensures content coherence")
    print("   ✅ Each step builds on previous step's output")
    print("   ✅ User can edit content between steps")
    print("   ✅ Final result is a cohesive learning path")
    print("   ✅ All components are properly linked")


if __name__ == "__main__":
    asyncio.run(demo_sequential_workflow())



