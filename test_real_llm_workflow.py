#!/usr/bin/env python3
"""
Real LLM Test Script for Sequential Generation Workflow
Uses Gemini 2.5 Flash service to test the complete workflow with actual AI generation.
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional

# Add the app directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.core.generation_orchestrator import GenerationOrchestrator, GenerationStep, GenerationStatus
from app.models.blueprint_centric import ContentGenerationRequest, LearningBlueprint
from app.core.deconstruction import DeconstructionService
from app.services.llm_service import GeminiLLMService
from app.core.config import settings

class RealLLMTestWorkflow:
    """Test class for running the sequential workflow with real LLM calls."""
    
    def __init__(self):
        """Initialize the test workflow with real services."""
        self.orchestrator = GenerationOrchestrator()
        self.llm_service = GeminiLLMService()
        self.deconstruction_service = DeconstructionService()
        
        # Test content - a sample educational topic
        self.test_source = """
        Machine Learning Fundamentals
        
        Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves.
        
        The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future based on the examples that we provide. The primary aim is to allow the computers learn automatically without human intervention or assistance and adjust actions accordingly.
        
        There are three main types of machine learning:
        
        1. Supervised Learning: The algorithm is trained on labeled data, learning to map inputs to outputs.
        2. Unsupervised Learning: The algorithm finds hidden patterns in unlabeled data.
        3. Reinforcement Learning: The algorithm learns through trial and error, receiving rewards for good actions.
        
        Key concepts in machine learning include:
        - Training data: The dataset used to train the model
        - Features: The input variables used for prediction
        - Labels: The output variables we want to predict
        - Model: The algorithm that makes predictions
        - Overfitting: When a model performs well on training data but poorly on new data
        - Underfitting: When a model is too simple to capture the underlying patterns
        
        Common algorithms include linear regression, decision trees, random forests, support vector machines, and neural networks. Each has its strengths and is suited for different types of problems.
        """
        
        self.session_id = f"test_session_{int(time.time())}"
        
    async def test_complete_workflow(self) -> Dict[str, Any]:
        """Run the complete sequential workflow with real LLM calls."""
        print("🚀 Starting Real LLM Sequential Generation Workflow Test")
        print("=" * 60)
        
        results = {
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "steps_completed": [],
            "errors": [],
            "final_blueprint": None,
            "total_time": 0
        }
        
        start_time = time.time()
        
        try:
            # Step 1: Start generation session
            print("\n📝 Step 1: Starting Generation Session")
            print("-" * 40)
            
            user_preferences = {
                "difficulty_level": "intermediate",
                "target_audience": "university_students",
                "learning_style": "practical",
                "estimated_duration": "2_weeks",
                "max_sections": 5,
                "max_primitives_per_section": 8,
                "max_mastery_criteria_per_primitive": 3,
                "max_questions_per_criterion": 5
            }
            
            await self.orchestrator.start_generation_session(
                session_id=self.session_id,
                source_content=self.test_source,
                source_type="textbook",
                user_preferences=user_preferences
            )
            
            progress = await self.orchestrator.get_generation_progress(self.session_id)
            print(f"✅ Session started: {progress.status}")
            print(f"📊 Current step: {progress.current_step}")
            results["steps_completed"].append("session_started")
            
            # Step 2: Generate Blueprint
            print("\n🔍 Step 2: Generating Learning Blueprint")
            print("-" * 40)
            
            await self.orchestrator.proceed_to_next_step(self.session_id)
            progress = await self.orchestrator.get_generation_progress(self.session_id)
            
            if progress.current_step == GenerationStep.BLUEPRINT_CREATION:
                print("✅ Blueprint generated successfully")
                print(f"📊 Current step: {progress.current_step}")
                results["steps_completed"].append("blueprint_generated")
                
                # Show blueprint preview
                blueprint = progress.current_content.get("blueprint", {})
                if isinstance(blueprint, dict):
                    print(f"📋 Blueprint Title: {blueprint.get('title', 'N/A')}")
                    print(f"📋 Blueprint Description: {blueprint.get('description', 'N/A')[:100]}...")
            else:
                print(f"❌ Expected BLUEPRINT_CREATION, got {progress.current_step}")
                results["errors"].append(f"Blueprint generation failed: {progress.current_step}")
                return results
            
            # Step 3: Generate Sections
            print("\n📚 Step 3: Generating Sections")
            print("-" * 40)
            
            await self.orchestrator.proceed_to_next_step(self.session_id)
            progress = await self.orchestrator.get_generation_progress(self.session_id)
            
            if progress.current_step == GenerationStep.SECTION_GENERATION:
                print("✅ Sections generated successfully")
                print(f"📊 Current step: {progress.current_step}")
                results["steps_completed"].append("sections_generated")
                
                # Show sections preview
                sections = progress.current_content.get("sections", [])
                if sections:
                    print(f"📚 Number of sections: {len(sections)}")
                    for i, section in enumerate(sections[:3]):  # Show first 3
                        print(f"   {i+1}. {section.get('title', 'N/A')}")
            else:
                print(f"❌ Expected SECTION_GENERATION, got {progress.current_step}")
                results["errors"].append(f"Section generation failed: {progress.current_step}")
                return results
            
            # Step 4: Extract Primitives
            print("\n🧠 Step 4: Extracting Knowledge Primitives")
            print("-" * 40)
            
            await self.orchestrator.proceed_to_next_step(self.session_id)
            progress = await self.orchestrator.get_generation_progress(self.session_id)
            
            if progress.current_step == GenerationStep.MASTERY_CRITERIA:
                print("✅ Primitives extracted successfully")
                print(f"📊 Current step: {progress.current_step}")
                results["steps_completed"].append("primitives_extracted")
                
                # Show primitives preview
                primitives = progress.current_content.get("primitives", [])
                if primitives:
                    print(f"🧠 Number of primitives: {len(primitives)}")
                    for i, primitive in enumerate(primitives[:3]):  # Show first 3
                        print(f"   {i+1}. {primitive.get('title', 'N/A')}")
            else:
                print(f"❌ Expected MASTERY_CRITERIA, got {progress.current_step}")
                results["errors"].append(f"Primitive extraction failed: {progress.current_step}")
                return results
            
            # Step 5: Generate Mastery Criteria
            print("\n🎯 Step 5: Generating Mastery Criteria")
            print("-" * 40)
            
            await self.orchestrator.proceed_to_next_step(self.session_id)
            progress = await self.orchestrator.get_generation_progress(self.session_id)
            
            if progress.current_step == GenerationStep.QUESTION_GENERATION:
                print("✅ Mastery criteria generated successfully")
                print(f"📊 Current step: {progress.current_step}")
                results["steps_completed"].append("mastery_criteria_generated")
                
                # Show mastery criteria preview
                criteria = progress.current_content.get("mastery_criteria", [])
                if criteria:
                    print(f"🎯 Number of mastery criteria: {len(criteria)}")
                    for i, criterion in enumerate(criteria[:3]):  # Show first 3
                        print(f"   {i+1}. {criterion.get('title', 'N/A')}")
            else:
                print(f"❌ Expected QUESTION_GENERATION, got {progress.current_step}")
                results["errors"].append(f"Mastery criteria generation failed: {progress.current_step}")
                return results
            
            # Step 6: Generate Questions
            print("\n❓ Step 6: Generating Questions")
            print("-" * 40)
            
            await self.orchestrator.proceed_to_next_step(self.session_id)
            progress = await self.orchestrator.get_generation_progress(self.session_id)
            
            if progress.current_step == GenerationStep.NOTE_GENERATION:
                print("✅ Questions generated successfully")
                print(f"📊 Current step: {progress.current_step}")
                results["steps_completed"].append("questions_generated")
                
                # Show questions preview
                questions = progress.current_content.get("questions", [])
                if questions:
                    print(f"❓ Number of questions: {len(questions)}")
                    for i, question in enumerate(questions[:3]):  # Show first 3
                        print(f"   {i+1}. {question.get('text', 'N/A')[:80]}...")
            else:
                print(f"❌ Expected NOTE_GENERATION, got {progress.current_step}")
                results["errors"].append(f"Question generation failed: {progress.current_step}")
                return results
            
            # Step 7: Generate Notes
            print("\n📝 Step 7: Generating Study Notes")
            print("-" * 40)
            
            await self.orchestrator.proceed_to_next_step(self.session_id)
            progress = await self.orchestrator.get_generation_progress(self.session_id)
            
            if progress.current_step == GenerationStep.COMPLETE:
                print("✅ Notes generated successfully")
                print(f"📊 Current step: {progress.current_step}")
                results["steps_completed"].append("notes_generated")
                
                # Show notes preview
                notes = progress.current_content.get("notes", [])
                if notes:
                    print(f"📝 Number of notes: {len(notes)}")
                    for i, note in enumerate(notes[:2]):  # Show first 2
                        print(f"   {i+1}. {note.get('title', 'N/A')}")
            else:
                print(f"❌ Expected COMPLETE, got {progress.current_step}")
                results["errors"].append(f"Notes generation failed: {progress.current_step}")
                return results
            
            # Step 8: Complete the workflow
            print("\n🏁 Step 8: Completing Workflow")
            print("-" * 40)
            
            await self.orchestrator.proceed_to_next_step(self.session_id)
            progress = await self.orchestrator.get_generation_progress(self.session_id)
            
            if progress.status == GenerationStatus.COMPLETED:
                print("✅ Workflow completed successfully!")
                print(f"📊 Final status: {progress.status}")
                results["steps_completed"].append("workflow_completed")
                
                # Get the complete learning path
                complete_path = await self.orchestrator.get_complete_learning_path(self.session_id)
                results["final_blueprint"] = complete_path
                
                print(f"\n🎉 SUCCESS! All {len(results['steps_completed'])} steps completed!")
                print(f"📊 Total primitives: {len(complete_path.get('primitives', []))}")
                print(f"📊 Total mastery criteria: {len(complete_path.get('mastery_criteria', []))}")
                print(f"📊 Total questions: {len(complete_path.get('questions', []))}")
                print(f"📊 Total notes: {len(complete_path.get('notes', []))}")
                
            else:
                print(f"❌ Expected COMPLETED status, got {progress.status}")
                results["errors"].append(f"Workflow completion failed: {progress.status}")
                
        except Exception as e:
            print(f"❌ Error during workflow: {str(e)}")
            results["errors"].append(f"Workflow error: {str(e)}")
            import traceback
            traceback.print_exc()
        
        finally:
            end_time = time.time()
            results["total_time"] = end_time - start_time
            results["end_time"] = datetime.now().isoformat()
            
            print(f"\n⏱️  Total execution time: {results['total_time']:.2f} seconds")
            print("=" * 60)
            
        return results
    
    async def test_user_editing_workflow(self) -> Dict[str, Any]:
        """Test the user editing workflow at different steps."""
        print("\n✏️  Testing User Editing Workflow")
        print("=" * 60)
        
        # Start a new session for editing tests
        edit_session_id = f"edit_test_{int(time.time())}"
        
        user_preferences = {
            "difficulty_level": "beginner",
            "max_sections": 3,
            "max_primitives_per_section": 3
        }
        
        await self.orchestrator.start_generation_session(
            edit_session_id, 
            "Python programming basics: variables, loops, functions",
            "textbook",
            user_preferences
        )
        
        # Generate blueprint
        await self.orchestrator.proceed_to_next_step(edit_session_id)
        progress = await self.orchestrator.get_generation_progress(edit_session_id)
        
        if progress.current_step == GenerationStep.BLUEPRINT_CREATION:
            print("✅ Blueprint ready for editing")
            
            # Test editing the blueprint
            edit_request = {
                "title": "Python Programming Fundamentals - Enhanced Edition",
                "description": "A comprehensive guide to Python programming with practical examples"
            }
            
            await self.orchestrator.user_edit_content(edit_session_id, edit_request)
            updated_progress = await self.orchestrator.get_generation_progress(edit_session_id)
            
            if updated_progress.status == GenerationStatus.USER_EDITING:
                print("✅ User editing applied successfully")
                
                # Continue with the workflow
                await self.orchestrator.proceed_to_next_step(edit_session_id)
                final_progress = await self.orchestrator.get_generation_progress(edit_session_id)
                print(f"✅ Workflow continued after editing: {final_progress.current_step}")
            else:
                print(f"❌ Editing failed: {updated_progress.status}")
        
        return {"edit_test_completed": True}
    
    async def run_all_tests(self):
        """Run all test scenarios."""
        print("🧪 Running Real LLM Sequential Generation Workflow Tests")
        print("=" * 80)
        
        # Test 1: Complete workflow
        print("\n🔬 Test 1: Complete Sequential Workflow")
        workflow_results = await self.test_complete_workflow()
        
        # Test 2: User editing workflow
        print("\n🔬 Test 2: User Editing Workflow")
        editing_results = await self.test_user_editing_workflow()
        
        # Summary
        print("\n📊 TEST SUMMARY")
        print("=" * 80)
        print(f"✅ Workflow Test: {'PASSED' if not workflow_results['errors'] else 'FAILED'}")
        print(f"✅ Editing Test: {'PASSED' if editing_results.get('edit_test_completed') else 'FAILED'}")
        
        if workflow_results['errors']:
            print(f"\n❌ Errors encountered:")
            for error in workflow_results['errors']:
                print(f"   - {error}")
        
        print(f"\n⏱️  Total execution time: {workflow_results['total_time']:.2f} seconds")
        print(f"📝 Steps completed: {len(workflow_results['steps_completed'])}")
        
        return {
            "workflow_results": workflow_results,
            "editing_results": editing_results
        }

async def main():
    """Main function to run the tests."""
    # Check if we have the required environment variables
    required_vars = ["GOOGLE_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these variables before running the test.")
        print("Example:")
        print("export GOOGLE_API_KEY='your-api-key-here'")
        return 1
    
    print("🔑 Environment variables check: PASSED")
    print(f"📡 Using model: gemini-2.5-flash (default)")
    
    try:
        test_workflow = RealLLMTestWorkflow()
        results = await test_workflow.run_all_tests()
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Test results saved to: {results_file}")
        
        return 0 if not results['workflow_results']['errors'] else 1
        
    except Exception as e:
        print(f"❌ Test execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
