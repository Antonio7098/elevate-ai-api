#!/usr/bin/env python3
"""
Comprehensive Output Generator for LangGraph Sequential Generation Workflow.
This script runs the workflow and outputs all source content and generated materials to a text file.
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from app.core.premium.workflows.sequential_generation_workflow import SequentialGenerationWorkflow

async def generate_comprehensive_output():
    """Generate comprehensive output from the LangGraph workflow"""
    
    print("🚀 Generating Comprehensive Output from LangGraph Workflow")
    print("=" * 70)
    
    # Initialize the workflow
    workflow = SequentialGenerationWorkflow()
    
    # Sample source content (more comprehensive than the test)
    source_content = """
    Photosynthesis is the fundamental biological process by which plants, algae, and some bacteria convert light energy into chemical energy. This process is essential for life on Earth as it provides the primary source of energy for most ecosystems and produces oxygen as a byproduct.

    The process occurs in specialized organelles called chloroplasts, which contain the green pigment chlorophyll. Chlorophyll is responsible for capturing light energy from the sun and converting it into chemical energy through a series of complex biochemical reactions.

    The overall chemical equation for photosynthesis is:
    6CO2 + 6H2O + light energy → C6H12O6 + 6O2

    This equation represents the conversion of carbon dioxide and water into glucose (a simple sugar) and oxygen, using light energy as the driving force.

    The process can be broken down into two main stages:

    1. Light-Dependent Reactions (Light Reactions):
       - Occur in the thylakoid membranes of chloroplasts
       - Light energy is absorbed by chlorophyll and converted to chemical energy
       - Water molecules are split, releasing oxygen as a byproduct
       - ATP and NADPH are produced as energy carriers

    2. Light-Independent Reactions (Calvin Cycle):
       - Occur in the stroma of chloroplasts
       - Carbon dioxide is fixed and converted into organic compounds
       - ATP and NADPH from light reactions provide energy and reducing power
       - Glucose and other carbohydrates are synthesized

    Key factors that affect photosynthesis include:
    - Light intensity and wavelength
    - Carbon dioxide concentration
    - Temperature
    - Water availability
    - Nutrient availability

    The importance of photosynthesis extends beyond just plant growth:
    - It's the primary source of oxygen in Earth's atmosphere
    - It forms the base of most food chains
    - It helps regulate Earth's climate by removing CO2 from the atmosphere
    - It provides the energy source for cellular respiration in all living organisms

    Understanding photosynthesis is crucial for:
    - Agriculture and crop improvement
    - Climate change mitigation
    - Biofuel production
    - Understanding ecosystem dynamics
    - Developing sustainable energy solutions
    """
    
    source_type = "educational_text"
    user_preferences = {
        "difficulty_level": "intermediate",
        "target_audience": "high_school_students",
        "include_examples": True,
        "include_diagrams": False,
        "focus_areas": ["process_steps", "importance", "applications"]
    }
    
    try:
        # Start the workflow
        print("\n📋 Starting comprehensive workflow...")
        workflow_id = await workflow.start_workflow(source_content, source_type, user_preferences)
        print(f"✅ Workflow started with ID: {workflow_id}")
        
        # Wait a moment for completion
        await asyncio.sleep(2)
        
        # Get workflow status
        print("\n📊 Retrieving workflow results...")
        status = await workflow.get_workflow_status(workflow_id)
        
        if status:
            print(f"✅ Workflow completed successfully!")
            print(f"   - Status: {status.get('status', 'N/A')}")
            print(f"   - Generated: {len(status.get('sections', []))} sections, {len(status.get('primitives', []))} primitives, {len(status.get('questions', []))} questions, {len(status.get('notes', []))} notes")
            
            # Generate comprehensive output file
            output_filename = f"langgraph_comprehensive_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            with open(output_filename, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("LANGGRAPH SEQUENTIAL GENERATION WORKFLOW - COMPREHENSIVE OUTPUT\n")
                f.write("=" * 80 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Workflow ID: {workflow_id}\n")
                f.write(f"Status: {status.get('status', 'N/A')}\n")
                f.write("=" * 80 + "\n\n")
                
                # Source Content
                f.write("📚 SOURCE CONTENT\n")
                f.write("-" * 40 + "\n")
                f.write(f"Type: {source_type}\n")
                f.write(f"User Preferences: {user_preferences}\n\n")
                f.write(source_content.strip() + "\n\n")
                
                # Generated Blueprint
                f.write("📋 GENERATED LEARNING BLUEPRINT\n")
                f.write("-" * 40 + "\n")
                blueprint = status.get('blueprint', {})
                if hasattr(blueprint, 'source_title'):
                    f.write(f"Title: {blueprint.source_title}\n")
                    f.write(f"Source Type: {blueprint.source_type}\n")
                    if hasattr(blueprint, 'source_summary') and blueprint.source_summary:
                        summary = blueprint.source_summary
                        if hasattr(summary, 'core_thesis_or_main_argument'):
                            f.write(f"Core Thesis: {summary.core_thesis_or_main_argument}\n")
                        if hasattr(summary, 'inferred_purpose'):
                            f.write(f"Purpose: {summary.inferred_purpose}\n")
                f.write("\n")
                
                # Generated Sections
                f.write("📖 GENERATED SECTIONS\n")
                f.write("-" * 40 + "\n")
                sections = status.get('sections', [])
                for i, section in enumerate(sections, 1):
                    f.write(f"{i}. {section.get('title', 'N/A')}\n")
                    f.write(f"   ID: {section.get('section_id', 'N/A')}\n")
                    f.write(f"   Content: {section.get('content', 'N/A')}\n")
                    f.write(f"   Level: {section.get('hierarchy_level', 'N/A')}\n\n")
                
                # Generated Primitives
                f.write("🧠 GENERATED KNOWLEDGE PRIMITIVES\n")
                f.write("-" * 40 + "\n")
                primitives = status.get('primitives', [])
                for i, primitive in enumerate(primitives, 1):
                    f.write(f"{i}. {primitive.get('title', 'N/A')}\n")
                    f.write(f"   ID: {primitive.get('primitive_id', 'N/A')}\n")
                    f.write(f"   Description: {primitive.get('description', 'N/A')}\n")
                    f.write(f"   Section: {primitive.get('section_id', 'N/A')}\n")
                    f.write(f"   Type: {primitive.get('primitive_type', 'N/A')}\n\n")
                
                # Generated Mastery Criteria
                f.write("🎯 GENERATED MASTERY CRITERIA\n")
                f.write("-" * 40 + "\n")
                mastery_criteria = status.get('mastery_criteria', [])
                for i, criterion in enumerate(mastery_criteria, 1):
                    f.write(f"{i}. {criterion.get('title', 'N/A')}\n")
                    f.write(f"   ID: {criterion.get('criterion_id', 'N/A')}\n")
                    f.write(f"   Description: {criterion.get('description', 'N/A')}\n")
                    f.write(f"   UUE Stage: {criterion.get('uue_stage', 'N/A')}\n")
                    f.write(f"   Difficulty: {criterion.get('difficulty_level', 'N/A')}\n")
                    f.write("   Success Criteria:\n")
                    for j, success_criterion in enumerate(criterion.get('success_criteria', []), 1):
                        f.write(f"     {j}. {success_criterion}\n")
                    f.write("\n")
                
                # Generated Questions
                f.write("❓ GENERATED ASSESSMENT QUESTIONS\n")
                f.write("-" * 40 + "\n")
                questions = status.get('questions', [])
                for i, question in enumerate(questions, 1):
                    f.write(f"{i}. {question.get('question_text', 'N/A')}\n")
                    f.write(f"   ID: {question.get('question_id', 'N/A')}\n")
                    f.write(f"   Type: {question.get('question_type', 'N/A')}\n")
                    f.write(f"   Difficulty: {question.get('difficulty', 'N/A')}\n")
                    f.write(f"   Criterion: {question.get('criterion_id', 'N/A')}\n")
                    if question.get('options'):
                        f.write("   Options:\n")
                        for j, option in enumerate(question.get('options', [])):
                            f.write(f"     {chr(65+j)}. {option}\n")
                    f.write(f"   Correct Answer: {question.get('correct_answer', 'N/A')}\n")
                    f.write(f"   Explanation: {question.get('explanation', 'N/A')}\n\n")
                
                # Generated Notes
                f.write("📝 GENERATED STUDY NOTES\n")
                f.write("-" * 40 + "\n")
                notes = status.get('notes', [])
                for i, note in enumerate(notes, 1):
                    f.write(f"{i}. {note.get('title', 'N/A')}\n")
                    f.write(f"   ID: {note.get('note_id', 'N/A')}\n")
                    f.write(f"   Type: {note.get('content_type', 'N/A')}\n")
                    f.write(f"   Content: {note.get('content', 'N/A')}\n")
                    if note.get('description'):
                        f.write(f"   Description: {note.get('description', 'N/A')}\n")
                    if note.get('related_content'):
                        f.write(f"   Related Content: {note.get('related_content', 'N/A')}\n")
                    f.write("\n")
                
                # Workflow Metadata
                f.write("📊 WORKFLOW METADATA\n")
                f.write("-" * 40 + "\n")
                metadata = status.get('metadata', {})
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")
                
                # User Edits (if any)
                if status.get('user_edits'):
                    f.write("\n✏️ USER EDITS APPLIED\n")
                    f.write("-" * 40 + "\n")
                    for i, edit in enumerate(status.get('user_edits', []), 1):
                        f.write(f"{i}. Timestamp: {edit.get('timestamp', 'N/A')}\n")
                        f.write(f"   Edits: {edit.get('edits', 'N/A')}\n\n")
                
                # Errors (if any)
                if status.get('errors'):
                    f.write("❌ ERRORS ENCOUNTERED\n")
                    f.write("-" * 40 + "\n")
                    for i, error in enumerate(status.get('errors', []), 1):
                        f.write(f"{i}. {error}\n")
                
                f.write("\n" + "=" * 80 + "\n")
                f.write("END OF COMPREHENSIVE OUTPUT\n")
                f.write("=" * 80 + "\n")
            
            print(f"✅ Comprehensive output saved to: {output_filename}")
            print(f"📄 File size: {os.path.getsize(output_filename)} bytes")
            
            # Show a preview of the file
            print(f"\n📖 Preview of generated content:")
            print(f"   - Sections: {len(sections)}")
            print(f"   - Primitives: {len(primitives)}")
            print(f"   - Mastery Criteria: {len(mastery_criteria)}")
            print(f"   - Questions: {len(questions)}")
            print(f"   - Notes: {len(notes)}")
            
        else:
            print("❌ Failed to retrieve workflow results")
        
    except Exception as e:
        print(f"❌ Error during comprehensive output generation: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Main function"""
    print("🧪 LangGraph Comprehensive Output Generator")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("app"):
        print("❌ Please run this script from the elevate-ai-api directory")
        sys.exit(1)
    
    # Generate comprehensive output
    await generate_comprehensive_output()
    
    print("\n🎯 Comprehensive output generation completed!")

if __name__ == "__main__":
    # Run the comprehensive output generator
    asyncio.run(main())

