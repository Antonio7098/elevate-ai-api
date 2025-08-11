#!/usr/bin/env python3
"""
Debug script to see what the LLM is actually returning.
"""

import asyncio
import sys
import os

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.llm_service import llm_service
from app.core.deconstruction import create_enhanced_criteria_prompt
from app.models.learning_blueprint import KnowledgePrimitive, MasteryCriterion

def create_test_primitive():
    """Create a test primitive for debugging."""
    primitive = KnowledgePrimitive(
        primitiveId="test_primitive_1",
        title="Photosynthesis Process",
        description="The process by which plants convert light energy into chemical energy.",
        primitiveType="process",
        difficultyLevel="intermediate",
        sourceId="test_source_1",
        masteryCriteria=[
            MasteryCriterion(
                criterionId="test_criterion_1",
                title="Basic Understanding",
                description="Understand the basic concept of photosynthesis",
                ueeLevel="UNDERSTAND",
                weight=2.0,
                isRequired=True
            )
        ]
    )
    return primitive

def create_test_preferences():
    """Create test user preferences."""
    return {
        "learning_style": "balanced",
        "focus_areas": ["biology", "science"],
        "difficulty_preference": "intermediate"
    }

async def main():
    """Main function to test the LLM response."""
    print("🧪 DEBUGGING LLM RESPONSE")
    print("=" * 50)
    
    # Create test data
    primitive = create_test_primitive()
    preferences = create_test_preferences()
    source_content = """
    Photosynthesis is a process used by plants and other organisms to convert light energy 
    into chemical energy that can later be released to fuel the organisms' activities. 
    This chemical energy is stored in carbohydrate molecules, such as sugars, which are 
    synthesized from carbon dioxide and water.
    """
    
    # Create the prompt
    prompt = create_enhanced_criteria_prompt(
        primitive=primitive,
        source_content=source_content,
        user_preferences=preferences
    )
    
    print("📝 PROMPT SENT TO LLM:")
    print(prompt)
    print("\n" + "=" * 50)
    
    try:
        # Call the LLM
        response = await llm_service.call_google_ai(
            prompt=prompt,
            operation="debug_criteria_generation"
        )
        
        print("📝 RAW LLM RESPONSE:")
        print(repr(response))
        print("\n" + "=" * 50)
        
        print("📝 LLM RESPONSE (formatted):")
        print(response)
        
    except Exception as e:
        print(f"❌ ERROR calling LLM: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(asyncio.run(main()))
