#!/usr/bin/env python3
"""
Debug script to test the parsing function.
"""

import sys
import os

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.deconstruction import parse_criteria_response
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

def main():
    """Main function to test the parsing function."""
    print("🧪 DEBUGGING PARSING FUNCTION")
    print("=" * 50)
    
    # Create test data
    primitive = create_test_primitive()
    
    # Simulate the LLM response
    response = '''[
  {
    "title": "Identify Key Components",
    "description": "Accurately identify the primary inputs (light energy, carbon dioxide, water) and outputs (carbohydrates/sugars, chemical energy) of the photosynthesis process, as well as the overall purpose of the process.",
    "ueeLevel": "UNDERSTAND",
    "weight": 3.0,
    "isRequired": true
  },
  {
    "title": "Describe the Process Flow",
    "description": "Explain the sequence of events in photosynthesis, detailing how light energy, carbon dioxide, and water are transformed into chemical energy stored in carbohydrates. This includes describing the conversion of light energy to chemical energy and the synthesis of sugars.",
    "ueeLevel": "USE",
    "weight": 4.0,
    "isRequired": true
  },
  {
    "title": "Relate to Organismal Function",
    "description": "Explain the fundamental importance of photosynthesis for the survival and energy needs of plants and, by extension, its role as the basis of energy flow in most ecosystems (e.g., as the foundation of food chains).",
    "ueeLevel": "EXPLORE",
    "weight": 3.5,
    "isRequired": true
  }
]'''
    
    print("📝 RESPONSE TO PARSE:")
    print(response)
    print("\n" + "=" * 50)
    
    try:
        # Parse the response
        criteria = parse_criteria_response(response, primitive)
        
        print(f"✅ PARSED {len(criteria)} CRITERIA:")
        for i, criterion in enumerate(criteria, 1):
            print(f"  {i}. {criterion.title} ({criterion.ueeLevel}) - Weight: {criterion.weight}")
        
    except Exception as e:
        print(f"❌ ERROR parsing response: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
