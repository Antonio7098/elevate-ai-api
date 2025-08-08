#!/usr/bin/env python3
"""
Debug script to test the proposition extraction function.
"""

import sys
import os
import asyncio

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.llm_service import llm_service, create_proposition_extraction_prompt

async def main():
    """Main function to test the proposition extraction."""
    print("🧪 DEBUGGING PROPOSITION EXTRACTION")
    print("=" * 50)
    
    # Test data
    test_text = """
    Photosynthesis is a process used by plants and other organisms to convert light energy 
    into chemical energy that can later be released to fuel the organisms' activities. 
    This chemical energy is stored in carbohydrate molecules, such as sugars, which are 
    synthesized from carbon dioxide and water.
    """
    
    section_id = "test_section_1"
    
    # Create prompt
    prompt = create_proposition_extraction_prompt(test_text, section_id)
    
    print("📝 PROMPT SENT TO LLM:")
    print(prompt)
    print("\n" + "=" * 50)
    
    try:
        # Call LLM directly
        response = await llm_service.call_llm(prompt, prefer_google=True, operation="extract_propositions")
        
        print("📝 RAW LLM RESPONSE:")
        print(repr(response))
        print("\n" + "=" * 50)
        
        # Try to parse the response
        import json
        propositions_data = json.loads(response.strip())
        
        print(f"✅ PARSED {len(propositions_data)} PROPOSITIONS:")
        for i, prop_data in enumerate(propositions_data, 1):
            print(f"  {i}. {prop_data.get('statement', 'No statement')}")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    return 0

if __name__ == "__main__":
    exit(asyncio.run(main()))
