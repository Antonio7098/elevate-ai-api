#!/usr/bin/env python3
"""
Debug script to test LLM service and identify issues.
"""

import asyncio
import json
from app.core.llm_service import llm_service, create_entity_extraction_prompt
from app.core.deconstruction import extract_key_terms


async def test_entity_extraction():
    """Test entity extraction with problematic text."""
    test_text = """
    Photosynthesis is the process by which plants convert light energy into chemical energy. 
    Chloroplasts are organelles found in plant cells that conduct photosynthesis.
    The Calvin cycle is a series of biochemical reactions that take place in the stroma of chloroplasts.
    """
    
    print("🧪 Testing Entity Extraction")
    print("=" * 50)
    print(f"Test text: {test_text.strip()}")
    print()
    
    try:
        # Test direct LLM call
        print("1. Testing direct LLM call...")
        prompt = create_entity_extraction_prompt(test_text, "test_section")
        response = await llm_service.call_llm(prompt, prefer_google=True, operation="test_extraction")
        print(f"✅ LLM Response: {response[:200]}...")
        
        # Test JSON parsing
        print("\n2. Testing JSON parsing...")
        entities_data = json.loads(response.strip())
        print(f"✅ Parsed {len(entities_data)} entities")
        
        # Test entity creation
        print("\n3. Testing entity creation...")
        entities = await extract_key_terms(test_text, "test_section")
        print(f"✅ Created {len(entities)} valid entities")
        
        for i, entity in enumerate(entities):
            print(f"   Entity {i+1}: {entity.entity} - {entity.definition[:50]}...")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def test_error_handling():
    """Test error handling with edge cases."""
    print("\n🧪 Testing Error Handling")
    print("=" * 50)
    
    # Test with empty text
    print("1. Testing with empty text...")
    try:
        entities = await extract_key_terms("", "empty_section")
        print(f"✅ Empty text handled: {len(entities)} entities")
    except Exception as e:
        print(f"❌ Empty text error: {e}")
    
    # Test with text that has no entities
    print("\n2. Testing with text that has no entities...")
    try:
        entities = await extract_key_terms("This is just some random text without any defined terms.", "no_entities")
        print(f"✅ No entities text handled: {len(entities)} entities")
    except Exception as e:
        print(f"❌ No entities error: {e}")


async def main():
    """Main test function."""
    print("🚀 LLM Debug Test")
    print("=" * 50)
    
    await test_entity_extraction()
    await test_error_handling()
    
    print("\n✅ Debug test completed!")


if __name__ == "__main__":
    asyncio.run(main()) 