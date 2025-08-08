#!/usr/bin/env python3
"""
Debug script to test simplified prompts that might avoid safety filters.
"""

import sys
import os
import asyncio

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.llm_service import llm_service

async def test_simplified_proposition_prompt():
    """Test a simplified proposition extraction prompt."""
    print("🧪 TESTING SIMPLIFIED PROPOSITION PROMPT")
    print("=" * 50)
    
    # Simplified prompt without potentially problematic words
    prompt = """
Identify key facts from this text:

Text: Photosynthesis is a process used by plants to convert light energy into chemical energy. This energy is stored in sugars made from carbon dioxide and water.

Return a JSON array:
[
  {
    "id": "1",
    "statement": "fact from the text",
    "evidence": [],
    "sections": ["main"]
  }
]

Return only the JSON array.
"""
    
    print("📝 SIMPLIFIED PROMPT:")
    print(prompt)
    print("\n" + "=" * 50)
    
    try:
        response = await llm_service.call_llm(prompt, prefer_google=True, operation="simple_proposition_test")
        print("✅ SUCCESS! Response:")
        print(response)
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

async def test_original_proposition_prompt():
    """Test the original proposition extraction prompt."""
    print("\n🧪 TESTING ORIGINAL PROPOSITION PROMPT")
    print("=" * 50)
    
    # Original prompt
    prompt = """
Extract ONLY explicitly stated propositions and facts from the following text section. Do not infer or add information not directly stated.

Section: test_section_1
Text: Photosynthesis is a process used by plants to convert light energy into chemical energy. This energy is stored in sugars made from carbon dioxide and water.

Return a JSON array of propositions with the following structure:
[
  {
    "id": "unique_id",
    "statement": "Exact statement as written in the text",
    "supporting_evidence": ["explicit evidence mentioned in text"],
    "sections": ["section_id"]
  }
]

IMPORTANT: 
- Only extract statements that are explicitly stated in the text
- Do not infer conclusions or add information not present
- Use the exact wording from the text when possible
- Only include evidence that is explicitly mentioned

Return only the JSON array, no additional text.
"""
    
    print("📝 ORIGINAL PROMPT:")
    print(prompt)
    print("\n" + "=" * 50)
    
    try:
        response = await llm_service.call_llm(prompt, prefer_google=True, operation="original_proposition_test")
        print("✅ SUCCESS! Response:")
        print(response)
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

async def test_word_variations():
    """Test if specific words trigger safety filters."""
    print("\n🧪 TESTING WORD VARIATIONS")
    print("=" * 50)
    
    # Test different variations of potentially problematic words
    test_words = [
        ("proposition", "statement"),
        ("extract", "identify"),
        ("explicitly stated", "mentioned"),
        ("supporting_evidence", "evidence"),
    ]
    
    for original, replacement in test_words:
        print(f"\n📝 Testing replacement: '{original}' -> '{replacement}'")
        
        prompt = f"""
Identify {replacement}s from this text:

Text: Photosynthesis converts light to chemical energy.

Return JSON array:
[{{"id": "1", "content": "fact from text"}}]
"""
        
        try:
            response = await llm_service.call_llm(prompt, prefer_google=True, operation=f"word_test_{replacement}")
            print(f"✅ '{replacement}' works!")
        except Exception as e:
            print(f"❌ '{replacement}' failed: {e}")

async def main():
    """Main function to test different prompt variations."""
    print("🔍 INVESTIGATING PROMPT SAFETY FILTER ISSUES")
    print("=" * 60)
    
    # Test simplified prompt first
    simple_works = await test_simplified_proposition_prompt()
    
    # Test original prompt
    original_works = await test_original_proposition_prompt()
    
    # Test word variations
    await test_word_variations()
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY:")
    print(f"Simplified prompt: {'✅ Works' if simple_works else '❌ Failed'}")
    print(f"Original prompt: {'✅ Works' if original_works else '❌ Failed'}")
    
    return 0

if __name__ == "__main__":
    exit(asyncio.run(main()))
