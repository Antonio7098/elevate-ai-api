#!/usr/bin/env python3
"""
Detailed debug script to test the proposition extraction function.
"""

import sys
import os
import asyncio
import google.generativeai as genai

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.config import settings
from app.core.llm_service import create_proposition_extraction_prompt

async def main():
    """Main function to test the proposition extraction."""
    print("🧪 DEBUGGING PROPOSITION EXTRACTION (DETAILED)")
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
        # Configure Google AI directly
        if settings.google_api_key and settings.google_api_key != "your_google_api_key_here":
            genai.configure(api_key=settings.google_api_key)
            
            # Create model directly
            model = genai.GenerativeModel(
                'gemini-1.5-flash',
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": 2000,
                },
                safety_settings=[
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"}
                ]
            )
            
            print("📝 CALLING GOOGLE AI DIRECTLY...")
            response = model.generate_content(prompt)
            
            print("📝 RAW RESPONSE OBJECT:")
            print(f"Response type: {type(response)}")
            print(f"Response parts: {len(response.parts) if hasattr(response, 'parts') else 'No parts'}")
            
            if hasattr(response, 'parts') and response.parts:
                for i, part in enumerate(response.parts):
                    print(f"Part {i}: {part}")
            
            print("\n📝 RAW RESPONSE TEXT:")
            if hasattr(response, 'text'):
                print(repr(response.text))
                print("\n📝 FORMATTED RESPONSE:")
                print(response.text)
            else:
                print("No text attribute found")
                
        else:
            print("❌ Google API key not configured")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    return 0

if __name__ == "__main__":
    exit(asyncio.run(main()))
