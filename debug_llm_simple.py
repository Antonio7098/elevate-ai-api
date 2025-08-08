#!/usr/bin/env python3
"""
Simple debug script to test the LLM service.
"""

import sys
import os
import asyncio

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.llm_service import llm_service

async def main():
    """Main function to test the LLM service."""
    print("🧪 DEBUGGING LLM SERVICE")
    print("=" * 50)
    
    # Simple test prompt
    prompt = "What is 2+2? Answer with just the number."
    
    print("📝 SIMPLE PROMPT SENT TO LLM:")
    print(prompt)
    print("\n" + "=" * 50)
    
    try:
        # Call LLM directly
        response = await llm_service.call_llm(prompt, prefer_google=True, operation="simple_test")
        
        print("📝 RAW LLM RESPONSE:")
        print(repr(response))
        print("\n" + "=" * 50)
        
        print(f"✅ RESPONSE: {response}")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    return 0

if __name__ == "__main__":
    exit(asyncio.run(main()))
