#!/usr/bin/env python3
"""
Quick diagnostic test for GeminiService initialization issue.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

print("🔍 DIAGNOSING GEMINI SERVICE INITIALIZATION")
print("=" * 50)

# Check environment variable
api_key = os.getenv("GOOGLE_API_KEY")
print(f"✅ GOOGLE_API_KEY found: {api_key[:20]}..." if api_key else "❌ GOOGLE_API_KEY not found")

# Test google.generativeai import
try:
    import google.generativeai as genai
    print("✅ google.generativeai imported successfully")
    print(f"✅ genai version: {genai.__version__}")
except ImportError as e:
    print(f"❌ google.generativeai import failed: {e}")
    genai = None

# Test GeminiService initialization
try:
    sys.path.append('/home/antonio/programming/elevate/elevate-ai-api')
    from app.services.gemini_service import GeminiService, GeminiConfig
    
    print("\n🧪 TESTING GEMINI SERVICE INITIALIZATION")
    print("-" * 40)
    
    # Test with explicit config
    config = GeminiConfig(api_key=api_key)
    service = GeminiService(config)
    
    print(f"✅ GeminiService created")
    print(f"✅ Mock mode: {service.mock_mode}")
    print(f"✅ Model available: {service.model is not None}")
    print(f"✅ Is available: {service.is_available()}")
    
    # Test response generation
    print("\n🎯 TESTING RESPONSE GENERATION")
    print("-" * 40)
    
    import asyncio
    
    async def test_response():
        test_prompt = "Generate a simple response about Python programming using the following context: Python is a programming language."
        response = await service.generate_response(test_prompt)
        print(f"✅ Response generated: {response[:100]}...")
        return response
    
    response = asyncio.run(test_response())
    
    if service.mock_mode:
        print("❌ STILL IN MOCK MODE - Response may be hardcoded!")
    else:
        print("✅ REAL API MODE - Response should use context!")
        
except Exception as e:
    print(f"❌ GeminiService test failed: {e}")
    import traceback
    traceback.print_exc()
