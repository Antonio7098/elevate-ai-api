#!/usr/bin/env python3
"""
Test Google API key validity with minimal request
"""

import os
from dotenv import load_dotenv

# Load environment
load_dotenv(override=True)

print("🧪 TESTING GOOGLE API KEY VALIDITY")
print("=" * 50)

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("❌ No API key found in environment")
    exit(1)

print(f"✅ API key found: {api_key[:10]}...")

# Test 1: Import google.generativeai
try:
    import google.generativeai as genai
    print("✅ google.generativeai imported successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    exit(1)

# Test 2: Configure API
try:
    genai.configure(api_key=api_key)
    print("✅ API configured")
except Exception as e:
    print(f"❌ API configuration failed: {e}")
    exit(1)

# Test 3: List models (simple API call to verify key)
try:
    print("\n🔍 Testing API key with list_models()...")
    models = list(genai.list_models())
    print(f"✅ API key valid! Found {len(models)} available models")
    
    # Show available models
    print("\n📋 Available models:")
    for model in models[:5]:  # Show first 5
        print(f"  - {model.name}")
    if len(models) > 5:
        print(f"  ... and {len(models) - 5} more")
        
except Exception as e:
    print(f"❌ API key test FAILED: {e}")
    print("\n🔧 TROUBLESHOOTING STEPS:")
    print("1. Check Google Cloud Console - APIs & Services - Credentials")
    print("2. Verify 'Generative Language API' is enabled")
    print("3. Check billing account is active")
    print("4. Verify API key has correct permissions")
    print("5. Try generating a new API key")

# Test 4: Try simple generation (if models work)
try:
    print("\n🎯 Testing text generation...")
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content("Hello, world!")
    print("✅ Text generation successful!")
    print(f"📝 Response: {response.text[:100]}...")
except Exception as e:
    print(f"⚠️ Text generation failed: {e}")
