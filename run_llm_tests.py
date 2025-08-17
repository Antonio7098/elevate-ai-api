#!/usr/bin/env python3
"""
Test Runner for LLM Services with REAL API calls.
Runs tests for Google Gemini, OpenRouter, and Mock LLM services.
"""

import asyncio
import sys
import os

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded")
except ImportError:
    print("⚠️  python-dotenv not available")

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("🚀 Running LLM Services Tests")
    print("=" * 50)
    print("Testing Google Gemini, OpenRouter, and Mock LLM services...")
    print()
    
    # Run the LLM tests
    from tests.services.test_llm_services import main
    asyncio.run(main())
