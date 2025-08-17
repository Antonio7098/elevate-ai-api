#!/usr/bin/env python3
"""
Test Runner for Premium Agents with REAL API calls.
Runs tests for content curation, explanation, and context agents.
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
    print("🚀 Running Premium Agents Tests")
    print("=" * 50)
    print("Testing content curation, explanation, and context agents...")
    print()
    
    # Run the premium agents tests
    from tests.services.test_premium_agents import main
    asyncio.run(main())
