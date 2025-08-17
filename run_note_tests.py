#!/usr/bin/env python3
"""
Test Runner for Note Services with REAL API calls.
Runs tests for note editing, granular editing, and orchestration.
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
    print("🚀 Running Note Services Tests")
    print("=" * 50)
    print("Testing note editing, granular editing, and orchestration...")
    print()
    
    # Run the note services tests
    from tests.services.test_note_services import main
    asyncio.run(main())
