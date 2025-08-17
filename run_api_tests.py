#!/usr/bin/env python3
"""
Test Runner for API Endpoints with REAL API calls.
Runs tests for note creation, editing, and search endpoints.
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
    print("🚀 Running API Endpoints Tests")
    print("=" * 50)
    print("Testing note creation, editing, and search endpoints...")
    print()
    
    # Run the API endpoints tests
    from tests.api.test_api_endpoints import main
    asyncio.run(main())
