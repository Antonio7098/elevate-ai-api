#!/usr/bin/env python3
"""
Test Runner for RAG & Search Services with REAL API calls.
Runs tests for RAG, search, and knowledge retrieval services.
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
    print("🚀 Running RAG & Search Services Tests")
    print("=" * 50)
    print("Testing RAG, search, and knowledge retrieval services...")
    print()
    
    # Run the RAG and search tests
    from tests.services.test_rag_search import main
    asyncio.run(main())
