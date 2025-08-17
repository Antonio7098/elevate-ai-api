#!/usr/bin/env python3
"""
Test Runner for Vector Stores with REAL API calls.
Runs tests for Pinecone and ChromaDB vector stores.
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
    print("🚀 Running Vector Stores Tests")
    print("=" * 50)
    print("Testing Pinecone and ChromaDB vector stores...")
    print()
    
    # Run the vector store tests
    from tests.services.test_vector_stores import main
    asyncio.run(main())
