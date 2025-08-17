#!/usr/bin/env python3
"""
Test script for Pinecone vector store only.
"""

import asyncio
import os
import sys
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.core.vector_store import PineconeVectorStore
from app.services.llm_service import create_llm_service

class PineconeTester:
    def __init__(self):
        self.test_index_name = f"test-index-{uuid.uuid4().hex[:8]}"
        
    async def test_pinecone(self):
        """Test Pinecone vector store with real API calls."""
        print("🌲 Testing Pinecone Vector Store")
        print("=" * 50)
        
        try:
            # Check environment variables
            api_key = os.getenv("PINECONE_API_KEY")
            environment = os.getenv("PINECONE_ENVIRONMENT")
            
            if not api_key or not environment:
                print("❌ Missing Pinecone environment variables")
                print("   PINECONE_API_KEY:", "✅" if api_key else "❌")
                print("   PINECONE_ENVIRONMENT:", "✅" if environment else "❌")
                return False
            
            print(f"   🔑 API Key: {api_key[:10]}...{api_key[-10:] if len(api_key) > 20 else ''}")
            print(f"   🌍 Environment: {environment}")
            
            # Initialize Pinecone vector store
            print("   🚀 Initializing Pinecone client...")
            vector_store = PineconeVectorStore(api_key=api_key, environment=environment)
            await vector_store.initialize()
            print("   ✅ Pinecone client initialized")
            
            # Test index creation
            print("   🏗️  Creating test index...")
            await vector_store.create_index(self.test_index_name, dimension=384)
            print("   ✅ Test index created")
            
            # Test vector upsert
            test_vectors = [
                {
                    "id": "pinecone-test-1",
                    "values": [0.1] * 384,
                    "metadata": {"text": "Pinecone test document 1", "category": "test"}
                },
                {
                    "id": "pinecone-test-2", 
                    "values": [0.2] * 384,
                    "metadata": {"text": "Pinecone test document 2", "category": "test"}
                }
            ]
            
            print("   📤 Upserting test vectors...")
            await vector_store.upsert_vectors(self.test_index_name, test_vectors)
            print("   ✅ Vectors upserted")
            
            # Wait a moment for indexing
            print("   ⏳ Waiting for vectors to be indexed...")
            await asyncio.sleep(2)
            
            # Test search
            print("   🔍 Testing vector search...")
            search_results = await vector_store.search(
                self.test_index_name,
                query_vector=[0.1] * 384,
                top_k=2
            )
            print(f"   ✅ Search returned {len(search_results)} results")
            
            # Test metadata filtering
            print("   🎯 Testing metadata filtering...")
            filtered_results = await vector_store.search(
                self.test_index_name,
                query_vector=[0.1] * 384,
                top_k=2,
                filter_metadata={"category": "test"}
            )
            print(f"   ✅ Filtered search returned {len(filtered_results)} results")
            
            # Test index existence check
            print("   ✅ Checking index existence...")
            exists = await vector_store.index_exists(self.test_index_name)
            print(f"   ✅ Index exists: {exists}")
            
            # Cleanup
            print("   🧹 Cleaning up test index...")
            await vector_store.delete_index(self.test_index_name)
            print("   ✅ Test index deleted")
            
            print("\n🎉 Pinecone test completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Pinecone test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

async def main():
    """Main test function."""
    print("🚀 Starting Pinecone Vector Store Test")
    print("=" * 50)
    
    tester = PineconeTester()
    success = await tester.test_pinecone()
    
    if success:
        print("\n✅ All Pinecone tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some Pinecone tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

