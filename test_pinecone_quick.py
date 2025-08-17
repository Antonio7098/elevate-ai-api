#!/usr/bin/env python3
"""
Quick test script for Pinecone vector store - core functionality only.
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

class QuickPineconeTester:
    def __init__(self):
        self.test_index_name = f"quick-test-{uuid.uuid4().hex[:8]}"
        
    async def test_pinecone_quick(self):
        """Quick test of Pinecone core functionality."""
        print("🌲 Quick Pinecone Test - Core Functionality")
        print("=" * 50)
        
        try:
            # Check environment variables
            api_key = os.getenv("PINECONE_API_KEY")
            environment = os.getenv("PINECONE_ENVIRONMENT")
            
            if not api_key or not environment:
                print("❌ Missing Pinecone environment variables")
                return False
            
            print(f"   🔑 API Key: {api_key[:10]}...{api_key[-10:] if len(api_key) > 20 else ''}")
            print(f"   🌍 Environment: {environment}")
            print(f"   🏷️  Test Index: {self.test_index_name}")
            
            # Initialize Pinecone vector store
            print("\n🚀 Initializing Pinecone client...")
            vector_store = PineconeVectorStore(api_key=api_key, environment=environment)
            await vector_store.initialize()
            print("   ✅ Pinecone client initialized")
            
            # Test index creation
            print("\n🏗️  Creating test index...")
            await vector_store.create_index(self.test_index_name, dimension=384)
            print("   ✅ Test index created")
            
            # Test vector upsert
            test_vectors = [
                {
                    "id": "quick-test-1",
                    "values": [0.1] * 384,
                    "metadata": {"text": "Quick test document 1", "category": "test"}
                },
                {
                    "id": "quick-test-2", 
                    "values": [0.2] * 384,
                    "metadata": {"text": "Quick test document 2", "category": "test"}
                }
            ]
            
            print(f"\n📤 Upserting {len(test_vectors)} test vectors...")
            await vector_store.upsert_vectors(self.test_index_name, test_vectors)
            print("   ✅ Vectors upserted")
            
            # Quick wait for indexing
            print("\n⏳ Quick indexing wait (3 seconds)...")
            await asyncio.sleep(3)
            
            # Test search
            print("\n🔍 Testing vector search...")
            search_results = await vector_store.search(
                self.test_index_name,
                query_vector=[0.1] * 384,
                top_k=2
            )
            print(f"   ✅ Search returned {len(search_results)} results")
            
            if search_results:
                print("   📊 First result:")
                print(f"      ID: {search_results[0].id}")
                print(f"      Score: {search_results[0].score:.6f}")
                print(f"      Content: {search_results[0].content}")
                print(f"      Metadata: {search_results[0].metadata}")
            
            # Test metadata filtering
            print("\n🎯 Testing metadata filtering...")
            filtered_results = await vector_store.search(
                self.test_index_name,
                query_vector=[0.1] * 384,
                top_k=2,
                filter_metadata={"category": "test"}
            )
            print(f"   ✅ Filtered search returned {len(filtered_results)} results")
            
            # Cleanup
            print("\n🧹 Cleaning up test index...")
            await vector_store.delete_index(self.test_index_name)
            print("   ✅ Test index deleted")
            
            print("\n🎉 Quick Pinecone test completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Quick Pinecone test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

async def main():
    """Main test function."""
    print("🚀 Starting Quick Pinecone Test")
    print("=" * 50)
    
    tester = QuickPineconeTester()
    success = await tester.test_pinecone_quick()
    
    if success:
        print("\n✅ Quick Pinecone test passed!")
        sys.exit(0)
    else:
        print("\n❌ Quick Pinecone test failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

