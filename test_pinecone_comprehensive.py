#!/usr/bin/env python3
"""
Comprehensive test script for Pinecone vector store.
"""

import asyncio
import os
import sys
import uuid
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.core.vector_store import PineconeVectorStore

class ComprehensivePineconeTester:
    def __init__(self):
        self.test_index_name = f"test-index-{uuid.uuid4().hex[:8]}"
        
    async def test_pinecone_comprehensive(self):
        """Test Pinecone vector store comprehensively with real API calls."""
        print("🌲 Comprehensive Pinecone Vector Store Test")
        print("=" * 60)
        
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
            
            # Verify index exists
            exists = await vector_store.index_exists(self.test_index_name)
            print(f"   ✅ Index exists: {exists}")
            
            # Test vector upsert with more diverse data
            test_vectors = [
                {
                    "id": "pinecone-test-1",
                    "values": [0.1] * 384,
                    "metadata": {
                        "text": "Pinecone test document 1", 
                        "category": "test",
                        "topic": "machine learning",
                        "score": 0.95
                    }
                },
                {
                    "id": "pinecone-test-2", 
                    "values": [0.2] * 384,
                    "metadata": {
                        "text": "Pinecone test document 2", 
                        "category": "test",
                        "topic": "artificial intelligence",
                        "score": 0.87
                    }
                },
                {
                    "id": "pinecone-test-3",
                    "values": [0.3] * 384,
                    "metadata": {
                        "text": "Pinecone test document 3", 
                        "category": "test",
                        "topic": "deep learning",
                        "score": 0.92
                    }
                }
            ]
            
            print(f"\n📤 Upserting {len(test_vectors)} test vectors...")
            await vector_store.upsert_vectors(self.test_index_name, test_vectors)
            print("   ✅ Vectors upserted")
            
            # Wait longer for indexing (Pinecone can take time)
            print("\n⏳ Waiting for vectors to be indexed (10 seconds)...")
            for i in range(10):
                print(f"   ⏳ {i+1}/10 seconds...")
                await asyncio.sleep(1)
            
            # Test basic search
            print("\n🔍 Testing basic vector search...")
            search_results = await vector_store.search(
                self.test_index_name,
                query_vector=[0.1] * 384,
                top_k=5
            )
            print(f"   ✅ Search returned {len(search_results)} results")
            
            if search_results:
                print("   📊 First result:")
                print(f"      ID: {search_results[0].id}")
                print(f"      Score: {search_results[0].score}")
                print(f"      Content: {search_results[0].content}")
                print(f"      Metadata: {search_results[0].metadata}")
            
            # Test search with different query vector
            print("\n🔍 Testing search with different query vector...")
            search_results_2 = await vector_store.search(
                self.test_index_name,
                query_vector=[0.25] * 384,  # Closer to test-2 and test-3
                top_k=3
            )
            print(f"   ✅ Search returned {len(search_results_2)} results")
            
            # Test metadata filtering
            print("\n🎯 Testing metadata filtering by category...")
            filtered_results = await vector_store.search(
                self.test_index_name,
                query_vector=[0.1] * 384,
                top_k=5,
                filter_metadata={"category": "test"}
            )
            print(f"   ✅ Filtered search returned {len(filtered_results)} results")
            
            # Test metadata filtering by topic
            print("\n🎯 Testing metadata filtering by topic...")
            topic_filtered = await vector_store.search(
                self.test_index_name,
                query_vector=[0.1] * 384,
                top_k=5,
                filter_metadata={"topic": "machine learning"}
            )
            print(f"   ✅ Topic filtered search returned {len(topic_filtered)} results")
            
            # Test score-based filtering
            print("\n🎯 Testing score-based filtering...")
            score_filtered = await vector_store.search(
                self.test_index_name,
                query_vector=[0.1] * 384,
                top_k=5,
                filter_metadata={"score": {"$gte": 0.9}}
            )
            print(f"   ✅ Score filtered search returned {len(score_filtered)} results")
            
            # Test index statistics
            print("\n📊 Testing index statistics...")
            try:
                # This might not be available in all Pinecone implementations
                stats = await vector_store.get_index_stats(self.test_index_name)
                print(f"   ✅ Index stats: {stats}")
            except Exception as e:
                print(f"   ℹ️  Index stats not available: {e}")
            
            # Cleanup
            print("\n🧹 Cleaning up test index...")
            await vector_store.delete_index(self.test_index_name)
            print("   ✅ Test index deleted")
            
            print("\n🎉 Comprehensive Pinecone test completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Comprehensive Pinecone test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

async def main():
    """Main test function."""
    print("🚀 Starting Comprehensive Pinecone Vector Store Test")
    print("=" * 60)
    
    tester = ComprehensivePineconeTester()
    success = await tester.test_pinecone_comprehensive()
    
    if success:
        print("\n✅ All comprehensive Pinecone tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some comprehensive Pinecone tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
