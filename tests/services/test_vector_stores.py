#!/usr/bin/env python3
"""
Test module for Vector Stores with REAL API calls.
Tests Pinecone, ChromaDB, and vector operations.
"""

import asyncio
import os
import time
import uuid
from typing import Dict, Any, List

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded")
except ImportError:
    print("⚠️  python-dotenv not available")

from app.core.vector_store import PineconeVectorStore, ChromaDBVectorStore, VectorStore
from app.core.embeddings import EmbeddingService


class VectorStoresTester:
    """Test suite for vector stores with real API calls."""
    
    def __init__(self):
        self.test_results = []
        self.test_index_name = f"test-index-{uuid.uuid4().hex[:8]}"
        self.test_vectors = []
        
    async def test_pinecone_vector_store(self):
        """Test Pinecone vector store with real API calls."""
        print("\n🔍 Testing Pinecone Vector Store")
        print("-" * 50)
        
        try:
            api_key = os.getenv("PINECONE_API_KEY")
            environment = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
            
            if not api_key:
                print("❌ PINECONE_API_KEY not found")
                return False
            
            print(f"   🔑 Using environment: {environment}")
            
            # Initialize Pinecone
            vector_store = PineconeVectorStore(
                api_key=api_key,
                environment=environment,
                index_name=self.test_index_name
            )
            
            # Test index creation
            print("   🏗️  Creating test index...")
            await vector_store.create_index(self.test_index_name, dimension=384)
            print("   ✅ Test index created")
            
            # Test vector upsert
            test_vectors = [
                {
                    "id": "test-1",
                    "values": [0.1] * 384,
                    "metadata": {"text": "Test document 1", "category": "test"}
                },
                {
                    "id": "test-2", 
                    "values": [0.2] * 384,
                    "metadata": {"text": "Test document 2", "category": "test"}
                }
            ]
            
            print("   📤 Upserting test vectors...")
            await vector_store.upsert_vectors(self.test_index_name, test_vectors)
            print("   ✅ Vectors upserted")
            
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
            
            # Cleanup
            print("   🧹 Cleaning up test index...")
            await vector_store.delete_index(self.test_index_name)
            print("   ✅ Test index deleted")
            
            return True
            
        except Exception as e:
            print(f"❌ Pinecone test failed: {e}")
            return False
    
    async def test_chromadb_vector_store(self):
        """Test ChromaDB vector store (local)."""
        print("\n🔍 Testing ChromaDB Vector Store")
        print("-" * 50)
        
        try:
            # Initialize ChromaDB (local)
            vector_store = ChromaDBVectorStore(
                persist_directory="./test_chroma_db"
            )
            
            # Test collection creation
            print("   🏗️  Creating test collection...")
            collection_name = f"test-collection-{uuid.uuid4().hex[:8]}"
            await vector_store.create_index(collection_name, dimension=384)
            print("   ✅ Test collection created")
            
            # Test vector upsert
            test_vectors = [
                {
                    "id": "chroma-test-1",
                    "values": [0.1] * 384,
                    "metadata": {"text": "Chroma test document 1", "category": "test"}
                }
            ]
            
            print("   📤 Upserting test vectors...")
            await vector_store.upsert_vectors(self.test_index_name, test_vectors)
            print("   ✅ Vectors upserted")
            
            # Test search
            print("   🔍 Testing vector search...")
            search_results = await vector_store.search(
                collection_name,
                query_vector=[0.1] * 384,
                top_k=1
            )
            print(f"   ✅ Search returned {len(search_results)} results")
            
            # Cleanup
            print("   🧹 Cleaning up test collection...")
            await vector_store.delete_index(collection_name)
            print("   ✅ Test collection deleted")
            
            return True
            
        except Exception as e:
            print(f"❌ ChromaDB test failed: {e}")
            return False
    
    async def test_embedding_service(self):
        """Test embedding service with real API calls."""
        print("\n🔍 Testing Embedding Service")
        print("-" * 50)
        
        try:
            # Initialize embedding service
            embedding_service = EmbeddingService()
            
            # Test text embedding
            test_texts = [
                "This is a test document about machine learning.",
                "Another test document about artificial intelligence."
            ]
            
            print("   🔤 Generating embeddings...")
            start_time = time.time()
            embeddings = await embedding_service.embed_texts(test_texts)
            end_time = time.time()
            
            print(f"   ✅ Generated {len(embeddings)} embeddings")
            print(f"   ⏱️  Time: {end_time - start_time:.2f}s")
            print(f"   📏 Dimension: {len(embeddings[0])}")
            
            return True
            
        except Exception as e:
            print(f"❌ Embedding service test failed: {e}")
            return False
    
    async def test_vector_operations(self):
        """Test vector operations and utilities."""
        print("\n🔍 Testing Vector Operations")
        print("-" * 50)
        
        try:
            from app.core.vector_store import cosine_similarity
            
            # Test cosine similarity
            vec1 = [1.0, 0.0, 0.0]
            vec2 = [0.0, 1.0, 0.0]
            vec3 = [1.0, 0.0, 0.0]
            
            sim1 = cosine_similarity(vec1, vec2)  # Should be 0 (orthogonal)
            sim2 = cosine_similarity(vec1, vec3)  # Should be 1 (identical)
            
            print(f"   📐 Cosine similarity tests:")
            print(f"      Orthogonal vectors: {sim1:.3f} (expected: 0.0)")
            print(f"      Identical vectors: {sim2:.3f} (expected: 1.0)")
            
            if abs(sim1) < 0.01 and abs(sim2 - 1.0) < 0.01:
                print("   ✅ Cosine similarity working correctly")
                return True
            else:
                print("   ❌ Cosine similarity results unexpected")
                return False
                
        except Exception as e:
            print(f"❌ Vector operations test failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all vector store tests."""
        print("🚀 Vector Stores Test Suite")
        print("=" * 60)
        
        tests = [
            ("Pinecone Vector Store", self.test_pinecone_vector_store),
            ("ChromaDB Vector Store", self.test_chromadb_vector_store),
            ("Embedding Service", self.test_embedding_service),
            ("Vector Operations", self.test_vector_operations)
        ]
        
        for test_name, test_func in tests:
            try:
                success = await test_func()
                self.test_results.append((test_name, success))
            except Exception as e:
                print(f"❌ {test_name} failed with exception: {e}")
                self.test_results.append((test_name, False))
        
        # Print summary
        print("\n📊 Vector Stores Test Results")
        print("-" * 40)
        passed = sum(1 for _, success in self.test_results if success)
        total = len(self.test_results)
        
        for test_name, success in self.test_results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"   {status} {test_name}")
        
        print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        return passed == total


async def main():
    """Run vector stores tests."""
    tester = VectorStoresTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
