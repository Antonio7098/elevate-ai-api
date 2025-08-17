#!/usr/bin/env python3
"""
Test module for RAG and Search Services with REAL API calls.
Tests retrieval, augmentation, and search capabilities.
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

from app.core.rag_engine import RAGEngine
from app.core.search_service import SearchService
from app.core.vector_store import PineconeVectorStore
from app.services.llm_service import create_llm_service


class RAGSearchTester:
    """Test suite for RAG and search services with real API calls."""
    
    def __init__(self):
        self.test_results = []
        self.test_index_name = f"rag-test-{uuid.uuid4().hex[:8]}"
        
    async def test_rag_service(self):
        """Test RAG service with real API calls."""
        print("\n🔍 Testing RAG Service")
        print("-" * 50)
        
        try:
            # Initialize services
            llm_service = create_llm_service(provider="gemini")
            vector_store = PineconeVectorStore(
                api_key=os.getenv("PINECONE_API_KEY"),
                environment=os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp"),
                index_name=self.test_index_name
            )
            
            rag_service = RAGEngine(
                llm_service=llm_service,
                vector_store=vector_store
            )
            
            # Test document ingestion
            test_documents = [
                {
                    "id": "doc-1",
                    "content": "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
                    "metadata": {"source": "test", "category": "AI"}
                },
                {
                    "id": "doc-2",
                    "content": "Deep learning uses neural networks with multiple layers to model complex patterns in data.",
                    "metadata": {"source": "test", "category": "AI"}
                }
            ]
            
            print("   📚 Ingesting test documents...")
            await rag_service.ingest_documents(test_documents)
            print("   ✅ Documents ingested")
            
            # Test RAG query
            print("   🔍 Testing RAG query...")
            start_time = time.time()
            response = await rag_service.query(
                "What is the difference between machine learning and deep learning?",
                top_k=2
            )
            end_time = time.time()
            
            print(f"   ✅ RAG query successful")
            print(f"   ⏱️  Response time: {end_time - start_time:.2f}s")
            print(f"   📝 Response: {response[:200]}...")
            
            # Cleanup
            print("   🧹 Cleaning up test index...")
            await vector_store.delete_index()
            print("   ✅ Test index deleted")
            
            return True
            
        except Exception as e:
            print(f"❌ RAG service test failed: {e}")
            return False
    
    async def test_search_service(self):
        """Test search service with real API calls."""
        print("\n🔍 Testing Search Service")
        print("-" * 50)
        
        try:
            # Initialize search service
            search_service = SearchService()
            
            # Test semantic search
            print("   🔍 Testing semantic search...")
            start_time = time.time()
            results = await search_service.semantic_search(
                query="artificial intelligence applications",
                top_k=5
            )
            end_time = time.time()
            
            print(f"   ✅ Semantic search successful")
            print(f"   ⏱️  Response time: {end_time - start_time:.2f}s")
            print(f"   📊 Results: {len(results)} items")
            
            # Test hybrid search
            print("   🔍 Testing hybrid search...")
            start_time = time.time()
            hybrid_results = await search_service.hybrid_search(
                query="machine learning algorithms",
                top_k=3,
                semantic_weight=0.7
            )
            end_time = time.time()
            
            print(f"   ✅ Hybrid search successful")
            print(f"   ⏱️  Response time: {end_time - start_time:.2f}s")
            print(f"   📊 Results: {len(hybrid_results)} items")
            
            return True
            
        except Exception as e:
            print(f"❌ Search service test failed: {e}")
            return False
    
    async def test_knowledge_retrieval(self):
        """Test knowledge retrieval with real API calls."""
        print("\n🔍 Testing Knowledge Retrieval")
        print("-" * 50)
        
        try:
            from app.core.knowledge_service import KnowledgeService
            
            knowledge_service = KnowledgeService()
            
            # Test knowledge search
            print("   🔍 Testing knowledge search...")
            start_time = time.time()
            knowledge_results = await knowledge_service.search_knowledge(
                query="machine learning fundamentals",
                max_results=5
            )
            end_time = time.time()
            
            print(f"   ✅ Knowledge search successful")
            print(f"   ⏱️  Response time: {end_time - start_time:.2f}s")
            print(f"   📊 Results: {len(knowledge_results)} items")
            
            # Test knowledge synthesis
            if knowledge_results:
                print("   🔍 Testing knowledge synthesis...")
                start_time = time.time()
                synthesis = await knowledge_service.synthesize_knowledge(
                    knowledge_items=knowledge_results[:3],
                    synthesis_prompt="Summarize the key concepts"
                )
                end_time = time.time()
                
                print(f"   ✅ Knowledge synthesis successful")
                print(f"   ⏱️  Response time: {end_time - start_time:.2f}s")
                print(f"   📝 Synthesis: {synthesis[:150]}...")
            
            return True
            
        except Exception as e:
            print(f"❌ Knowledge retrieval test failed: {e}")
            return False
    
    async def test_context_assembly(self):
        """Test context assembly with real API calls."""
        print("\n🔍 Testing Context Assembly")
        print("-" * 50)
        
        try:
            from app.core.context_assembly_service import ContextAssemblyService
            
            context_service = ContextAssemblyService()
            
            # Test context building
            print("   🧩 Testing context building...")
            start_time = time.time()
            context = await context_service.build_context(
                query="Explain neural networks",
                max_context_length=1000
            )
            end_time = time.time()
            
            print(f"   ✅ Context assembly successful")
            print(f"   ⏱️  Response time: {end_time - start_time:.2f}s")
            print(f"   📏 Context length: {len(context)} characters")
            
            return True
            
        except Exception as e:
            print(f"❌ Context assembly test failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all RAG and search tests."""
        print("🚀 RAG and Search Services Test Suite")
        print("=" * 60)
        
        tests = [
            ("RAG Service", self.test_rag_service),
            ("Search Service", self.test_search_service),
            ("Knowledge Retrieval", self.test_knowledge_retrieval),
            ("Context Assembly", self.test_context_assembly)
        ]
        
        for test_name, test_func in tests:
            try:
                success = await test_func()
                self.test_results.append((test_name, success))
            except Exception as e:
                print(f"❌ {test_name} failed with exception: {e}")
                self.test_results.append((test_name, False))
        
        # Print summary
        print("\n📊 RAG and Search Test Results")
        print("-" * 40)
        passed = sum(1 for _, success in self.test_results if success)
        total = len(self.test_results)
        
        for test_name, success in self.test_results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"   {status} {test_name}")
        
        print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        return passed == total


async def main():
    """Run RAG and search tests."""
    tester = RAGSearchTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())

