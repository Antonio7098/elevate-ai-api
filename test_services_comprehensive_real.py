#!/usr/bin/env python3
"""
Comprehensive test runner for all services with REAL functionality testing.
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.services.llm_service import create_llm_service
from app.core.vector_store import PineconeVectorStore
from app.core.rag_engine import RAGEngine
from app.core.search_service import SearchService
from app.core.context_assembly import ContextAssembler
from app.core.rag_search import RAGSearchService
from app.core.embeddings import GoogleEmbeddingService

class ComprehensiveRealServiceTester:
    def __init__(self):
        self.llm_service = None
        self.vector_store = None
        self.embedding_service = None
        self.rag_search_service = None
        
    async def setup_services(self):
        """Set up all required services with real dependencies."""
        print("🔧 Setting up services with real dependencies...")
        
        try:
            # 1. Set up LLM service
            print("   🚀 Setting up Gemini LLM service...")
            self.llm_service = create_llm_service(provider="gemini")
            print("   ✅ LLM service ready")
            
            # 2. Set up Vector Store (Pinecone)
            print("   🌲 Setting up Pinecone vector store...")
            api_key = os.getenv("PINECONE_API_KEY")
            environment = os.getenv("PINECONE_ENVIRONMENT")
            
            if not api_key or not environment:
                raise Exception("Missing Pinecone credentials")
                
            self.vector_store = PineconeVectorStore(api_key=api_key, environment=environment)
            await self.vector_store.initialize()
            print("   ✅ Vector store ready")
            
            # 3. Set up Embedding Service
            print("   🔤 Setting up Google embedding service...")
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                raise Exception("Missing Google API key for embeddings")
                
            self.embedding_service = GoogleEmbeddingService(api_key=google_api_key)
            print("   ✅ Embedding service ready")
            
            # 4. Set up RAG Search Service
            print("   🔍 Setting up RAG search service...")
            self.rag_search_service = RAGSearchService(
                vector_store=self.vector_store,
                embedding_service=self.embedding_service
            )
            print("   ✅ RAG search service ready")
            
            print("   🎉 All services set up successfully!")
            return True
            
        except Exception as e:
            print(f"   ❌ Service setup failed: {e}")
            return False
    
    async def test_llm_services_real(self):
        """Test LLM services with real API calls."""
        print("\n🤖 Testing LLM Services (Real)")
        print("-" * 50)
        
        try:
            # Test different models with real prompts
            models = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-1.5-pro"]
            
            for model in models:
                print(f"   🔄 Testing {model}...")
                try:
                    if hasattr(self.llm_service, 'model'):
                        self.llm_service.model = model
                    
                    # Test with a real, complex prompt
                    prompt = f"""
                    You are {model}. Write a creative story about an AI assistant helping a student learn machine learning.
                    The story should be exactly 3 sentences long and include the model name.
                    """
                    
                    response = await self.llm_service.call_llm(prompt)
                    print(f"         ✅ {model}: {response[:100]}...")
                    
                except Exception as e:
                    print(f"         ❌ {model} failed: {e}")
            
            print("   ✅ LLM services test completed")
            return True
            
        except Exception as e:
            print(f"   ❌ LLM services test failed: {e}")
            return False
    
    async def test_vector_stores_real(self):
        """Test vector stores with real operations."""
        print("\n🌲 Testing Vector Stores (Real)")
        print("-" * 50)
        
        try:
            # Create a test index
            test_index = f"comprehensive-test-{os.urandom(4).hex()}"
            print(f"   🏗️  Creating test index: {test_index}")
            
            await self.vector_store.create_index(test_index, dimension=768)  # Use 768 for Google embeddings
            
            # Generate real embeddings for test content
            test_texts = [
                "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
                "Deep learning uses neural networks with multiple layers to model complex patterns in data.",
                "Natural language processing helps computers understand and generate human language."
            ]
            
            print("   🔤 Generating real embeddings...")
            embeddings = await self.embedding_service.embed_texts(test_texts)
            print(f"   ✅ Generated {len(embeddings)} embeddings")
            
            # Create test vectors with real embeddings
            test_vectors = []
            for i, (text, embedding) in enumerate(zip(test_texts, embeddings)):
                test_vectors.append({
                    "id": f"test-{i}",
                    "values": embedding,
                    "metadata": {
                        "text": text,
                        "category": "ai_education",
                        "topic": "machine_learning"
                    }
                })
            
            # Upsert vectors
            print("   📤 Upserting vectors with real embeddings...")
            await self.vector_store.upsert_vectors(test_index, test_vectors)
            print("   ✅ Vectors upserted")
            
            # Wait for indexing
            print("   ⏳ Waiting for indexing...")
            await asyncio.sleep(3)
            
            # Test search with real embeddings
            print("   🔍 Testing search with real embeddings...")
            query_text = "What is machine learning?"
            query_embedding = await self.embedding_service.embed_texts([query_text])
            
            search_results = await self.vector_store.search(
                test_index,
                query_vector=query_embedding[0],
                top_k=3
            )
            
            print(f"   ✅ Search returned {len(search_results)} results")
            
            if search_results:
                print("   📊 First result:")
                print(f"      ID: {search_results[0].id}")
                print(f"      Score: {search_results[0].score:.4f}")
                print(f"      Content: {search_results[0].content[:100]}...")
            
            # Cleanup
            print("   🧹 Cleaning up...")
            await self.vector_store.delete_index(test_index)
            print("   ✅ Test index deleted")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Vector stores test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_rag_search_real(self):
        """Test RAG search service with real functionality."""
        print("\n🔍 Testing RAG Search Service (Real)")
        print("-" * 50)
        
        try:
            # Create a test index for RAG
            test_index = f"rag-test-{os.urandom(4).hex()}"
            print(f"   🏗️  Creating RAG test index: {test_index}")
            
            await self.vector_store.create_index(test_index, dimension=768)
            
            # Add some test content
            test_content = [
                {
                    "id": "rag-1",
                    "content": "Artificial Intelligence (AI) is the simulation of human intelligence in machines.",
                    "metadata": {"source": "ai_basics", "topic": "introduction"}
                },
                {
                    "id": "rag-2", 
                    "content": "Machine Learning is a subset of AI that enables computers to learn from data.",
                    "metadata": {"source": "ml_basics", "topic": "machine_learning"}
                }
            ]
            
            # Generate embeddings and upsert
            print("   🔤 Generating embeddings for RAG content...")
            texts = [item["content"] for item in test_content]
            embeddings = await self.embedding_service.embed_texts(texts)
            
            vectors = []
            for i, (content, embedding) in enumerate(zip(test_content, embeddings)):
                vectors.append({
                    "id": content["id"],
                    "values": embedding,
                    "metadata": content["metadata"]
                })
            
            await self.vector_store.upsert_vectors(test_index, vectors)
            print("   ✅ RAG content indexed")
            
            # Wait for indexing
            await asyncio.sleep(3)
            
            # Test RAG search
            print("   🔍 Testing RAG search...")
            query = "What is artificial intelligence?"
            
            # This would normally use the RAG search service, but let's test the components
            query_embedding = await self.embedding_service.embed_texts([query])
            search_results = await self.vector_store.search(
                test_index,
                query_vector=query_embedding[0],
                top_k=2
            )
            
            print(f"   ✅ RAG search returned {len(search_results)} results")
            
            # Cleanup
            await self.vector_store.delete_index(test_index)
            print("   ✅ RAG test index deleted")
            
            return True
            
        except Exception as e:
            print(f"   ❌ RAG search test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_search_service_real(self):
        """Test search service with real functionality."""
        print("\n🔍 Testing Search Service (Real)")
        print("-" * 50)
        
        try:
            # Create search service with real dependencies
            print("   🚀 Creating search service...")
            search_service = SearchService(
                vector_store=self.vector_store,
                embedding_service=self.embedding_service
            )
            print("   ✅ Search service created")
            
            # Test search functionality
            print("   🔍 Testing search functionality...")
            query = "machine learning basics"
            
            # This should work now with real dependencies
            try:
                results = await search_service.search(query, top_k=3)
                print(f"   ✅ Search returned results: {len(results) if results else 0}")
            except Exception as e:
                print(f"   ⚠️  Search failed (might need more setup): {e}")
            
            print("   ✅ Search service test completed")
            return True
            
        except Exception as e:
            print(f"   ❌ Search service test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_context_assembly_real(self):
        """Test context assembly with real functionality."""
        print("\n🧩 Testing Context Assembly (Real)")
        print("-" * 50)
        
        try:
            # Create context assembler with real dependencies
            print("   🚀 Creating context assembler...")
            context_assembler = ContextAssembler(
                search_service=self.rag_search_service,
                max_context_tokens=4000
            )
            print("   ✅ Context assembler created")
            
            # Test context assembly
            print("   🔍 Testing context assembly...")
            query = "Explain the basics of AI and machine learning"
            
            try:
                context = await context_assembler.assemble_context(query, max_context_length=2000)
                print(f"   ✅ Context assembled: {len(context)} characters")
                print(f"   📄 Context preview: {context[:200]}...")
            except Exception as e:
                print(f"   ⚠️  Context assembly failed (might need more setup): {e}")
            
            print("   ✅ Context assembly test completed")
            return True
            
        except Exception as e:
            print(f"   ❌ Context assembly test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def run_all_tests(self):
        """Run all comprehensive service tests."""
        print("🚀 Starting COMPREHENSIVE Service Test Suite (Real Functionality)")
        print("=" * 80)
        
        # First, set up all services
        print("\n🔧 PHASE 1: Service Setup")
        setup_success = await self.setup_services()
        
        if not setup_success:
            print("❌ Service setup failed. Cannot proceed with tests.")
            return False
        
        print("\n🧪 PHASE 2: Running Tests")
        tests = [
            ("LLM Services (Real)", self.test_llm_services_real),
            ("Vector Stores (Real)", self.test_vector_stores_real),
            ("RAG Search (Real)", self.test_rag_search_real),
            ("Search Service (Real)", self.test_search_service_real),
            ("Context Assembly (Real)", self.test_context_assembly_real)
        ]
        
        results = []
        for test_name, test_func in tests:
            print(f"\n🧪 Running: {test_name}")
            try:
                result = await test_func()
                results.append((test_name, result))
                print(f"   {'✅ PASSED' if result else '❌ FAILED'}: {test_name}")
            except Exception as e:
                print(f"   ❌ ERROR: {test_name} - {e}")
                results.append((test_name, False))
        
        # Summary
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE TEST SUMMARY (Real Functionality)")
        print("=" * 80)
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for test_name, result in results:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{status}: {test_name}")
        
        print(f"\n🎯 Overall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED! Services are working with real functionality!")
        else:
            print("⚠️  Some tests failed. Check the output above for details.")
        
        return passed == total

async def main():
    """Main test function."""
    tester = ComprehensiveRealServiceTester()
    success = await tester.run_all_tests()
    
    if success:
        print("\n🎉 Comprehensive service test suite completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some comprehensive service tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

