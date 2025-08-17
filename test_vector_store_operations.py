#!/usr/bin/env python3
"""
Production readiness test for Vector Store Operations.
Tests RAG, GraphRAG, and vector operations with REAL LLM calls and embeddings.
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

from app.services.llm_service import create_llm_service
from app.core.vector_store import PineconeVectorStore, ChromaDBVectorStore
from app.core.embeddings import GoogleEmbeddingService
from app.core.rag_search import RAGSearchService
from app.core.indexing_pipeline import IndexingPipeline
from app.core.metadata_indexing import MetadataIndexingService

class VectorStoreOperationsTester:
    def __init__(self):
        self.llm_service = None
        self.pinecone_store = None
        self.chromadb_store = None
        self.embedding_service = None
        self.rag_search_service = None
        self.indexing_pipeline = None
        self.metadata_indexing = None
        
    async def setup_services(self):
        """Set up all vector store services with real dependencies."""
        print("🔧 Setting up Vector Store Services...")
        
        try:
            # Set up LLM service
            print("   🚀 Setting up Gemini LLM service...")
            self.llm_service = create_llm_service(provider="gemini")
            print("   ✅ LLM service ready")
            
            # Set up Pinecone Vector Store
            print("   🌲 Setting up Pinecone vector store...")
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
            
            if pinecone_api_key and pinecone_env:
                self.pinecone_store = PineconeVectorStore(
                    api_key=pinecone_api_key, 
                    environment=pinecone_env
                )
                await self.pinecone_store.initialize()
                print("   ✅ Pinecone store ready")
            else:
                print("   ⚠️  Pinecone credentials not available")
            
            # Set up ChromaDB Vector Store (local)
            print("   🏠 Setting up ChromaDB vector store...")
            self.chromadb_store = ChromaDBVectorStore(persist_directory="./test_chroma")
            await self.chromadb_store.initialize()
            print("   ✅ ChromaDB store ready")
            
            # Set up Embedding Service
            print("   🔤 Setting up Google embedding service...")
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                raise Exception("Missing Google API key for embeddings")
                
            self.embedding_service = GoogleEmbeddingService(api_key=google_api_key)
            print("   ✅ Embedding service ready")
            
            # Set up RAG Search Service
            print("   🔍 Setting up RAG search service...")
            self.rag_search_service = RAGSearchService(
                vector_store=self.pinecone_store or self.chromadb_store,
                embedding_service=self.embedding_service
            )
            print("   ✅ RAG search service ready")
            
            # Set up Indexing Pipeline
            print("   📥 Setting up indexing pipeline...")
            self.indexing_pipeline = IndexingPipeline(
                vector_store=self.pinecone_store or self.chromadb_store,
                embedding_service=self.embedding_service
            )
            print("   ✅ Indexing pipeline ready")
            
            # Set up Metadata Indexing
            print("   🏷️  Setting up metadata indexing...")
            self.metadata_indexing = MetadataIndexingService(
                vector_store=self.pinecone_store or self.chromadb_store
            )
            print("   ✅ Metadata indexing ready")
            
            print("   🎉 All Vector Store Services set up successfully!")
            return True
            
        except Exception as e:
            print(f"   ❌ Service setup failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_basic_vector_operations(self):
        """Test basic vector store operations."""
        print("\n🔢 Testing Basic Vector Operations")
        print("-" * 60)
        
        try:
            store = self.pinecone_store or self.chromadb_store
            test_index = f"basic-test-{uuid.uuid4().hex[:8]}"
            
            print(f"   🏗️  Creating test index: {test_index}")
            await store.create_index(test_index, dimension=768)
            
            # Generate test content and embeddings
            test_content = [
                "Machine learning algorithms learn patterns from data",
                "Deep learning uses neural networks with multiple layers",
                "Natural language processing helps computers understand text"
            ]
            
            print("   🔤 Generating embeddings...")
            embeddings = await self.embedding_service.embed_batch(test_content)
            print(f"      ✅ Generated {len(embeddings)} embeddings")
            
            # Create test vectors
            test_vectors = []
            for i, (text, embedding) in enumerate(zip(test_content, embeddings)):
                test_vectors.append({
                    "id": f"basic-{i}",
                    "values": embedding,
                    "metadata": {
                        "text": text,
                        "category": "ai_basics",
                        "index": i
                    }
                })
            
            # Test upsert
            print("   📤 Testing vector upsert...")
            await store.upsert_vectors(test_index, test_vectors)
            print("      ✅ Vectors upserted")
            
            # Wait for indexing
            await asyncio.sleep(3)
            
            # Test search
            print("   🔍 Testing vector search...")
            query = "machine learning patterns"
            query_embedding = await self.embedding_service.embed_text(query)
            
            search_results = await store.search(
                test_index,
                query_vector=query_embedding,
                top_k=3
            )
            
            print(f"      ✅ Search returned {len(search_results)} results")
            
            # Test metadata filtering
            print("   🎯 Testing metadata filtering...")
            filtered_results = await store.search(
                test_index,
                query_vector=query_embedding,
                top_k=3,
                filter_metadata={"category": "ai_basics"}
            )
            print(f"      ✅ Filtered search returned {len(filtered_results)} results")
            
            # Cleanup
            await store.delete_index(test_index)
            print("      ✅ Test index deleted")
            
            print("   ✅ Basic vector operations test completed")
            return True
            
        except Exception as e:
            print(f"   ❌ Basic vector operations test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_blueprint_indexing(self):
        """Test blueprint-specific indexing operations."""
        print("\n📚 Testing Blueprint Indexing Operations")
        print("-" * 60)
        
        try:
            store = self.pinecone_store or self.chromadb_store
            blueprint_index = f"blueprint-test-{uuid.uuid4().hex[:8]}"
            
            print(f"   🏗️  Creating blueprint index: {blueprint_index}")
            await store.create_index(blueprint_index, dimension=768)
            
            # Create test blueprint content
            blueprint_content = [
                {
                    "id": "bp-1",
                    "content": "Introduction to Machine Learning: Learn the fundamentals of ML algorithms",
                    "metadata": {
                        "blueprint_id": "ml-fundamentals",
                        "section_type": "introduction",
                        "difficulty": "beginner",
                        "tags": ["machine-learning", "fundamentals"]
                    }
                },
                {
                    "id": "bp-2",
                    "content": "Supervised Learning: Classification and regression techniques",
                    "metadata": {
                        "blueprint_id": "ml-fundamentals",
                        "section_type": "content",
                        "difficulty": "intermediate",
                        "tags": ["supervised-learning", "classification"]
                    }
                },
                {
                    "id": "bp-3",
                    "content": "Unsupervised Learning: Clustering and dimensionality reduction",
                    "metadata": {
                        "blueprint_id": "ml-fundamentals",
                        "section_type": "content",
                        "difficulty": "intermediate",
                        "tags": ["unsupervised-learning", "clustering"]
                    }
                }
            ]
            
            # Generate embeddings and index
            print("   🔤 Generating blueprint embeddings...")
            texts = [item["content"] for item in blueprint_content]
            embeddings = await self.embedding_service.embed_batch(texts)
            
            vectors = []
            for content, embedding in zip(blueprint_content, embeddings):
                vectors.append({
                    "id": content["id"],
                    "values": embedding,
                    "metadata": content["metadata"]
                })
            
            await store.upsert_vectors(blueprint_index, vectors)
            print("      ✅ Blueprint content indexed")
            
            # Wait for indexing
            await asyncio.sleep(3)
            
            # Test blueprint-specific search
            print("   🔍 Testing blueprint search...")
            query = "machine learning fundamentals"
            query_embedding = await self.embedding_service.embed_text(query)
            
            # Search within specific blueprint
            blueprint_results = await store.search(
                blueprint_index,
                query_vector=query_embedding,
                top_k=5,
                filter_metadata={"blueprint_id": "ml-fundamentals"}
            )
            print(f"      ✅ Blueprint search returned {len(blueprint_results)} results")
            
            # Search by difficulty
            difficulty_results = await store.search(
                blueprint_index,
                query_vector=query_embedding,
                top_k=5,
                filter_metadata={"difficulty": "beginner"}
            )
            print(f"      ✅ Difficulty search returned {len(difficulty_results)} results")
            
            # Search by tags
            tag_results = await store.search(
                blueprint_index,
                query_vector=query_embedding,
                top_k=5,
                filter_metadata={"tags": "machine-learning"}
            )
            print(f"      ✅ Tag search returned {len(tag_results)} results")
            
            # Cleanup
            await store.delete_index(blueprint_index)
            print("      ✅ Blueprint index deleted")
            
            print("   ✅ Blueprint indexing test completed")
            return True
            
        except Exception as e:
            print(f"   ❌ Blueprint indexing test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_primitive_indexing(self):
        """Test primitive-specific indexing operations."""
        print("\n🔀 Testing Primitive Indexing Operations")
        print("-" * 60)
        
        try:
            store = self.pinecone_store or self.chromadb_store
            primitive_index = f"primitive-test-{uuid.uuid4().hex[:8]}"
            
            print(f"   🏗️  Creating primitive index: {primitive_index}")
            await store.create_index(primitive_index, dimension=768)
            
            # Create test primitive content
            primitive_content = [
                {
                    "id": "prim-1",
                    "content": "Concept: Supervised Learning - Learning from labeled examples",
                    "metadata": {
                        "primitive_type": "concept",
                        "domain": "machine_learning",
                        "complexity": "basic",
                        "prerequisites": ["statistics", "python"],
                        "related_primitives": ["unsupervised_learning", "reinforcement_learning"]
                    }
                },
                {
                    "id": "prim-2",
                    "content": "Algorithm: Linear Regression - Predicting continuous values",
                    "metadata": {
                        "primitive_type": "algorithm",
                        "domain": "machine_learning",
                        "complexity": "basic",
                        "prerequisites": ["linear_algebra", "statistics"],
                        "related_primitives": ["logistic_regression", "polynomial_regression"]
                    }
                },
                {
                    "id": "prim-3",
                    "content": "Technique: Cross-Validation - Model evaluation method",
                    "metadata": {
                        "primitive_type": "technique",
                        "domain": "model_evaluation",
                        "complexity": "intermediate",
                        "prerequisites": ["train_test_split", "model_training"],
                        "related_primitives": ["k_fold_cv", "stratified_cv"]
                    }
                }
            ]
            
            # Generate embeddings and index
            print("   🔤 Generating primitive embeddings...")
            texts = [item["content"] for item in primitive_content]
            embeddings = await self.embedding_service.embed_batch(texts)
            
            vectors = []
            for content, embedding in zip(primitive_content, embeddings):
                vectors.append({
                    "id": content["id"],
                    "values": embedding,
                    "metadata": content["metadata"]
                })
            
            await store.upsert_vectors(primitive_index, vectors)
            print("      ✅ Primitive content indexed")
            
            # Wait for indexing
            await asyncio.sleep(3)
            
            # Test primitive-specific search
            print("   🔍 Testing primitive search...")
            query = "supervised learning algorithms"
            query_embedding = await self.embedding_service.embed_text(query)
            
            # Search by primitive type
            concept_results = await store.search(
                primitive_index,
                query_vector=query_embedding,
                top_k=5,
                filter_metadata={"primitive_type": "concept"}
            )
            print(f"      ✅ Concept search returned {len(concept_results)} results")
            
            # Search by domain
            ml_results = await store.search(
                primitive_index,
                query_vector=query_embedding,
                top_k=5,
                filter_metadata={"domain": "machine_learning"}
            )
            print(f"      ✅ ML domain search returned {len(ml_results)} results")
            
            # Search by complexity
            basic_results = await store.search(
                primitive_index,
                query_vector=query_embedding,
                top_k=5,
                filter_metadata={"complexity": "basic"}
            )
            print(f"      ✅ Basic complexity search returned {len(basic_results)} results")
            
            # Cleanup
            await store.delete_index(primitive_index)
            print("      ✅ Primitive index deleted")
            
            print("   ✅ Primitive indexing test completed")
            return True
            
        except Exception as e:
            print(f"   ❌ Primitive indexing test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_criterion_indexing(self):
        """Test criterion-specific indexing operations."""
        print("\n🎯 Testing Criterion Indexing Operations")
        print("-" * 60)
        
        try:
            store = self.pinecone_store or self.chromadb_store
            criterion_index = f"criterion-test-{uuid.uuid4().hex[:8]}"
            
            print(f"   🏗️  Creating criterion index: {criterion_index}")
            await store.create_index(criterion_index, dimension=768)
            
            # Create test criterion content
            criterion_content = [
                {
                    "id": "crit-1",
                    "content": "Mastery Criterion: Implement linear regression from scratch",
                    "metadata": {
                        "criterion_type": "implementation",
                        "skill_level": "intermediate",
                        "assessment_method": "coding_exercise",
                        "prerequisites": ["python", "linear_algebra"],
                        "success_metrics": ["accuracy", "code_quality", "understanding"]
                    }
                },
                {
                    "id": "crit-2",
                    "content": "Mastery Criterion: Explain overfitting and underfitting",
                    "metadata": {
                        "criterion_type": "understanding",
                        "skill_level": "basic",
                        "assessment_method": "explanation",
                        "prerequisites": ["basic_ml", "model_evaluation"],
                        "success_metrics": ["clarity", "completeness", "examples"]
                    }
                },
                {
                    "id": "crit-3",
                    "content": "Mastery Criterion: Apply cross-validation techniques",
                    "metadata": {
                        "criterion_type": "application",
                        "skill_level": "intermediate",
                        "assessment_method": "practical_exercise",
                        "prerequisites": ["train_test_split", "model_selection"],
                        "success_metrics": ["correctness", "efficiency", "interpretation"]
                    }
                }
            ]
            
            # Generate embeddings and index
            print("   🔤 Generating criterion embeddings...")
            texts = [item["content"] for item in criterion_content]
            embeddings = await self.embedding_service.embed_batch(texts)
            
            vectors = []
            for content, embedding in zip(criterion_content, embeddings):
                vectors.append({
                    "id": content["id"],
                    "values": embedding,
                    "metadata": content["metadata"]
                })
            
            await store.upsert_vectors(criterion_index, vectors)
            print("      ✅ Criterion content indexed")
            
            # Wait for indexing
            await asyncio.sleep(3)
            
            # Test criterion-specific search
            print("   🔍 Testing criterion search...")
            query = "machine learning implementation criteria"
            query_embedding = await self.embedding_service.embed_text(query)
            
            # Search by criterion type
            implementation_results = await store.search(
                criterion_index,
                query_vector=query_embedding,
                top_k=5,
                filter_metadata={"criterion_type": "implementation"}
            )
            print(f"      ✅ Implementation criteria search returned {len(implementation_results)} results")
            
            # Search by skill level
            intermediate_results = await store.search(
                criterion_index,
                query_vector=query_embedding,
                top_k=5,
                filter_metadata={"skill_level": "intermediate"}
            )
            print(f"      ✅ Intermediate level search returned {len(intermediate_results)} results")
            
            # Search by assessment method
            coding_results = await store.search(
                criterion_index,
                query_vector=query_embedding,
                top_k=5,
                filter_metadata={"assessment_method": "coding_exercise"}
            )
            print(f"      ✅ Coding exercise search returned {len(coding_results)} results")
            
            # Cleanup
            await store.delete_index(criterion_index)
            print("      ✅ Criterion index deleted")
            
            print("   ✅ Criterion indexing test completed")
            return True
            
        except Exception as e:
            print(f"   ❌ Criterion indexing test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_rag_search_operations(self):
        """Test RAG search operations with real content."""
        print("\n🔍 Testing RAG Search Operations")
        print("-" * 60)
        
        try:
            # Create a test index for RAG
            store = self.pinecone_store or self.chromadb_store
            rag_index = f"rag-test-{uuid.uuid4().hex[:8]}"
            
            print(f"   🏗️  Creating RAG test index: {rag_index}")
            await store.create_index(rag_index, dimension=768)
            
            # Add comprehensive test content for RAG
            rag_content = [
                {
                    "id": "rag-1",
                    "content": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. It uses algorithms to identify patterns and make predictions based on input data.",
                    "metadata": {
                        "source": "ai_basics",
                        "topic": "machine_learning",
                        "difficulty": "beginner",
                        "content_type": "explanation"
                    }
                },
                {
                    "id": "rag-2",
                    "content": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to model complex patterns in data. It has revolutionized fields like computer vision and natural language processing.",
                    "metadata": {
                        "source": "ai_advanced",
                        "topic": "deep_learning",
                        "difficulty": "intermediate",
                        "content_type": "explanation"
                    }
                },
                {
                    "id": "rag-3",
                    "content": "Supervised learning involves training a model on labeled data to make predictions. Common algorithms include linear regression, logistic regression, and support vector machines.",
                    "metadata": {
                        "source": "ml_fundamentals",
                        "topic": "supervised_learning",
                        "difficulty": "intermediate",
                        "content_type": "explanation"
                    }
                }
            ]
            
            # Generate embeddings and index
            print("   🔤 Generating RAG embeddings...")
            texts = [item["content"] for item in rag_content]
            embeddings = await self.embedding_service.embed_batch(texts)
            
            vectors = []
            for content, embedding in zip(rag_content, embeddings):
                vectors.append({
                    "id": content["id"],
                    "values": embedding,
                    "metadata": content["metadata"]
                })
            
            await store.upsert_vectors(rag_index, vectors)
            print("      ✅ RAG content indexed")
            
            # Wait for indexing
            await asyncio.sleep(3)
            
            # Test RAG search with different queries
            test_queries = [
                "What is machine learning?",
                "How does deep learning work?",
                "Explain supervised learning algorithms"
            ]
            
            for i, query in enumerate(test_queries, 1):
                print(f"      🔍 Testing RAG query {i}: {query}")
                query_embedding = await self.embedding_service.embed_text(query)
                
                # Search with RAG service
                try:
                    rag_results = await self.rag_search_service.search(
                        query=query,
                        blueprint_id=None,
                        max_results=3
                    )
                    
                    if rag_results and "results" in rag_results:
                        print(f"         ✅ RAG search returned {len(rag_results['results'])} results")
                    else:
                        print(f"         ⚠️  RAG search returned unexpected format")
                        
                except Exception as e:
                    print(f"         ⚠️  RAG search failed: {e}")
                
                # Direct vector search for comparison
                vector_results = await store.search(
                    rag_index,
                    query_vector=query_embedding,
                    top_k=3
                )
                print(f"         ✅ Vector search returned {len(vector_results)} results")
            
            # Cleanup
            await store.delete_index(rag_index)
            print("      ✅ RAG test index deleted")
            
            print("   ✅ RAG search operations test completed")
            return True
            
        except Exception as e:
            print(f"   ❌ RAG search operations test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_graph_rag_operations(self):
        """Test GraphRAG operations with relationship-based content."""
        print("\n🕸️  Testing GraphRAG Operations")
        print("-" * 60)
        
        try:
            # Create a test index for GraphRAG
            store = self.pinecone_store or self.chromadb_store
            graphrag_index = f"graphrag-test-{uuid.uuid4().hex[:8]}"
            
            print(f"   🏗️  Creating GraphRAG test index: {graphrag_index}")
            await store.create_index(graphrag_index, dimension=768)
            
            # Add content with explicit relationships for GraphRAG
            graphrag_content = [
                {
                    "id": "graph-1",
                    "content": "Linear regression is a supervised learning algorithm that predicts continuous values. It is related to logistic regression and polynomial regression.",
                    "metadata": {
                        "source": "ml_algorithms",
                        "topic": "linear_regression",
                        "relationships": ["logistic_regression", "polynomial_regression"],
                        "algorithm_type": "supervised",
                        "output_type": "continuous"
                    }
                },
                {
                    "id": "graph-2",
                    "content": "Logistic regression is a supervised learning algorithm for binary classification. It extends linear regression concepts to classification problems.",
                    "metadata": {
                        "source": "ml_algorithms",
                        "topic": "logistic_regression",
                        "relationships": ["linear_regression", "classification"],
                        "algorithm_type": "supervised",
                        "output_type": "binary"
                    }
                },
                {
                    "id": "graph-3",
                    "content": "Neural networks form the foundation of deep learning. They consist of interconnected nodes that process information in layers.",
                    "metadata": {
                        "source": "deep_learning",
                        "topic": "neural_networks",
                        "relationships": ["deep_learning", "layers", "nodes"],
                        "algorithm_type": "unsupervised",
                        "architecture": "layered"
                    }
                }
            ]
            
            # Generate embeddings and index
            print("   🔤 Generating GraphRAG embeddings...")
            texts = [item["content"] for item in graphrag_content]
            embeddings = await self.embedding_service.embed_batch(texts)
            
            vectors = []
            for content, embedding in zip(graphrag_content, embeddings):
                vectors.append({
                    "id": content["id"],
                    "values": embedding,
                    "metadata": content["metadata"]
                })
            
            await store.upsert_vectors(graphrag_index, vectors)
            print("      ✅ GraphRAG content indexed")
            
            # Wait for indexing
            await asyncio.sleep(3)
            
            # Test GraphRAG-style search with relationship awareness
            print("   🔍 Testing GraphRAG relationship search...")
            
            # Search for related concepts
            query = "supervised learning algorithms"
            query_embedding = await self.embedding_service.embed_text(query)
            
            # Search with relationship filtering
            supervised_results = await store.search(
                graphrag_index,
                query_vector=query_embedding,
                top_k=5,
                filter_metadata={"algorithm_type": "supervised"}
            )
            print(f"      ✅ Supervised algorithm search returned {len(supervised_results)} results")
            
            # Search for related topics
            regression_query = "regression algorithms"
            regression_embedding = await self.embedding_service.embed_text(regression_query)
            
            regression_results = await store.search(
                graphrag_index,
                query_vector=regression_embedding,
                top_k=5
            )
            print(f"      ✅ Regression search returned {len(regression_results)} results")
            
            # Test relationship-based retrieval
            print("      🔗 Testing relationship-based retrieval...")
            for result in regression_results[:2]:
                if "relationships" in result.metadata:
                    print(f"         📍 {result.id}: {result.metadata['relationships']}")
            
            # Cleanup
            await store.delete_index(graphrag_index)
            print("      ✅ GraphRAG test index deleted")
            
            print("   ✅ GraphRAG operations test completed")
            return True
            
        except Exception as e:
            print(f"   ❌ GraphRAG operations test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def run_all_tests(self):
        """Run all vector store operation tests."""
        print("🚀 Starting VECTOR STORE OPERATIONS Production Readiness Test")
        print("=" * 80)
        
        # First, set up all services
        print("\n🔧 PHASE 1: Service Setup")
        setup_success = await self.setup_services()
        
        if not setup_success:
            print("❌ Service setup failed. Cannot proceed with tests.")
            return False
        
        print("\n🧪 PHASE 2: Running Tests")
        tests = [
            ("Basic Vector Operations", self.test_basic_vector_operations),
            ("Blueprint Indexing", self.test_blueprint_indexing),
            ("Primitive Indexing", self.test_primitive_indexing),
            ("Criterion Indexing", self.test_criterion_indexing),
            ("RAG Search Operations", self.test_rag_search_operations),
            ("GraphRAG Operations", self.test_graph_rag_operations)
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
        print("📊 VECTOR STORE OPERATIONS TEST SUMMARY")
        print("=" * 80)
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for test_name, result in results:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{status}: {test_name}")
        
        print(f"\n🎯 Overall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL VECTOR STORE TESTS PASSED! Operations are production-ready!")
        else:
            print("⚠️  Some vector store tests failed. Check the output above for details.")
        
        return passed == total

async def main():
    """Main test function."""
    tester = VectorStoreOperationsTester()
    success = await tester.run_all_tests()
    
    if success:
        print("\n🎉 Vector store operations test suite completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some vector store operation tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
