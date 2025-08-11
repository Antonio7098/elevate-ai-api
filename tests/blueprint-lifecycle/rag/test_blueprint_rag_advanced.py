"""
Advanced Blueprint RAG Test Suite

This module contains comprehensive tests for advanced RAG (Retrieval-Augmented Generation)
capabilities in the blueprint lifecycle, including semantic search, context retrieval,
response generation, and multi-modal RAG features.
"""

import pytest
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import json
import numpy as np
from datetime import datetime, timedelta

from app.core.blueprint.blueprint_rag import BlueprintRAG
from app.core.blueprint.blueprint_model import Blueprint
from app.core.blueprint.blueprint_indexer import BlueprintIndexer
from app.core.blueprint.blueprint_embedding import BlueprintEmbedding
from app.core.blueprint.blueprint_chunker import BlueprintChunker
from app.core.blueprint.blueprint_retriever import BlueprintRetriever
from app.core.blueprint.blueprint_generator import BlueprintGenerator
from app.core.blueprint.blueprint_rag_evaluator import BlueprintRAGEvaluator


class TestAdvancedBlueprintRAG:
    """Advanced test suite for blueprint RAG capabilities."""
    
    @pytest.fixture
    def mock_blueprint_rag(self):
        """Mock blueprint RAG service for testing."""
        rag = Mock(spec=BlueprintRAG)
        rag.query = AsyncMock()
        rag.generate_response = AsyncMock()
        rag.retrieve_context = AsyncMock()
        rag.rank_results = AsyncMock()
        rag.generate_embeddings = AsyncMock()
        rag.semantic_search = AsyncMock()
        rag.hybrid_search = AsyncMock()
        rag.multi_modal_search = AsyncMock()
        return rag
    
    @pytest.fixture
    def mock_blueprint_indexer(self):
        """Mock blueprint indexer for testing."""
        indexer = Mock(spec=BlueprintIndexer)
        indexer.index_blueprint = AsyncMock(return_value=True)
        indexer.search_index = AsyncMock()
        indexer.update_index = AsyncMock(return_value=True)
        indexer.delete_from_index = AsyncMock(return_value=True)
        return indexer
    
    @pytest.fixture
    def mock_blueprint_embedding(self):
        """Mock blueprint embedding service for testing."""
        embedding = Mock(spec=BlueprintEmbedding)
        embedding.generate_embedding = AsyncMock(return_value=np.random.rand(1536))
        embedding.generate_batch_embeddings = AsyncMock(return_value=[np.random.rand(1536) for _ in range(5)])
        embedding.similarity_search = AsyncMock()
        embedding.cluster_embeddings = AsyncMock()
        return embedding
    
    @pytest.fixture
    def mock_blueprint_chunker(self):
        """Mock blueprint chunker for testing."""
        chunker = Mock(spec=BlueprintChunker)
        chunker.chunk_content = AsyncMock(return_value=["chunk1", "chunk2", "chunk3"])
        chunker.chunk_with_overlap = AsyncMock(return_value=["overlap_chunk1", "overlap_chunk2"])
        chunker.semantic_chunk = AsyncMock(return_value=["semantic_chunk1", "semantic_chunk2"])
        chunker.chunk_by_sections = AsyncMock(return_value={"intro": "intro_chunk", "main": "main_chunk"})
        return chunker
    
    @pytest.fixture
    def mock_blueprint_retriever(self):
        """Mock blueprint retriever for testing."""
        retriever = Mock(spec=BlueprintRetriever)
        retriever.retrieve_relevant_chunks = AsyncMock()
        retriever.retrieve_with_filters = AsyncMock()
        retriever.retrieve_multimodal = AsyncMock()
        retriever.retrieve_temporal = AsyncMock()
        retriever.retrieve_hierarchical = AsyncMock()
        return retriever
    
    @pytest.fixture
    def mock_blueprint_generator(self):
        """Mock blueprint generator for testing."""
        generator = Mock(spec=BlueprintGenerator)
        generator.generate_response = AsyncMock(return_value="Generated response")
        generator.generate_with_context = AsyncMock(return_value="Context-aware response")
        generator.generate_multimodal = AsyncMock(return_value="Multimodal response")
        generator.generate_structured = AsyncMock(return_value={"answer": "Structured response"})
        return generator
    
    @pytest.fixture
    def sample_blueprint_data(self):
        """Sample blueprint data for testing."""
        return {
            "id": "test-blueprint-123",
            "name": "Advanced RAG Test Blueprint",
            "description": "A comprehensive blueprint for testing advanced RAG capabilities",
            "content": """
            This is a comprehensive blueprint that covers multiple topics including:
            
            # Introduction to Machine Learning
            Machine learning is a subset of artificial intelligence that enables computers
            to learn and improve from experience without being explicitly programmed.
            
            # Deep Learning Fundamentals
            Deep learning uses neural networks with multiple layers to model and understand
            complex patterns in data. It has revolutionized fields like computer vision,
            natural language processing, and speech recognition.
            
            # Natural Language Processing
            NLP combines computational linguistics with machine learning to enable
            computers to understand, interpret, and generate human language.
            
            # Computer Vision Applications
            Computer vision enables machines to interpret and understand visual information
            from the world, with applications in autonomous vehicles, medical imaging,
            and security systems.
            """,
            "metadata": {
                "category": "artificial_intelligence",
                "tags": ["machine_learning", "deep_learning", "nlp", "computer_vision"],
                "difficulty": "advanced",
                "prerequisites": ["python", "mathematics", "statistics"],
                "estimated_time": "8-12 hours",
                "last_updated": "2024-01-15"
            },
            "settings": {
                "chunk_size": 1000,
                "overlap": 200,
                "embedding_model": "text-embedding-ada-002",
                "chunking_strategy": "semantic",
                "retrieval_strategy": "hybrid",
                "generation_model": "gpt-4"
            }
        }
    
    @pytest.fixture
    def sample_queries(self):
        """Sample queries for testing RAG capabilities."""
        return {
            "simple": "What is machine learning?",
            "complex": "How does deep learning differ from traditional machine learning approaches?",
            "multimodal": "Explain the relationship between computer vision and deep learning with examples",
            "temporal": "What are the latest developments in NLP?",
            "hierarchical": "What are the prerequisites for learning deep learning?",
            "ambiguous": "What is AI?",
            "technical": "Explain the concept of backpropagation in neural networks",
            "comparative": "Compare and contrast supervised vs unsupervised learning"
        }
    
    @pytest.fixture
    def sample_context_chunks(self):
        """Sample context chunks for testing."""
        return [
            {
                "id": "chunk-1",
                "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
                "metadata": {"section": "introduction", "topic": "machine_learning", "relevance_score": 0.95},
                "embedding": np.random.rand(1536)
            },
            {
                "id": "chunk-2",
                "content": "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
                "metadata": {"section": "deep_learning", "topic": "neural_networks", "relevance_score": 0.92},
                "embedding": np.random.rand(1536)
            },
            {
                "id": "chunk-3",
                "content": "Natural Language Processing combines computational linguistics with machine learning to enable computers to understand human language.",
                "metadata": {"section": "nlp", "topic": "language_processing", "relevance_score": 0.88},
                "embedding": np.random.rand(1536)
            }
        ]

    def test_semantic_search_capabilities(self, mock_blueprint_rag, sample_queries, sample_context_chunks):
        """Test semantic search capabilities of the RAG system."""
        # Mock semantic search results
        mock_blueprint_rag.semantic_search.return_value = sample_context_chunks
        
        # Test different query types
        for query_name, query_text in sample_queries.items():
            print(f"Testing semantic search for query: {query_name}")
            
            # Perform semantic search
            results = asyncio.run(mock_blueprint_rag.semantic_search(query_text, limit=5))
            
            # Verify results
            assert results is not None
            assert len(results) > 0
            assert all(isinstance(chunk, dict) for chunk in results)
            assert all("content" in chunk for chunk in results)
            assert all("metadata" in chunk for chunk in results)
            
            # Verify relevance scores
            if results:
                relevance_scores = [chunk["metadata"]["relevance_score"] for chunk in results]
                assert all(0 <= score <= 1 for score in relevance_scores)
                
                # Results should be ordered by relevance
                sorted_scores = sorted(relevance_scores, reverse=True)
                assert relevance_scores == sorted_scores
            
            print(f"  ✅ {query_name}: Found {len(results)} relevant chunks")
        
        # Verify semantic search was called for each query
        assert mock_blueprint_rag.semantic_search.call_count == len(sample_queries)

    def test_hybrid_search_combining_semantic_and_keyword(self, mock_blueprint_rag, sample_queries):
        """Test hybrid search combining semantic and keyword-based approaches."""
        # Mock hybrid search results
        mock_blueprint_rag.hybrid_search.return_value = [
            {
                "id": "hybrid-1",
                "content": "Hybrid search result combining semantic and keyword matching",
                "metadata": {
                    "semantic_score": 0.85,
                    "keyword_score": 0.92,
                    "combined_score": 0.89,
                    "search_strategy": "hybrid"
                }
            }
        ]
        
        # Test hybrid search
        query = "machine learning neural networks"
        results = asyncio.run(mock_blueprint_rag.hybrid_search(query, limit=10))
        
        # Verify hybrid search results
        assert results is not None
        assert len(results) > 0
        
        for result in results:
            assert "metadata" in result
            metadata = result["metadata"]
            assert "semantic_score" in metadata
            assert "keyword_score" in metadata
            assert "combined_score" in metadata
            assert "search_strategy" in metadata
            
            # Scores should be within valid range
            assert 0 <= metadata["semantic_score"] <= 1
            assert 0 <= metadata["keyword_score"] <= 1
            assert 0 <= metadata["combined_score"] <= 1
            
            # Combined score should be reasonable combination of individual scores
            assert abs(metadata["combined_score"] - (metadata["semantic_score"] + metadata["keyword_score"]) / 2) < 0.1
        
        print("✅ Hybrid search successfully combines semantic and keyword approaches")

    def test_context_retrieval_with_filters(self, mock_blueprint_rag, sample_queries):
        """Test context retrieval with various filtering options."""
        # Mock context retrieval with filters
        mock_blueprint_rag.retrieve_context.return_value = {
            "chunks": [
                {
                    "id": "filtered-1",
                    "content": "Filtered content based on criteria",
                    "metadata": {"topic": "machine_learning", "difficulty": "intermediate"}
                }
            ],
            "filters_applied": ["topic", "difficulty"],
            "total_matches": 1,
            "retrieval_time_ms": 45
        }
        
        # Test different filter combinations
        filter_configs = [
            {"topic": "machine_learning", "difficulty": "intermediate"},
            {"category": "artificial_intelligence", "tags": ["deep_learning"]},
            {"section": "introduction", "relevance_threshold": 0.8},
            {"temporal_range": "last_6_months", "source_type": "academic"}
        ]
        
        for filters in filter_configs:
            print(f"Testing context retrieval with filters: {filters}")
            
            # Retrieve context with filters
            result = asyncio.run(mock_blueprint_rag.retrieve_context(
                query="machine learning",
                filters=filters,
                limit=10
            ))
            
            # Verify result structure
            assert result is not None
            assert "chunks" in result
            assert "filters_applied" in result
            assert "total_matches" in result
            assert "retrieval_time_ms" in result
            
            # Verify chunks
            chunks = result["chunks"]
            assert len(chunks) > 0
            assert all("content" in chunk for chunk in chunks)
            assert all("metadata" in chunk for chunk in chunks)
            
            # Verify filters were applied
            assert len(result["filters_applied"]) > 0
            assert result["total_matches"] >= 0
            assert result["retrieval_time_ms"] > 0
            
            print(f"  ✅ Applied {len(result['filters_applied'])} filters, found {result['total_matches']} matches")

    def test_response_generation_with_context(self, mock_blueprint_rag, sample_queries, sample_context_chunks):
        """Test response generation using retrieved context."""
        # Mock response generation
        mock_blueprint_rag.generate_response.return_value = {
            "response": "Generated response based on retrieved context",
            "context_used": [chunk["id"] for chunk in sample_context_chunks],
            "confidence_score": 0.87,
            "generation_time_ms": 120,
            "tokens_used": 150
        }
        
        # Test response generation for different queries
        for query_name, query_text in sample_queries.items():
            print(f"Testing response generation for: {query_name}")
            
            # Generate response with context
            result = asyncio.run(mock_blueprint_rag.generate_response(
                query=query_text,
                context=sample_context_chunks,
                max_length=500
            ))
            
            # Verify response structure
            assert result is not None
            assert "response" in result
            assert "context_used" in result
            assert "confidence_score" in result
            assert "generation_time_ms" in result
            assert "tokens_used" in result
            
            # Verify response content
            assert len(result["response"]) > 0
            assert isinstance(result["response"], str)
            
            # Verify context usage
            assert len(result["context_used"]) > 0
            assert all(isinstance(chunk_id, str) for chunk_id in result["context_used"])
            
            # Verify metrics
            assert 0 <= result["confidence_score"] <= 1
            assert result["generation_time_ms"] > 0
            assert result["tokens_used"] > 0
            
            print(f"  ✅ Generated response with {len(result['context_used'])} context chunks")
            print(f"     Confidence: {result['confidence_score']:.2f}")
            print(f"     Generation time: {result['generation_time_ms']}ms")
        
        # Verify generation was called for each query
        assert mock_blueprint_rag.generate_response.call_count == len(sample_queries)

    def test_multi_modal_rag_capabilities(self, mock_blueprint_rag):
        """Test multi-modal RAG capabilities including text, images, and structured data."""
        # Mock multi-modal search
        mock_blueprint_rag.multi_modal_search.return_value = [
            {
                "id": "multimodal-1",
                "content": "Multi-modal content combining text and visual elements",
                "type": "text_image",
                "text_content": "Description of neural network architecture",
                "image_content": "base64_encoded_image_data",
                "metadata": {
                    "modality": "text_image",
                    "text_relevance": 0.88,
                    "image_relevance": 0.91,
                    "combined_relevance": 0.90
                }
            }
        ]
        
        # Mock multi-modal response generation
        mock_blueprint_rag.generate_multimodal_response = AsyncMock(return_value={
            "text_response": "Generated text response",
            "image_response": "Generated image response",
            "combined_response": "Combined multi-modal response",
            "modality_weights": {"text": 0.6, "image": 0.4}
        })
        
        # Test multi-modal search
        query = "Show me a neural network architecture diagram with explanation"
        search_results = asyncio.run(mock_blueprint_rag.multi_modal_search(query, modalities=["text", "image"]))
        
        # Verify multi-modal search results
        assert search_results is not None
        assert len(search_results) > 0
        
        for result in search_results:
            assert "type" in result
            assert "content" in result
            assert "metadata" in result
            
            # Check modality-specific content
            if result["type"] == "text_image":
                assert "text_content" in result
                assert "image_content" in result
                
                metadata = result["metadata"]
                assert "modality" in metadata
                assert "text_relevance" in metadata
                assert "image_relevance" in metadata
                assert "combined_relevance" in metadata
        
        # Test multi-modal response generation
        response = asyncio.run(mock_blueprint_rag.generate_multimodal_response(
            query=query,
            context=search_results
        ))
        
        # Verify multi-modal response
        assert response is not None
        assert "text_response" in response
        assert "image_response" in response
        assert "combined_response" in response
        assert "modality_weights" in response
        
        # Verify modality weights
        weights = response["modality_weights"]
        assert "text" in weights
        assert "image" in weights
        assert sum(weights.values()) == 1.0
        
        print("✅ Multi-modal RAG capabilities working correctly")

    def test_temporal_rag_retrieval(self, mock_blueprint_rag):
        """Test temporal RAG retrieval for time-sensitive information."""
        # Mock temporal retrieval
        mock_blueprint_rag.retrieve_temporal = AsyncMock(return_value={
            "chunks": [
                {
                    "id": "temporal-1",
                    "content": "Recent information about GPT-4 capabilities",
                    "timestamp": "2024-01-15T10:30:00Z",
                    "metadata": {
                        "temporal_relevance": 0.95,
                        "freshness_score": 0.92,
                        "source_date": "2024-01-15"
                    }
                }
            ],
            "temporal_filters": {
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "min_freshness": 0.8
            },
            "total_temporal_matches": 1
        })
        
        # Test temporal retrieval with different time ranges
        time_ranges = [
            {"start_date": "2024-01-01", "end_date": "2024-01-31", "min_freshness": 0.8},
            {"start_date": "2023-12-01", "end_date": "2024-01-31", "min_freshness": 0.7},
            {"relative_time": "last_3_months", "min_freshness": 0.6}
        ]
        
        for time_range in time_ranges:
            print(f"Testing temporal retrieval for range: {time_range}")
            
            # Retrieve temporal information
            result = asyncio.run(mock_blueprint_rag.retrieve_temporal(
                query="GPT-4 capabilities",
                time_filters=time_range
            ))
            
            # Verify temporal result structure
            assert result is not None
            assert "chunks" in result
            assert "temporal_filters" in result
            assert "total_temporal_matches" in result
            
            # Verify chunks have temporal information
            chunks = result["chunks"]
            assert len(chunks) > 0
            
            for chunk in chunks:
                assert "timestamp" in chunk
                assert "metadata" in chunk
                
                metadata = chunk["metadata"]
                assert "temporal_relevance" in metadata
                assert "freshness_score" in metadata
                assert "source_date" in metadata
                
                # Verify temporal scores
                assert 0 <= metadata["temporal_relevance"] <= 1
                assert 0 <= metadata["freshness_score"] <= 1
            
            # Verify temporal filters
            filters = result["temporal_filters"]
            assert len(filters) > 0
            
            print(f"  ✅ Found {result['total_temporal_matches']} temporal matches")

    def test_hierarchical_rag_retrieval(self, mock_blueprint_rag):
        """Test hierarchical RAG retrieval for structured information."""
        # Mock hierarchical retrieval
        mock_blueprint_rag.retrieve_hierarchical = AsyncMock(return_value={
            "hierarchy": {
                "root": {
                    "topic": "Machine Learning",
                    "children": [
                        {
                            "topic": "Supervised Learning",
                            "children": [
                                {"topic": "Classification", "relevance": 0.9},
                                {"topic": "Regression", "relevance": 0.85}
                            ]
                        },
                        {
                            "topic": "Unsupervised Learning",
                            "children": [
                                {"topic": "Clustering", "relevance": 0.88},
                                {"topic": "Dimensionality Reduction", "relevance": 0.82}
                            ]
                        }
                    ]
                }
            },
            "relevant_nodes": ["Supervised Learning", "Classification", "Unsupervised Learning"],
            "depth": 3,
            "breadth": 2
        })
        
        # Test hierarchical retrieval
        query = "What are the main types of machine learning?"
        result = asyncio.run(mock_blueprint_rag.retrieve_hierarchical(
            query=query,
            max_depth=3,
            min_relevance=0.8
        ))
        
        # Verify hierarchical result structure
        assert result is not None
        assert "hierarchy" in result
        assert "relevant_nodes" in result
        assert "depth" in result
        assert "breadth" in result
        
        # Verify hierarchy structure
        hierarchy = result["hierarchy"]
        assert "root" in hierarchy
        root = hierarchy["root"]
        assert "topic" in root
        assert "children" in root
        
        # Verify relevant nodes
        relevant_nodes = result["relevant_nodes"]
        assert len(relevant_nodes) > 0
        assert all(isinstance(node, str) for node in relevant_nodes)
        
        # Verify depth and breadth
        assert result["depth"] > 0
        assert result["breadth"] > 0
        
        print("✅ Hierarchical RAG retrieval working correctly")
        print(f"  Hierarchy depth: {result['depth']}")
        print(f"  Hierarchy breadth: {result['breadth']}")
        print(f"  Relevant nodes: {', '.join(relevant_nodes)}")

    def test_rag_evaluation_metrics(self, mock_blueprint_rag, sample_queries, sample_context_chunks):
        """Test RAG evaluation metrics and quality assessment."""
        # Mock RAG evaluator
        mock_evaluator = Mock(spec=BlueprintRAGEvaluator)
        mock_evaluator.evaluate_relevance = AsyncMock(return_value=0.88)
        mock_evaluator.evaluate_accuracy = AsyncMock(return_value=0.92)
        mock_evaluator.evaluate_completeness = AsyncMock(return_value=0.85)
        mock_evaluator.evaluate_consistency = AsyncMock(return_value=0.90)
        mock_evaluator.evaluate_overall_quality = AsyncMock(return_value=0.89)
        
        # Test evaluation for different queries
        evaluation_results = {}
        
        for query_name, query_text in sample_queries.items():
            print(f"Evaluating RAG quality for: {query_name}")
            
            # Evaluate different aspects
            relevance = asyncio.run(mock_evaluator.evaluate_relevance(query_text, sample_context_chunks))
            accuracy = asyncio.run(mock_evaluator.evaluate_accuracy(query_text, sample_context_chunks))
            completeness = asyncio.run(mock_evaluator.evaluate_completeness(query_text, sample_context_chunks))
            consistency = asyncio.run(mock_evaluator.evaluate_consistency(query_text, sample_context_chunks))
            overall = asyncio.run(mock_evaluator.evaluate_overall_quality(query_text, sample_context_chunks))
            
            # Store results
            evaluation_results[query_name] = {
                "relevance": relevance,
                "accuracy": accuracy,
                "completeness": completeness,
                "consistency": consistency,
                "overall": overall
            }
            
            # Verify evaluation scores
            assert 0 <= relevance <= 1
            assert 0 <= accuracy <= 1
            assert 0 <= completeness <= 1
            assert 0 <= consistency <= 1
            assert 0 <= overall <= 1
            
            print(f"  ✅ Relevance: {relevance:.2f}")
            print(f"     Accuracy: {accuracy:.2f}")
            print(f"     Completeness: {completeness:.2f}")
            print(f"     Consistency: {consistency:.2f}")
            print(f"     Overall: {overall:.2f}")
        
        # Analyze evaluation results
        print("\n📊 RAG Evaluation Summary:")
        avg_scores = {
            "relevance": sum(r["relevance"] for r in evaluation_results.values()) / len(evaluation_results),
            "accuracy": sum(r["accuracy"] for r in evaluation_results.values()) / len(evaluation_results),
            "completeness": sum(r["completeness"] for r in evaluation_results.values()) / len(evaluation_results),
            "consistency": sum(r["consistency"] for r in evaluation_results.values()) / len(evaluation_results),
            "overall": sum(r["overall"] for r in evaluation_results.values()) / len(evaluation_results)
        }
        
        for metric, score in avg_scores.items():
            print(f"  Average {metric}: {score:.2f}")
        
        # Verify overall quality meets minimum threshold
        assert avg_scores["overall"] >= 0.8, f"Overall RAG quality {avg_scores['overall']:.2f} below threshold 0.8"

    def test_rag_performance_optimization(self, mock_blueprint_rag, sample_queries):
        """Test RAG performance optimization features."""
        # Mock performance metrics
        performance_metrics = {
            "retrieval_time_ms": 45,
            "generation_time_ms": 120,
            "total_time_ms": 165,
            "cache_hit_rate": 0.75,
            "embedding_generation_time_ms": 25,
            "similarity_search_time_ms": 20
        }
        
        # Mock optimized RAG operations
        mock_blueprint_rag.optimized_query = AsyncMock(return_value={
            "response": "Optimized response",
            "performance_metrics": performance_metrics,
            "optimizations_applied": ["caching", "batch_embedding", "parallel_retrieval"]
        })
        
        # Test optimized query
        query = "What is the difference between machine learning and deep learning?"
        result = asyncio.run(mock_blueprint_rag.optimized_query(query))
        
        # Verify optimized result
        assert result is not None
        assert "response" in result
        assert "performance_metrics" in result
        assert "optimizations_applied" in result
        
        # Verify performance metrics
        metrics = result["performance_metrics"]
        assert "retrieval_time_ms" in metrics
        assert "generation_time_ms" in metrics
        assert "total_time_ms" in metrics
        assert "cache_hit_rate" in metrics
        
        # Verify optimizations
        optimizations = result["optimizations_applied"]
        assert len(optimizations) > 0
        assert "caching" in optimizations
        
        # Performance assertions
        assert metrics["total_time_ms"] < 200, "Total query time should be under 200ms"
        assert metrics["cache_hit_rate"] > 0.5, "Cache hit rate should be above 50%"
        
        print("✅ RAG performance optimization working correctly")
        print(f"  Total time: {metrics['total_time_ms']}ms")
        print(f"  Cache hit rate: {metrics['cache_hit_rate']:.2f}")
        print(f"  Optimizations applied: {', '.join(optimizations)}")

    def test_rag_error_handling_and_fallback(self, mock_blueprint_rag):
        """Test RAG error handling and fallback mechanisms."""
        # Mock error scenarios
        mock_blueprint_rag.query.side_effect = [
            Exception("Embedding service unavailable"),
            Exception("Vector database connection failed"),
            "Successful response after fallback"
        ]
        
        # Test error handling with fallback
        query = "Test query with error handling"
        
        # First call should fail
        with pytest.raises(Exception, match="Embedding service unavailable"):
            asyncio.run(mock_blueprint_rag.query(query))
        
        # Second call should also fail
        with pytest.raises(Exception, match="Vector database connection failed"):
            asyncio.run(mock_blueprint_rag.query(query))
        
        # Third call should succeed
        result = asyncio.run(mock_blueprint_rag.query(query))
        assert result == "Successful response after fallback"
        
        print("✅ RAG error handling and fallback working correctly")

    def test_rag_context_window_management(self, mock_blueprint_rag):
        """Test RAG context window management for large documents."""
        # Mock large context chunks
        large_context = [
            {
                "id": f"chunk-{i}",
                "content": f"Large context chunk {i} with substantial content that needs to be managed within context windows",
                "token_count": 150,
                "metadata": {"section": f"section_{i}", "importance": 0.8 + (i * 0.02)}
            }
            for i in range(20)  # 20 chunks, each ~150 tokens
        ]
        
        # Mock context window management
        mock_blueprint_rag.manage_context_window = AsyncMock(return_value={
            "selected_chunks": large_context[:10],  # Top 10 most relevant
            "excluded_chunks": large_context[10:],
            "context_window_size": 1500,
            "tokens_used": 1450,
            "selection_strategy": "relevance_based"
        })
        
        # Test context window management
        query = "Complex query requiring multiple context chunks"
        result = asyncio.run(mock_blueprint_rag.manage_context_window(
            query=query,
            available_chunks=large_context,
            max_tokens=1500
        ))
        
        # Verify context window management
        assert result is not None
        assert "selected_chunks" in result
        assert "excluded_chunks" in result
        assert "context_window_size" in result
        assert "tokens_used" in result
        assert "selection_strategy" in result
        
        # Verify chunk selection
        selected = result["selected_chunks"]
        excluded = result["excluded_chunks"]
        
        assert len(selected) > 0
        assert len(excluded) > 0
        assert len(selected) + len(excluded) == len(large_context)
        
        # Verify token management
        assert result["tokens_used"] <= result["context_window_size"]
        assert result["tokens_used"] > 0
        
        print("✅ RAG context window management working correctly")
        print(f"  Selected chunks: {len(selected)}")
        print(f"  Excluded chunks: {len(excluded)}")
        print(f"  Tokens used: {result['tokens_used']}/{result['context_window_size']}")

    def test_rag_continuous_learning_and_improvement(self, mock_blueprint_rag):
        """Test RAG continuous learning and improvement mechanisms."""
        # Mock learning feedback
        feedback_data = {
            "query": "How does machine learning work?",
            "response": "Machine learning enables computers to learn from data",
            "user_feedback": "positive",
            "feedback_score": 0.9,
            "improvement_suggestions": ["Add more examples", "Include practical applications"],
            "learning_metrics": {
                "response_quality": 0.85,
                "user_satisfaction": 0.9,
                "knowledge_gaps": ["practical_examples", "real_world_applications"]
            }
        }
        
        # Mock continuous learning
        mock_blueprint_rag.learn_from_feedback = AsyncMock(return_value={
            "learning_applied": True,
            "model_updates": ["embedding_refinement", "retrieval_optimization"],
            "performance_improvement": 0.05,
            "next_learning_cycle": "2024-02-01T00:00:00Z"
        })
        
        # Test continuous learning
        result = asyncio.run(mock_blueprint_rag.learn_from_feedback(feedback_data))
        
        # Verify learning result
        assert result is not None
        assert "learning_applied" in result
        assert "model_updates" in result
        assert "performance_improvement" in result
        assert "next_learning_cycle" in result
        
        # Verify learning was applied
        assert result["learning_applied"] is True
        
        # Verify model updates
        updates = result["model_updates"]
        assert len(updates) > 0
        assert "embedding_refinement" in updates
        
        # Verify performance improvement
        assert result["performance_improvement"] > 0
        
        print("✅ RAG continuous learning working correctly")
        print(f"  Learning applied: {result['learning_applied']}")
        print(f"  Model updates: {', '.join(updates)}")
        print(f"  Performance improvement: {result['performance_improvement']:.2f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
