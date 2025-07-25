"""
RAG-optimized Semantic Search Implementation.

This module provides advanced search capabilities optimized for RAG (Retrieval-Augmented Generation)
including vector-based similarity search, metadata filtering, hybrid search, and intelligent re-ranking.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum

from app.core.vector_store import VectorStore, SearchResult, VectorStoreError
from app.core.embeddings import EmbeddingService, EmbeddingError
from app.core.query_transformer import QueryTransformer, QueryTransformation, QueryIntent
from app.models.text_node import TextNode, LocusType, UUEStage

logger = logging.getLogger(__name__)


class SearchStrategy(Enum):
    """Different search strategies for RAG retrieval."""
    EXACT_MATCH = "exact_match"
    SEMANTIC_BROAD = "semantic_broad"
    SEQUENTIAL_SEARCH = "sequential_search"
    MULTI_CONCEPT_SEARCH = "multi_concept_search"
    CONTEXTUAL_SEARCH = "contextual_search"
    ASSOCIATIVE_SEARCH = "associative_search"


@dataclass
class RAGSearchResult:
    """Enhanced search result for RAG applications."""
    id: str
    content: str
    score: float
    relevance_score: float
    diversity_score: float
    final_score: float
    blueprint_id: str
    locus_id: str
    locus_type: str
    uue_stage: str
    chunk_index: Optional[int]
    chunk_total: Optional[int]
    word_count: int
    relationships: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    search_reason: str  # Why this result was selected
    created_at: str
    indexed_at: str


@dataclass
class RAGSearchRequest:
    """Request for RAG-optimized search."""
    query: str
    user_context: Optional[Dict[str, Any]] = None
    conversation_history: Optional[List[Dict[str, Any]]] = None
    top_k: int = 10
    similarity_threshold: float = 0.7
    diversity_factor: float = 0.3
    include_relationships: bool = True
    max_chunk_size: Optional[int] = None
    min_chunk_size: Optional[int] = None
    blueprint_filter: Optional[str] = None


@dataclass
class RAGSearchResponse:
    """Response from RAG-optimized search."""
    results: List[RAGSearchResult]
    query_transformation: QueryTransformation
    total_results: int
    search_strategy: str
    search_time_ms: float
    embedding_time_ms: float
    reranking_time_ms: float
    filters_applied: Dict[str, Any]
    created_at: str


class RAGSearchService:
    """
    Advanced search service optimized for RAG applications.
    
    This service provides:
    1. Vector-based similarity search
    2. Metadata filtering for targeted retrieval
    3. Hybrid search (semantic + keyword)
    4. Intelligent re-ranking based on relevance and context
    5. Diversity-aware result selection
    """
    
    def __init__(self, vector_store: VectorStore, embedding_service: EmbeddingService):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.query_transformer = QueryTransformer(embedding_service)
        self.index_name = "blueprint-nodes"
        
        # Re-ranking weights
        self.reranking_weights = {
            'similarity': 0.4,
            'intent_relevance': 0.3,
            'diversity': 0.2,
            'context_fit': 0.1
        }
    
    async def search(self, request: RAGSearchRequest) -> RAGSearchResponse:
        """
        Perform RAG-optimized search with intelligent query transformation and re-ranking.
        
        Args:
            request: RAG search request
            
        Returns:
            RAGSearchResponse with optimized results
        """
        print(f"[DEBUG] === RAGSearchService.search STARTED ===")
        print(f"[DEBUG] Request query: {request.query}")
        print(f"[DEBUG] Request top_k: {request.top_k}")
        print(f"[DEBUG] Request similarity_threshold: {request.similarity_threshold}")
        print(f"[DEBUG] Query transformer available: {self.query_transformer is not None}")
        print(f"[DEBUG] Vector store available: {self.vector_store is not None}")
        
        start_time = time.time()
        
        try:
            # Step 1: Transform query using query transformer
            print(f"[DEBUG] Step 1: Starting query transformation...")
            transform_start = time.time()
            query_transformation = await self.query_transformer.transform_query(
                request.query, 
                request.user_context
            )
            transform_time = (time.time() - transform_start) * 1000
            print(f"[DEBUG] Query transformation completed in {transform_time:.2f}ms")
            print(f"[DEBUG] Intent: {query_transformation.intent}")
            print(f"[DEBUG] Confidence: {query_transformation.confidence}")
            print(f"[DEBUG] Search strategy: {query_transformation.search_strategy}")
            
            # Step 2: Get search parameters from transformation
            print(f"[DEBUG] Step 2: Getting search parameters...")
            search_params = self.query_transformer.get_search_parameters(query_transformation)
            print(f"[DEBUG] Search params: {search_params}")
            
            # Step 3: Perform multi-strategy search
            print(f"[DEBUG] Step 3: Starting multi-strategy search...")
            search_start = time.time()
            raw_results = await self._perform_multi_strategy_search(
                query_transformation, 
                search_params,
                request
            )
            search_time = (time.time() - search_start) * 1000
            print(f"[DEBUG] Multi-strategy search completed in {search_time:.2f}ms")
            print(f"[DEBUG] Raw results count: {len(raw_results)}")
            
            # Step 4: Re-rank results
            print(f"[DEBUG] Step 4: Starting re-ranking...")
            rerank_start = time.time()
            reranked_results = await self._rerank_results(
                raw_results, 
                query_transformation, 
                request
            )
            rerank_time = (time.time() - rerank_start) * 1000
            print(f"[DEBUG] Re-ranking completed in {rerank_time:.2f}ms")
            print(f"[DEBUG] Re-ranked results count: {len(reranked_results)}")
            
            # Step 5: Apply diversity filtering
            print(f"[DEBUG] Step 5: Applying diversity filtering...")
            final_results = self._apply_diversity_filtering(
                reranked_results, 
                request.diversity_factor,
                request.top_k
            )
            print(f"[DEBUG] Diversity filtering completed")
            print(f"[DEBUG] Final results count: {len(final_results)}")
            
            total_time = (time.time() - start_time) * 1000
            print(f"[DEBUG] RAG search completed in {total_time:.2f}ms total")
            
            response = RAGSearchResponse(
                results=final_results,
                query_transformation=query_transformation,
                total_results=len(final_results),
                search_strategy=search_params.get('search_strategy', 'semantic_broad'),
                search_time_ms=search_time,
                embedding_time_ms=transform_time,
                reranking_time_ms=rerank_time,
                filters_applied=search_params.get('metadata_filters', {}),
                created_at=datetime.utcnow().isoformat()
            )
            print(f"[DEBUG] === RAGSearchService.search COMPLETED ===")
            return response
            
        except Exception as e:
            print(f"[DEBUG] RAG search failed with error: {e}")
            print(f"[DEBUG] Error type: {type(e).__name__}")
            import traceback
            print(f"[DEBUG] Full traceback: {traceback.format_exc()}")
            logger.error(f"RAG search failed: {e}")
            raise RAGSearchServiceError(f"Search operation failed: {e}")
    
    async def _perform_multi_strategy_search(
        self, 
        transformation: QueryTransformation, 
        search_params: Dict[str, Any],
        request: RAGSearchRequest
    ) -> List[SearchResult]:
        """
        Perform search using multiple strategies based on query intent.
        """
        strategy_str = search_params.get('search_strategy', 'semantic_broad')
        print(f"[DEBUG] Multi-strategy search - Raw strategy: {strategy_str}")
        print(f"[DEBUG] Search params: {search_params}")
        print(f"[DEBUG] Query intent: {transformation.intent}")
        
        try:
            strategy = SearchStrategy(strategy_str)
            print(f"[DEBUG] Converted to enum: {strategy}")
        except ValueError as e:
            print(f"[DEBUG] Invalid search strategy '{strategy_str}': {e}")
            strategy = SearchStrategy.SEMANTIC_BROAD
        
        print(f"[DEBUG] Using search strategy: {strategy}")
        
        if strategy == SearchStrategy.EXACT_MATCH:
            print(f"[DEBUG] Calling _exact_match_search")
            return await self._exact_match_search(transformation, search_params, request)
        elif strategy == SearchStrategy.SEMANTIC_BROAD:
            print(f"[DEBUG] Calling _semantic_broad_search")
            return await self._semantic_broad_search(transformation, search_params, request)
        elif strategy == SearchStrategy.SEQUENTIAL_SEARCH:
            print(f"[DEBUG] Calling _sequential_search")
            return await self._sequential_search(transformation, search_params, request)
        elif strategy == SearchStrategy.MULTI_CONCEPT_SEARCH:
            print(f"[DEBUG] Calling _multi_concept_search")
            return await self._multi_concept_search(transformation, search_params, request)
        elif strategy == SearchStrategy.CONTEXTUAL_SEARCH:
            print(f"[DEBUG] Calling _contextual_search")
            return await self._contextual_search(transformation, search_params, request)
        elif strategy == SearchStrategy.ASSOCIATIVE_SEARCH:
            print(f"[DEBUG] Calling _associative_search")
            return await self._associative_search(transformation, search_params, request)
        else:
            print(f"[DEBUG] Fallback to _semantic_broad_search")
            return await self._semantic_broad_search(transformation, search_params, request)
    
    async def _exact_match_search(
        self, 
        transformation: QueryTransformation, 
        search_params: Dict[str, Any],
        request: RAGSearchRequest
    ) -> List[SearchResult]:
        """
        Perform exact match search with high precision.
        """
        print(f"[DEBUG] === Starting _exact_match_search ===")
        print(f"[DEBUG] Expanded query: {transformation.expanded_query}")
        print(f"[DEBUG] Original query: {transformation.original_query}")
        print(f"[DEBUG] Search params: {search_params}")
        
        # Get search parameters
        top_k = search_params.get('top_k', 5)
        similarity_threshold = search_params.get('similarity_threshold', 0.8)
        metadata_filters = search_params.get('metadata_filters', {})
        exact_match_boost = search_params.get('exact_match_boost', 1.0)
        
        print(f"[DEBUG] Search parameters:")
        print(f"[DEBUG]   top_k: {top_k}")
        print(f"[DEBUG]   similarity_threshold: {similarity_threshold}")
        print(f"[DEBUG]   metadata_filters: {metadata_filters}")
        print(f"[DEBUG]   exact_match_boost: {exact_match_boost}")
        print(f"[DEBUG] Vector store type: {type(self.vector_store)}")
        print(f"[DEBUG] Vector store available: {self.vector_store is not None}")
        
        # Generate embedding for the query
        print(f"[DEBUG] About to generate embedding for query...")
        try:
            query_embedding = await self.embedding_service.generate_embedding(transformation.expanded_query)
            print(f"[DEBUG] Query embedding generated successfully, dimension: {len(query_embedding)}")
        except Exception as e:
            print(f"[DEBUG] Failed to generate embedding: {e}")
            return []
        
        # First test search WITHOUT filters to debug
        print(f"[DEBUG] Testing search WITHOUT filters first...")
        try:
            test_results = await self.vector_store.search(
                index_name=self.index_name,
                query_vector=query_embedding,
                top_k=top_k
            )
            print(f"[DEBUG] Test search (no filters) returned {len(test_results)} results")
            for i, r in enumerate(test_results[:3]):
                print(f"[DEBUG]   Test result {i+1}: Score={r.score:.4f}, Content={r.content[:50]}...")
                print(f"[DEBUG]   Test metadata: {r.metadata}")
        except Exception as e:
            print(f"[DEBUG] Test search failed: {e}")
        
        # Fix metadata filters to match actual stored values
        print(f"[DEBUG] Original filters: {metadata_filters}")
        
        # Update locus_type to match actual stored values
        if 'locus_type' in metadata_filters:
            # Replace with actual stored locus_type values
            metadata_filters['locus_type'] = ['foundational_concept', 'key_propositions_and_facts', 'key_entities_and_definitions']
        
        # Remove difficulty_level filter as it doesn't exist in stored metadata
        if 'difficulty_level' in metadata_filters:
            print(f"[DEBUG] Removing difficulty_level filter (not in stored metadata)")
            del metadata_filters['difficulty_level']
        
        print(f"[DEBUG] Corrected filters: {metadata_filters}")
        
        # Now perform vector search with exact match emphasis
        print(f"[DEBUG] Now testing WITH corrected filters...")
        try:
            results = await self.vector_store.search(
                index_name=self.index_name,
                query_vector=query_embedding,
                top_k=top_k,
                filter_metadata=metadata_filters
            )
            print(f"[DEBUG] Vector search completed successfully!")
            print(f"[DEBUG] Raw results count: {len(results)}")
            print(f"[DEBUG] Raw results type: {type(results)}")
            
            # Log details of first few results
            for i, result in enumerate(results[:3]):
                print(f"[DEBUG] Raw result {i+1}: ID={result.id}, Score={result.score:.4f}, Content={result.content[:50]}...")
        
        except Exception as e:
            print(f"[DEBUG] Vector search failed with error: {e}")
            print(f"[DEBUG] Error type: {type(e).__name__}")
            import traceback
            print(f"[DEBUG] Full traceback: {traceback.format_exc()}")
            return []
        
        # Convert to SearchResult objects
        print(f"[DEBUG] Converting {len(results)} results to SearchResult objects...")
        search_results = []
        for i, result in enumerate(results):
            boosted_score = result.score * exact_match_boost
            search_result = SearchResult(
                id=result.id,
                content=result.content,
                metadata=result.metadata,
                score=boosted_score
            )
            search_results.append(search_result)
            print(f"[DEBUG] Converted result {i+1}: Score={result.score:.4f} → Boosted={boosted_score:.4f}")
        
        print(f"[DEBUG] Final search results count: {len(search_results)}")
        print(f"[DEBUG] === _exact_match_search COMPLETED ===")
        return search_results
    
    async def _semantic_broad_search(
        self, 
        transformation: QueryTransformation, 
        search_params: Dict[str, Any],
        request: RAGSearchRequest
    ) -> List[SearchResult]:
        """Perform broad semantic search for conceptual queries."""
        # Use expanded query for broader search
        query_embedding = await self.embedding_service.generate_embedding(transformation.expanded_query)
        
        metadata_filters = search_params.get('metadata_filters', {})
        
        # Get more results for broader search
        results = await self.vector_store.search(
            index_name=self.index_name,
            query_vector=query_embedding,
            top_k=search_params.get('top_k', 15) * 2,  # Get more for filtering
            filter_metadata=metadata_filters
        )
        
        return results
    
    async def _sequential_search(
        self, 
        transformation: QueryTransformation, 
        search_params: Dict[str, Any],
        request: RAGSearchRequest
    ) -> List[SearchResult]:
        """Perform sequential search for procedural queries."""
        # Focus on process-related content
        metadata_filters = search_params.get('metadata_filters', {})
        metadata_filters['locus_type'] = 'described_processes_and_steps'
        
        query_embedding = await self.embedding_service.generate_embedding(transformation.expanded_query)
        
        results = await self.vector_store.search(
            index_name=self.index_name,
            query_vector=query_embedding,
            top_k=search_params.get('top_k', 10),
            filter_metadata=metadata_filters
        )
        
        return results
    
    async def _multi_concept_search(
        self, 
        transformation: QueryTransformation, 
        search_params: Dict[str, Any],
        request: RAGSearchRequest
    ) -> List[SearchResult]:
        """Perform multi-concept search for comparative queries."""
        # Search with multiple reformulated queries
        all_results = []
        
        # Main query
        main_embedding = await self.embedding_service.generate_embedding(transformation.expanded_query)
        main_results = await self.vector_store.search(
            index_name=self.index_name,
            query_vector=main_embedding,
            top_k=search_params.get('top_k', 10),
            filter_metadata=search_params.get('metadata_filters', {})
        )
        all_results.extend(main_results)
        
        # Reformulated queries
        for reformulated in transformation.reformulated_queries[:3]:  # Top 3 reformulations
            try:
                reform_embedding = await self.embedding_service.generate_embedding(reformulated)
                reform_results = await self.vector_store.search(
                    index_name=self.index_name,
                    query_vector=reform_embedding,
                    top_k=5,  # Fewer results per reformulation
                    filter_metadata=search_params.get('metadata_filters', {})
                )
                all_results.extend(reform_results)
            except Exception as e:
                logger.warning(f"Reformulated query search failed: {e}")
        
        # Remove duplicates by ID
        unique_results = {}
        for result in all_results:
            if result.id not in unique_results or result.score > unique_results[result.id].score:
                unique_results[result.id] = result
        
        return list(unique_results.values())
    
    async def _contextual_search(
        self, 
        transformation: QueryTransformation, 
        search_params: Dict[str, Any],
        request: RAGSearchRequest
    ) -> List[SearchResult]:
        """Perform contextual search for analytical queries."""
        # Include conversation history in search context
        context_query = transformation.expanded_query
        
        if request.conversation_history:
            # Add recent conversation context
            recent_context = []
            for msg in request.conversation_history[-3:]:  # Last 3 messages
                if msg.get('role') == 'user':
                    recent_context.append(msg.get('content', ''))
            
            if recent_context:
                context_query += " " + " ".join(recent_context)
        
        query_embedding = await self.embedding_service.generate_embedding(context_query)
        
        results = await self.vector_store.search(
            index_name=self.index_name,
            query_vector=query_embedding,
            top_k=search_params.get('top_k', 15),
            filter_metadata=search_params.get('metadata_filters', {})
        )
        
        return results
    
    async def _associative_search(
        self, 
        transformation: QueryTransformation, 
        search_params: Dict[str, Any],
        request: RAGSearchRequest
    ) -> List[SearchResult]:
        """Perform associative search for creative queries."""
        # Use broader, more exploratory search
        metadata_filters = search_params.get('metadata_filters', {})
        
        # Search with lower threshold for more diverse results
        query_embedding = await self.embedding_service.generate_embedding(transformation.expanded_query)
        
        results = await self.vector_store.search(
            index_name=self.index_name,
            query_vector=query_embedding,
            top_k=search_params.get('top_k', 25),
            filter_metadata=metadata_filters
        )
        
        # Lower similarity threshold for more diverse results
        threshold = search_params.get('similarity_threshold', 0.5)
        return [r for r in results if r.score >= threshold]
    
    async def _rerank_results(
        self, 
        results: List[SearchResult], 
        transformation: QueryTransformation,
        request: RAGSearchRequest
    ) -> List[RAGSearchResult]:
        """
        Re-rank search results based on multiple factors.
        """
        rag_results = []
        
        for result in results:
            # Calculate different scoring factors
            similarity_score = result.score
            intent_relevance = self._calculate_intent_relevance(result, transformation)
            diversity_score = self._calculate_diversity_score(result, results)
            context_fit = self._calculate_context_fit(result, request)
            
            # Calculate final score
            final_score = (
                self.reranking_weights['similarity'] * similarity_score +
                self.reranking_weights['intent_relevance'] * intent_relevance +
                self.reranking_weights['diversity'] * diversity_score +
                self.reranking_weights['context_fit'] * context_fit
            )
            
            # Determine search reason
            search_reason = self._determine_search_reason(result, transformation, final_score)
            
            rag_result = RAGSearchResult(
                id=result.id,
                content=result.content,
                score=result.score,
                relevance_score=intent_relevance,
                diversity_score=diversity_score,
                final_score=final_score,
                blueprint_id=result.metadata.get("blueprint_id", ""),
                locus_id=result.metadata.get("locus_id", ""),
                locus_type=result.metadata.get("locus_type", ""),
                uue_stage=result.metadata.get("uue_stage", ""),
                chunk_index=result.metadata.get("chunk_index"),
                chunk_total=result.metadata.get("chunk_total"),
                word_count=result.metadata.get("word_count", 0),
                relationships=result.metadata.get("relationships", []),
                metadata=result.metadata,
                search_reason=search_reason,
                created_at=result.metadata.get("created_at", ""),
                indexed_at=result.metadata.get("indexed_at", "")
            )
            
            rag_results.append(rag_result)
        
        # Sort by final score
        rag_results.sort(key=lambda x: x.final_score, reverse=True)
        
        return rag_results
    
    def _calculate_intent_relevance(self, result: SearchResult, transformation: QueryTransformation) -> float:
        """Calculate how well the result matches the query intent."""
        locus_type = result.metadata.get("locus_type", "")
        intent = transformation.intent
        
        # Intent-based relevance scoring
        relevance_map = {
            QueryIntent.FACTUAL: {
                'key_propositions_and_facts': 1.0,
                'key_entities_and_definitions': 0.9,
                'described_processes_and_steps': 0.3,
                'identified_relationships': 0.3,
                'implicit_and_open_questions': 0.1
            },
            QueryIntent.CONCEPTUAL: {
                'key_propositions_and_facts': 0.9,
                'key_entities_and_definitions': 1.0,
                'described_processes_and_steps': 0.5,
                'identified_relationships': 0.7,
                'implicit_and_open_questions': 0.4
            },
            QueryIntent.PROCEDURAL: {
                'described_processes_and_steps': 1.0,
                'key_propositions_and_facts': 0.6,
                'key_entities_and_definitions': 0.4,
                'identified_relationships': 0.3,
                'implicit_and_open_questions': 0.2
            },
            QueryIntent.COMPARATIVE: {
                'key_propositions_and_facts': 0.8,
                'key_entities_and_definitions': 0.7,
                'identified_relationships': 1.0,
                'described_processes_and_steps': 0.5,
                'implicit_and_open_questions': 0.3
            },
            QueryIntent.ANALYTICAL: {
                'key_propositions_and_facts': 0.9,
                'identified_relationships': 0.8,
                'key_entities_and_definitions': 0.6,
                'described_processes_and_steps': 0.4,
                'implicit_and_open_questions': 0.7
            },
            QueryIntent.CREATIVE: {
                'implicit_and_open_questions': 1.0,
                'identified_relationships': 0.8,
                'key_propositions_and_facts': 0.5,
                'key_entities_and_definitions': 0.4,
                'described_processes_and_steps': 0.3
            }
        }
        
        return relevance_map.get(intent, {}).get(locus_type, 0.5)
    
    def _calculate_diversity_score(self, result: SearchResult, all_results: List[SearchResult]) -> float:
        """Calculate diversity score to promote varied results."""
        # Simple diversity based on content uniqueness
        content_words = set(result.content.lower().split())
        
        similarity_scores = []
        for other_result in all_results:
            if other_result.id != result.id:
                other_words = set(other_result.content.lower().split())
                intersection = len(content_words.intersection(other_words))
                union = len(content_words.union(other_words))
                similarity = intersection / union if union > 0 else 0
                similarity_scores.append(similarity)
        
        # Diversity is inverse of average similarity
        if similarity_scores:
            avg_similarity = sum(similarity_scores) / len(similarity_scores)
            return 1.0 - avg_similarity
        
        return 1.0  # Maximum diversity if no other results
    
    def _calculate_context_fit(self, result: SearchResult, request: RAGSearchRequest) -> float:
        """Calculate how well the result fits the conversation context."""
        # Simple context fit based on user context matching
        if not request.user_context:
            return 0.5  # Neutral score
        
        score = 0.5
        
        # Check UUE stage alignment
        if 'learning_stage' in request.user_context:
            user_stage = request.user_context['learning_stage']
            result_stage = result.metadata.get('uue_stage', '')
            if user_stage == result_stage:
                score += 0.3
        
        # Check subject area alignment
        if 'subject_area' in request.user_context:
            # This would require more sophisticated subject matching
            score += 0.2
        
        return min(score, 1.0)
    
    def _determine_search_reason(self, result: SearchResult, transformation: QueryTransformation, final_score: float) -> str:
        """Determine why this result was selected."""
        if result.score > 0.9:
            return "High semantic similarity"
        elif transformation.intent == QueryIntent.FACTUAL and result.metadata.get("locus_type") == "key_propositions_and_facts":
            return "Factual content match"
        elif transformation.intent == QueryIntent.PROCEDURAL and result.metadata.get("locus_type") == "described_processes_and_steps":
            return "Process information"
        elif final_score > 0.8:
            return "High overall relevance"
        else:
            return "Contextual relevance"
    
    def _apply_diversity_filtering(self, results: List[RAGSearchResult], diversity_factor: float, top_k: int) -> List[RAGSearchResult]:
        """Apply diversity filtering to promote varied results."""
        if diversity_factor == 0 or len(results) <= top_k:
            return results[:top_k]
        
        # Select diverse results using MMR-like approach
        selected = []
        remaining = results.copy()
        
        # Always select the top result
        if remaining:
            selected.append(remaining.pop(0))
        
        # Select remaining results balancing relevance and diversity
        while len(selected) < top_k and remaining:
            best_score = -1
            best_idx = -1
            
            for i, candidate in enumerate(remaining):
                # Calculate combined score
                relevance_score = candidate.final_score
                diversity_score = self._calculate_diversity_with_selected(candidate, selected)
                
                combined_score = ((1 - diversity_factor) * relevance_score + 
                                diversity_factor * diversity_score)
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_idx = i
            
            if best_idx >= 0:
                selected.append(remaining.pop(best_idx))
            else:
                break
        
        return selected
    
    def _calculate_diversity_with_selected(self, candidate: RAGSearchResult, selected: List[RAGSearchResult]) -> float:
        """Calculate diversity score with already selected results."""
        if not selected:
            return 1.0
        
        candidate_words = set(candidate.content.lower().split())
        
        max_similarity = 0
        for selected_result in selected:
            selected_words = set(selected_result.content.lower().split())
            intersection = len(candidate_words.intersection(selected_words))
            union = len(candidate_words.union(selected_words))
            similarity = intersection / union if union > 0 else 0
            max_similarity = max(max_similarity, similarity)
        
        return 1.0 - max_similarity


class RAGSearchServiceError(Exception):
    """Base exception for RAG search service operations."""
    pass
