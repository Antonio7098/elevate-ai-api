"""
RAG-Fusion service for premium advanced retrieval features.
Implements multiple retrieval strategies and fusion algorithms.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import asyncio
from datetime import datetime

from .core_api_client import CoreAPIClient

@dataclass
class RetrievalResult:
    """Result from a single retrieval strategy"""
    content: str
    score: float
    source: str
    metadata: Dict[str, Any]
    strategy: str

@dataclass
class FusedResults:
    """Results from RAG-Fusion"""
    fused_chunks: List[RetrievalResult]
    strategy_scores: Dict[str, float]
    fusion_quality: float
    user_context: Dict[str, Any]
    timestamp: datetime

@dataclass
class AdaptiveResults:
    """Results from adaptive fusion"""
    results: FusedResults
    strategy_used: str
    adaptation_reason: str
    performance_metrics: Dict[str, float]

class DenseRetriever:
    """Dense vector retriever"""
    
    async def retrieve(self, query: str, user_id: str = None) -> List[RetrievalResult]:
        """Retrieve using dense embeddings"""
        # Mock implementation - in production, use actual vector search
        return [
            RetrievalResult(
                content=f"Dense retrieval result for: {query}",
                score=0.85,
                source="vector_db",
                metadata={"embedding_model": "text-embedding-3-small"},
                strategy="dense"
            ),
            RetrievalResult(
                content=f"Second dense result for: {query}",
                score=0.78,
                source="vector_db",
                metadata={"embedding_model": "text-embedding-3-small"},
                strategy="dense"
            )
        ]

class SparseRetriever:
    """Sparse keyword retriever"""
    
    async def retrieve(self, query: str, user_id: str = None) -> List[RetrievalResult]:
        """Retrieve using sparse keyword search"""
        # Mock implementation - in production, use BM25 or similar
        return [
            RetrievalResult(
                content=f"Sparse keyword result for: {query}",
                score=0.82,
                source="keyword_index",
                metadata={"algorithm": "BM25"},
                strategy="sparse"
            )
        ]

class HybridRetriever:
    """Hybrid dense + sparse retriever"""
    
    async def retrieve(self, query: str, user_id: str = None) -> List[RetrievalResult]:
        """Retrieve using hybrid approach"""
        # Mock implementation - in production, combine dense and sparse
        return [
            RetrievalResult(
                content=f"Hybrid result for: {query}",
                score=0.88,
                source="hybrid_search",
                metadata={"dense_weight": 0.7, "sparse_weight": 0.3},
                strategy="hybrid"
            )
        ]

class GraphRetriever:
    """Graph-based retriever using knowledge graph"""
    
    async def retrieve(self, query: str, user_id: str = None) -> List[RetrievalResult]:
        """Retrieve using knowledge graph traversal"""
        # Mock implementation - in production, use Neo4j or similar
        return [
            RetrievalResult(
                content=f"Graph-based result for: {query}",
                score=0.83,
                source="knowledge_graph",
                metadata={"graph_depth": 2, "relationship_types": ["PREREQUISITE", "RELATED"]},
                strategy="graph"
            )
        ]

class SemanticRetriever:
    """Semantic similarity retriever"""
    
    async def retrieve(self, query: str, user_id: str = None) -> List[RetrievalResult]:
        """Retrieve using semantic similarity"""
        # Mock implementation - in production, use semantic search
        return [
            RetrievalResult(
                content=f"Semantic result for: {query}",
                score=0.87,
                source="semantic_index",
                metadata={"similarity_threshold": 0.8},
                strategy="semantic"
            )
        ]

class CoreAPIRetriever:
    """Retriever that uses Core API data"""
    
    def __init__(self):
        self.core_api_client = CoreAPIClient()
    
    async def retrieve(self, query: str, user_id: str, user_analytics: Dict[str, Any]) -> List[RetrievalResult]:
        """Retrieve using Core API data and user context"""
        try:
            # Get user's learning analytics and memory insights
            memory_insights = await self.core_api_client.get_user_memory_insights(user_id)
            learning_paths = await self.core_api_client.get_user_learning_paths(user_id)
            
            # Use user context to enhance retrieval
            results = []
            
            # Add results based on user's learning path
            for path in learning_paths:
                if any(keyword in query.lower() for keyword in path.get("keywords", [])):
                    results.append(RetrievalResult(
                        content=f"Learning path content for: {query}",
                        score=0.9,
                        source="core_api_learning_path",
                        metadata={"path_id": path.get("id"), "progress": path.get("progress", 0)},
                        strategy="core_api"
                    ))
            
            # Add results based on memory insights
            if memory_insights:
                results.append(RetrievalResult(
                    content=f"Memory-based content for: {query}",
                    score=0.86,
                    source="core_api_memory",
                    metadata={"insight_type": "learning_pattern"},
                    strategy="core_api"
                ))
            
            return results
            
        except Exception as e:
            print(f"Error in Core API retrieval: {e}")
            return []

class ReciprocalRankFusion:
    """Reciprocal Rank Fusion algorithm for combining results"""
    
    def fuse(self, results: Dict[str, List[RetrievalResult]], k: int = 60) -> List[RetrievalResult]:
        """Fuse results using RRF algorithm"""
        try:
            # Create a dictionary to store fused scores
            fused_scores = {}
            
            # Calculate RRF scores for each result
            for strategy, strategy_results in results.items():
                for i, result in enumerate(strategy_results):
                    # RRF formula: 1 / (k + rank)
                    rrf_score = 1 / (k + i + 1)
                    
                    # Use content as key for deduplication
                    content_key = result.content[:100]  # First 100 chars as key
                    
                    if content_key in fused_scores:
                        fused_scores[content_key]["score"] += rrf_score
                        fused_scores[content_key]["strategies"].append(strategy)
                    else:
                        fused_scores[content_key] = {
                            "result": result,
                            "score": rrf_score,
                            "strategies": [strategy]
                        }
            
            # Convert back to list and sort by score
            fused_results = []
            for content_key, fused_data in fused_scores.items():
                result = fused_data["result"]
                result.score = fused_data["score"]
                result.metadata["fusion_strategies"] = fused_data["strategies"]
                fused_results.append(result)
            
            # Sort by fused score
            fused_results.sort(key=lambda x: x.score, reverse=True)
            
            return fused_results
            
        except Exception as e:
            print(f"Error in RRF fusion: {e}")
            # Fallback: concatenate all results
            all_results = []
            for strategy_results in results.values():
                all_results.extend(strategy_results)
            return all_results

class RAGFusionService:
    """RAG-Fusion service for premium users"""
    
    def __init__(self):
        self.retrievers = {
            'dense': DenseRetriever(),
            'sparse': SparseRetriever(),
            'hybrid': HybridRetriever(),
            'graph': GraphRetriever(),
            'semantic': SemanticRetriever(),
            'core_api': CoreAPIRetriever()
        }
        self.fusion_strategy = ReciprocalRankFusion()
        self.core_api_client = CoreAPIClient()
    
    async def multi_retrieve(self, query: str, user_id: str) -> FusedResults:
        """Retrieve using multiple strategies and fuse results with Core API context"""
        try:
            # Get user's learning context from Core API
            user_analytics = await self.core_api_client.get_user_learning_analytics(user_id)
            memory_insights = await self.core_api_client.get_user_memory_insights(user_id)
            
            # Use Core API data to enhance retrieval
            results = {}
            for name, retriever in self.retrievers.items():
                if name == 'core_api':
                    results[name] = await retriever.retrieve(query, user_id, user_analytics)
                else:
                    results[name] = await retriever.retrieve(query, user_id)
            
            # Fuse results
            fused_chunks = self.fusion_strategy.fuse(results)
            
            # Calculate strategy scores
            strategy_scores = {}
            for strategy, strategy_results in results.items():
                if strategy_results:
                    strategy_scores[strategy] = sum(r.score for r in strategy_results) / len(strategy_results)
                else:
                    strategy_scores[strategy] = 0.0
            
            # Calculate fusion quality
            fusion_quality = sum(strategy_scores.values()) / len(strategy_scores) if strategy_scores else 0.0
            
            return FusedResults(
                fused_chunks=fused_chunks,
                strategy_scores=strategy_scores,
                fusion_quality=fusion_quality,
                user_context={"analytics": user_analytics, "insights": memory_insights},
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            print(f"Error in multi_retrieve: {e}")
            return FusedResults(
                fused_chunks=[],
                strategy_scores={},
                fusion_quality=0.0,
                user_context={},
                timestamp=datetime.utcnow()
            )
    
    async def adaptive_fusion(self, query: str, user_id: str) -> AdaptiveResults:
        """Adapt fusion strategy based on Core API user context"""
        try:
            # Get user's learning efficiency and preferences
            user_memory = await self.core_api_client.get_user_memory(user_id)
            analytics = await self.core_api_client.get_user_learning_analytics(user_id)
            
            # Adapt strategy based on user's cognitive profile
            if user_memory.get('cognitiveApproach') == 'TOP_DOWN':
                # Prefer graph and semantic retrieval for big picture
                strategy = 'graph_semantic_heavy'
                adaptation_reason = "User prefers top-down learning approach"
            elif analytics.get('learningEfficiency', 0) > 0.8:
                # High efficiency users get more complex fusion
                strategy = 'complex_fusion'
                adaptation_reason = "High learning efficiency user"
            else:
                # Default strategy
                strategy = 'balanced_fusion'
                adaptation_reason = "Default balanced approach"
            
            # Apply the selected strategy
            results = await self.apply_strategy(strategy, query, user_id)
            
            return AdaptiveResults(
                results=results,
                strategy_used=strategy,
                adaptation_reason=adaptation_reason,
                performance_metrics={"adaptation_quality": 0.85}
            )
            
        except Exception as e:
            print(f"Error in adaptive_fusion: {e}")
            # Fallback to standard fusion
            results = await self.multi_retrieve(query, user_id)
            return AdaptiveResults(
                results=results,
                strategy_used="fallback",
                adaptation_reason="Error in adaptation, using fallback",
                performance_metrics={"adaptation_quality": 0.0}
            )
    
    async def apply_strategy(self, strategy: str, query: str, user_id: str) -> FusedResults:
        """Apply specific fusion strategy"""
        try:
            if strategy == 'graph_semantic_heavy':
                # Focus on graph and semantic retrieval
                results = {
                    'graph': await self.retrievers['graph'].retrieve(query, user_id),
                    'semantic': await self.retrievers['semantic'].retrieve(query, user_id),
                    'core_api': await self.retrievers['core_api'].retrieve(query, user_id, {})
                }
            elif strategy == 'complex_fusion':
                # Use all retrievers with higher weights
                results = {}
                for name, retriever in self.retrievers.items():
                    if name == 'core_api':
                        results[name] = await retriever.retrieve(query, user_id, {})
                    else:
                        results[name] = await retriever.retrieve(query, user_id)
            else:
                # Balanced fusion - use all retrievers
                return await self.multi_retrieve(query, user_id)
            
            # Fuse results
            fused_chunks = self.fusion_strategy.fuse(results)
            
            # Calculate strategy scores
            strategy_scores = {}
            for strategy_name, strategy_results in results.items():
                if strategy_results:
                    strategy_scores[strategy_name] = sum(r.score for r in strategy_results) / len(strategy_results)
                else:
                    strategy_scores[strategy_name] = 0.0
            
            # Calculate fusion quality
            fusion_quality = sum(strategy_scores.values()) / len(strategy_scores) if strategy_scores else 0.0
            
            return FusedResults(
                fused_chunks=fused_chunks,
                strategy_scores=strategy_scores,
                fusion_quality=fusion_quality,
                user_context={},
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            print(f"Error applying strategy {strategy}: {e}")
            return await self.multi_retrieve(query, user_id)











