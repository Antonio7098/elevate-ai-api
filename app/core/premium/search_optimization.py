"""
Search optimization service for premium users.
Implements multiple optimization strategies for search results.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from .core_api_client import CoreAPIClient

@dataclass
class Result:
    """Search result with optimization metadata"""
    content: str
    score: float
    source: str
    metadata: Dict[str, Any]
    optimization_scores: Dict[str, float] = None
    
    def __post_init__(self):
        if self.optimization_scores is None:
            self.optimization_scores = {}

@dataclass
class OptimizedResults:
    """Optimized search results"""
    results: List[Result]
    optimization_metrics: Dict[str, float]
    user_preferences: Dict[str, Any]
    timestamp: datetime

@dataclass
class PersonalizedResults:
    """Personalized search results"""
    results: List[Result]
    personalization_factors: Dict[str, float]
    user_profile: Dict[str, Any]
    timestamp: datetime

class DiversityOptimizer:
    """Optimizer for result diversity"""
    
    def optimize(self, results: List[Result], query: str) -> List[Result]:
        """Optimize for diversity in results"""
        try:
            # Simple diversity optimization - ensure different sources and content types
            optimized = []
            seen_sources = set()
            seen_content_types = set()
            
            for result in results:
                source = result.metadata.get("source", "unknown")
                content_type = result.metadata.get("content_type", "text")
                
                # Prefer diverse sources and content types
                if source not in seen_sources or content_type not in seen_content_types:
                    result.optimization_scores["diversity"] = 0.9
                    optimized.append(result)
                    seen_sources.add(source)
                    seen_content_types.add(content_type)
                else:
                    result.optimization_scores["diversity"] = 0.3
                    optimized.append(result)
            
            return optimized
            
        except Exception as e:
            print(f"Error in diversity optimization: {e}")
            return results

class RelevanceOptimizer:
    """Optimizer for result relevance"""
    
    def optimize(self, results: List[Result], query: str, user_context: Dict[str, Any]) -> List[Result]:
        """Optimize for relevance based on user context"""
        try:
            for result in results:
                # Calculate relevance score based on query similarity
                query_terms = set(query.lower().split())
                content_terms = set(result.content.lower().split())
                
                # Simple Jaccard similarity
                intersection = len(query_terms.intersection(content_terms))
                union = len(query_terms.union(content_terms))
                relevance_score = intersection / union if union > 0 else 0.0
                
                # Boost score based on user's learning level
                user_level = user_context.get("masteryLevel", "BEGINNER")
                if user_level == "EXPERT" and "advanced" in result.content.lower():
                    relevance_score *= 1.2
                elif user_level == "BEGINNER" and "basic" in result.content.lower():
                    relevance_score *= 1.2
                
                result.optimization_scores["relevance"] = min(relevance_score, 1.0)
            
            # Sort by relevance score
            results.sort(key=lambda x: x.optimization_scores.get("relevance", 0), reverse=True)
            return results
            
        except Exception as e:
            print(f"Error in relevance optimization: {e}")
            return results

class CoverageOptimizer:
    """Optimizer for comprehensive coverage"""
    
    def optimize(self, results: List[Result], query: str) -> List[Result]:
        """Optimize for comprehensive coverage of the topic"""
        try:
            # Identify key concepts in the query
            key_concepts = self._extract_key_concepts(query)
            
            for result in results:
                # Calculate coverage score based on concept coverage
                covered_concepts = 0
                for concept in key_concepts:
                    if concept.lower() in result.content.lower():
                        covered_concepts += 1
                
                coverage_score = covered_concepts / len(key_concepts) if key_concepts else 0.0
                result.optimization_scores["coverage"] = coverage_score
            
            # Sort by coverage score
            results.sort(key=lambda x: x.optimization_scores.get("coverage", 0), reverse=True)
            return results
            
        except Exception as e:
            print(f"Error in coverage optimization: {e}")
            return results
    
    def _extract_key_concepts(self, query: str) -> List[str]:
        """Extract key concepts from query"""
        # Simple concept extraction - in production, use NLP
        concepts = []
        words = query.lower().split()
        
        # Filter out common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        concepts = [word for word in words if word not in stop_words and len(word) > 2]
        
        return concepts

class UserPreferenceOptimizer:
    """Optimizer based on user preferences"""
    
    def __init__(self):
        self.core_api_client = CoreAPIClient()
    
    async def optimize(self, results: List[Result], user_id: str) -> List[Result]:
        """Optimize based on user preferences from Core API"""
        try:
            # Get user preferences from Core API
            user_memory = await self.core_api_client.get_user_memory(user_id)
            analytics = await self.core_api_client.get_user_learning_analytics(user_id)
            
            for result in results:
                preference_score = 0.0
                
                # Consider learning style
                learning_style = user_memory.get("learningStyle", "VISUAL")
                if learning_style == "VISUAL" and "diagram" in result.content.lower():
                    preference_score += 0.3
                elif learning_style == "AUDITORY" and "audio" in result.content.lower():
                    preference_score += 0.3
                elif learning_style == "KINESTHETIC" and "example" in result.content.lower():
                    preference_score += 0.3
                
                # Consider cognitive approach
                cognitive_approach = user_memory.get("cognitiveApproach", "BALANCED")
                if cognitive_approach == "TOP_DOWN" and "overview" in result.content.lower():
                    preference_score += 0.2
                elif cognitive_approach == "BOTTOM_UP" and "detail" in result.content.lower():
                    preference_score += 0.2
                
                # Consider learning efficiency
                efficiency = analytics.get("learningEfficiency", 0.5)
                if efficiency > 0.8 and "advanced" in result.content.lower():
                    preference_score += 0.2
                elif efficiency < 0.3 and "basic" in result.content.lower():
                    preference_score += 0.2
                
                result.optimization_scores["user_preference"] = min(preference_score, 1.0)
            
            # Sort by preference score
            results.sort(key=lambda x: x.optimization_scores.get("user_preference", 0), reverse=True)
            return results
            
        except Exception as e:
            print(f"Error in user preference optimization: {e}")
            return results

class EnsembleOptimizer:
    """Ensemble optimizer combining multiple strategies"""
    
    def __init__(self):
        self.optimizers = {
            'diversity': DiversityOptimizer(),
            'relevance': RelevanceOptimizer(),
            'coverage': CoverageOptimizer(),
            'user_preference': UserPreferenceOptimizer()
        }
        self.weights = {
            'diversity': 0.2,
            'relevance': 0.4,
            'coverage': 0.2,
            'user_preference': 0.2
        }
    
    async def optimize(self, results: List[Result], query: str, user_id: str, user_context: Dict[str, Any]) -> OptimizedResults:
        """Apply ensemble optimization"""
        try:
            optimized_results = results.copy()
            
            # Apply each optimizer
            optimized_results = self.optimizers['diversity'].optimize(optimized_results, query)
            optimized_results = self.optimizers['relevance'].optimize(optimized_results, query, user_context)
            optimized_results = self.optimizers['coverage'].optimize(optimized_results, query)
            optimized_results = await self.optimizers['user_preference'].optimize(optimized_results, user_id)
            
            # Calculate ensemble scores
            for result in optimized_results:
                ensemble_score = 0.0
                for optimizer_name, weight in self.weights.items():
                    score = result.optimization_scores.get(optimizer_name, 0.0)
                    ensemble_score += score * weight
                
                result.optimization_scores["ensemble"] = ensemble_score
            
            # Sort by ensemble score
            optimized_results.sort(key=lambda x: x.optimization_scores.get("ensemble", 0), reverse=True)
            
            # Calculate optimization metrics
            optimization_metrics = {
                "avg_diversity": sum(r.optimization_scores.get("diversity", 0) for r in optimized_results) / len(optimized_results),
                "avg_relevance": sum(r.optimization_scores.get("relevance", 0) for r in optimized_results) / len(optimized_results),
                "avg_coverage": sum(r.optimization_scores.get("coverage", 0) for r in optimized_results) / len(optimized_results),
                "avg_preference": sum(r.optimization_scores.get("user_preference", 0) for r in optimized_results) / len(optimized_results),
                "avg_ensemble": sum(r.optimization_scores.get("ensemble", 0) for r in optimized_results) / len(optimized_results)
            }
            
            return OptimizedResults(
                results=optimized_results,
                optimization_metrics=optimization_metrics,
                user_preferences=user_context,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            print(f"Error in ensemble optimization: {e}")
            return OptimizedResults(
                results=results,
                optimization_metrics={},
                user_preferences=user_context,
                timestamp=datetime.utcnow()
            )

class SearchOptimizer:
    """Main search optimizer for premium users"""
    
    def __init__(self):
        self.optimizers = {
            'diversity': DiversityOptimizer(),
            'relevance': RelevanceOptimizer(),
            'coverage': CoverageOptimizer(),
            'user_preference': UserPreferenceOptimizer()
        }
        self.ensemble = EnsembleOptimizer()
        self.core_api_client = CoreAPIClient()
    
    async def optimize_search_results(self, results: List[Result], query: str, context: Dict[str, Any]) -> OptimizedResults:
        """Apply multiple optimization strategies"""
        try:
            # Convert to Result objects if needed
            if not isinstance(results[0], Result):
                results = [Result(
                    content=r.get("content", ""),
                    score=r.get("score", 0.0),
                    source=r.get("source", ""),
                    metadata=r.get("metadata", {})
                ) for r in results]
            
            # Apply ensemble optimization
            return await self.ensemble.optimize(results, query, context.get("user_id", ""), context)
            
        except Exception as e:
            print(f"Error optimizing search results: {e}")
            return OptimizedResults(
                results=results,
                optimization_metrics={},
                user_preferences=context,
                timestamp=datetime.utcnow()
            )
    
    async def personalize_search(self, results: List[Result], user_profile: Dict[str, Any]) -> PersonalizedResults:
        """Personalize search results based on user preferences"""
        try:
            user_id = user_profile.get("user_id", "")
            
            # Apply user preference optimization
            personalized_results = await self.optimizers['user_preference'].optimize(results, user_id)
            
            # Calculate personalization factors
            personalization_factors = {
                "learning_style_alignment": sum(r.optimization_scores.get("user_preference", 0) for r in personalized_results) / len(personalized_results),
                "cognitive_approach_match": 0.8,  # Mock value
                "efficiency_optimization": 0.7   # Mock value
            }
            
            return PersonalizedResults(
                results=personalized_results,
                personalization_factors=personalization_factors,
                user_profile=user_profile,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            print(f"Error personalizing search: {e}")
            return PersonalizedResults(
                results=results,
                personalization_factors={},
                user_profile=user_profile,
                timestamp=datetime.utcnow()
            )











