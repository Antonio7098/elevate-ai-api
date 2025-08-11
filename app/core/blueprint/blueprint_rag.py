"""
Blueprint RAG (Retrieval-Augmented Generation) module.

This module provides RAG functionality for blueprints.
"""

from typing import List, Dict, Any, Optional, Tuple
from app.models.blueprint import Blueprint
from app.core.blueprint.blueprint_indexer import BlueprintIndexer
import json


class BlueprintRAG:
    """Retrieval-Augmented Generation system for blueprints."""
    
    def __init__(self, indexer: BlueprintIndexer):
        self.indexer = indexer
        self.context_window_size = 2000  # Maximum context length
    
    async def retrieve_relevant_context(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant blueprint context for a given query."""
        # Search for relevant blueprints
        relevant_blueprint_ids = await self.indexer.search(query, limit=limit)
        
        contexts = []
        for blueprint_id in relevant_blueprint_ids:
            context = await self._extract_context_from_blueprint(blueprint_id, query)
            if context:
                contexts.append(context)
        
        # Sort by relevance score
        contexts.sort(key=lambda x: x['relevance_score'], reverse=True)
        return contexts[:limit]
    
    async def _extract_context_from_blueprint(self, blueprint_id: str, query: str) -> Optional[Dict[str, Any]]:
        """Extract relevant context from a specific blueprint."""
        # Get blueprint info from indexer
        blueprint_info = self.indexer.index.get(blueprint_id)
        if not blueprint_info:
            return None
        
        # Get content from indexer
        content = self.indexer.content_index.get(blueprint_id, [])
        
        # Calculate relevance score
        relevance_score = self._calculate_relevance_score(query, blueprint_info, content)
        
        # Extract most relevant content snippets
        relevant_snippets = self._extract_relevant_snippets(query, content)
        
        return {
            'blueprint_id': blueprint_id,
            'title': blueprint_info.get('title', ''),
            'description': blueprint_info.get('description', ''),
            'type': blueprint_info.get('type', ''),
            'relevance_score': relevance_score,
            'relevant_snippets': relevant_snippets,
            'metadata': {
                'author_id': blueprint_info.get('author_id'),
                'tags': blueprint_info.get('tags', []),
                'created_at': blueprint_info.get('created_at'),
                'version': blueprint_info.get('version')
            }
        }
    
    def _calculate_relevance_score(self, query: str, blueprint_info: Dict[str, Any], content: List[str]) -> float:
        """Calculate relevance score for a blueprint based on query."""
        query_lower = query.lower()
        score = 0.0
        
        # Title match (highest weight)
        title = blueprint_info.get('title', '').lower()
        if query_lower in title:
            score += 10.0
            # Exact match bonus
            if query_lower == title:
                score += 5.0
        
        # Description match (medium weight)
        description = blueprint_info.get('description', '').lower()
        if description and query_lower in description:
            score += 5.0
        
        # Content match (lower weight)
        for content_text in content:
            if query_lower in content_text.lower():
                score += 1.0
                # Multiple matches in content
                score += content_text.lower().count(query_lower) * 0.1
        
        # Tag match (medium weight)
        tags = blueprint_info.get('tags', [])
        for tag in tags:
            if query_lower in tag.lower():
                score += 3.0
        
        # Type match (low weight)
        blueprint_type = blueprint_info.get('type', '').lower()
        if query_lower in blueprint_type:
            score += 2.0
        
        return score
    
    def _extract_relevant_snippets(self, query: str, content: List[str]) -> List[str]:
        """Extract the most relevant content snippets for a query."""
        query_lower = query.lower()
        snippets = []
        
        for text in content:
            if query_lower in text.lower():
                # Find the position of the query in the text
                start_pos = text.lower().find(query_lower)
                
                # Extract context around the query
                context_start = max(0, start_pos - 100)
                context_end = min(len(text), start_pos + len(query_lower) + 100)
                
                snippet = text[context_start:context_end]
                
                # Add ellipsis if we're not at the beginning/end
                if context_start > 0:
                    snippet = "..." + snippet
                if context_end < len(text):
                    snippet = snippet + "..."
                
                snippets.append(snippet)
        
        # Limit snippets to avoid overwhelming context
        return snippets[:3]
    
    async def generate_response_with_context(self, query: str, user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate a response using retrieved blueprint context."""
        # Retrieve relevant context
        contexts = await self.retrieve_relevant_context(query, limit=5)
        
        if not contexts:
            return {
                'response': f"I couldn't find any relevant blueprints for your query: '{query}'",
                'contexts': [],
                'suggestions': self._generate_search_suggestions(query)
            }
        
        # Build comprehensive response
        response = self._build_response_from_contexts(query, contexts)
        
        # Add user context if available
        if user_context:
            response = self._personalize_response(response, user_context)
        
        return {
            'response': response,
            'contexts': contexts,
            'suggestions': self._generate_follow_up_suggestions(contexts),
            'confidence_score': self._calculate_confidence_score(contexts)
        }
    
    def _build_response_from_contexts(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """Build a comprehensive response from retrieved contexts."""
        if not contexts:
            return f"I couldn't find any relevant information for your query: '{query}'"
        
        # Start with the most relevant context
        primary_context = contexts[0]
        
        response_parts = []
        
        # Main response based on primary context
        if primary_context['relevance_score'] >= 8.0:
            response_parts.append(f"Based on the blueprint '{primary_context['title']}', here's what I found:")
        else:
            response_parts.append(f"I found some relevant information in the blueprint '{primary_context['title']}':")
        
        # Add relevant snippets
        if primary_context['relevant_snippets']:
            response_parts.append("\nRelevant content:")
            for i, snippet in enumerate(primary_context['relevant_snippets'][:2], 1):
                response_parts.append(f"{i}. {snippet}")
        
        # Add additional contexts if they're also relevant
        additional_contexts = [ctx for ctx in contexts[1:] if ctx['relevance_score'] >= 5.0]
        if additional_contexts:
            response_parts.append(f"\nI also found related information in {len(additional_contexts)} other blueprints:")
            for ctx in additional_contexts[:2]:
                response_parts.append(f"- {ctx['title']} ({ctx['type']})")
        
        return "\n".join(response_parts)
    
    def _personalize_response(self, response: str, user_context: Dict[str, Any]) -> str:
        """Personalize the response based on user context."""
        # This is a simplified personalization
        # In a real system, you might use user preferences, learning history, etc.
        
        user_level = user_context.get('learning_level', 'beginner')
        if user_level == 'beginner':
            response += "\n\nThis information is suitable for beginners. Let me know if you need any clarification!"
        elif user_level == 'advanced':
            response += "\n\nThis covers advanced concepts. Feel free to ask for deeper insights!"
        
        return response
    
    def _generate_search_suggestions(self, query: str) -> List[str]:
        """Generate search suggestions when no relevant content is found."""
        suggestions = [
            f"Try searching for '{query}' with different keywords",
            "Check if the spelling is correct",
            "Try using broader or more specific terms",
            "Look for related concepts or synonyms"
        ]
        return suggestions
    
    def _generate_follow_up_suggestions(self, contexts: List[Dict[str, Any]]) -> List[str]:
        """Generate follow-up question suggestions based on retrieved contexts."""
        suggestions = []
        
        if not contexts:
            return suggestions
        
        primary_context = contexts[0]
        
        # Suggest exploring the blueprint further
        suggestions.append(f"Would you like me to explain more about '{primary_context['title']}'?")
        
        # Suggest related topics based on tags
        tags = primary_context['metadata'].get('tags', [])
        if tags:
            suggestions.append(f"Would you like to explore other blueprints with tags like {', '.join(tags[:3])}?")
        
        # Suggest different types of blueprints
        blueprint_type = primary_context['type']
        if blueprint_type:
            suggestions.append(f"Would you like to see more {blueprint_type} blueprints?")
        
        return suggestions
    
    def _calculate_confidence_score(self, contexts: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for the response."""
        if not contexts:
            return 0.0
        
        # Base confidence on relevance scores
        primary_score = contexts[0]['relevance_score']
        
        # Boost confidence if multiple relevant contexts exist
        context_boost = min(len(contexts) * 0.1, 0.3)
        
        # Normalize to 0-1 range
        confidence = min((primary_score / 15.0) + context_boost, 1.0)
        
        return round(confidence, 2)
    
    async def get_rag_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system."""
        index_stats = await self.indexer.get_index_stats()
        
        return {
            'rag_system': 'BlueprintRAG',
            'index_stats': index_stats,
            'context_window_size': self.context_window_size,
            'max_contexts_per_query': 5
        }
