"""
Tavily Search Service for enhanced context assembly.

This service provides real-time web search capabilities to enhance
the RAG system with current, relevant information.
"""

import os
import json
from typing import List, Dict, Any, Optional
from langchain_tavily import TavilySearch


class TavilySearchService:
    """Enhanced search service using Tavily for real-time web search."""
    
    def __init__(self, max_results: int = 10, topic: str = "general"):
        """Initialize Tavily search service."""
        # Check if API key is available
        if not os.getenv("TAVILY_API_KEY"):
            raise ValueError("TAVILY_API_KEY environment variable is required")
        
        # Initialize Tavily search tool
        self.tavily_search = TavilySearch(
            max_results=max_results,
            topic=topic,
        )
        
        self.max_results = max_results
        self.topic = topic
    
    async def search(
        self, 
        query_or_request, 
        top_k: int = 10, 
        threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Perform search using Tavily.
        
        Args:
            query_or_request: Search query string or RAGSearchRequest object
            top_k: Maximum number of results
            threshold: Relevance threshold
            filters: Additional search filters
            
        Returns:
            SearchResponse object with results attribute
        """
        # Extract query from RAGSearchRequest if needed
        if hasattr(query_or_request, 'query'):
            query = query_or_request.query
            # Also extract top_k and other parameters if available
            if hasattr(query_or_request, 'top_k'):
                top_k = query_or_request.top_k
        else:
            query = str(query_or_request)
        
        try:
            # Use Tavily search
            search_results = await self.tavily_search.ainvoke(query)
            
            # Parse and format results
            formatted_results = []
            
            if hasattr(search_results, 'content') and search_results.content:
                # Handle Tavily response format
                if isinstance(search_results.content, str):
                    # Try to parse JSON content
                    try:
                        parsed = json.loads(search_results.content)
                        if 'results' in parsed:
                            for result in parsed['results'][:top_k]:
                                formatted_results.append({
                                    'content': result.get('content', ''),
                                    'title': result.get('title', ''),
                                    'url': result.get('url', ''),
                                    'score': result.get('score', 0.0),
                                    'source': 'tavily_web_search'
                                })
                        else:
                            # Fallback: treat content as single result
                            formatted_results.append({
                                'content': parsed.get('answer', str(parsed)),
                                'title': 'Tavily Search Result',
                                'url': '',
                                'score': 1.0,
                                'source': 'tavily_web_search'
                            })
                    except json.JSONDecodeError:
                        # Fallback: treat as plain text
                        formatted_results.append({
                            'content': search_results.content,
                            'title': 'Tavily Search Result',
                            'url': '',
                            'score': 1.0,
                            'source': 'tavily_web_search'
                        })
                else:
                    # Handle other response formats
                    formatted_results.append({
                        'content': str(search_results.content),
                        'title': 'Tavily Search Result',
                        'url': '',
                        'score': 1.0,
                        'source': 'tavily_web_search'
                    })
            
            # Filter by threshold if needed
            if threshold > 0:
                formatted_results = [
                    result for result in formatted_results 
                    if result.get('score', 0.0) >= threshold
                ]
            
            # Create a response object that matches what the ContextAssembler expects
            class SearchResponse:
                def __init__(self, results):
                    self.results = results
                    self.search_strategy = "tavily_web_search"
            
            return SearchResponse(formatted_results[:top_k])
            
        except Exception as e:
            print(f"Tavily search failed: {e}")
            # Return empty response object
            class SearchResponse:
                def __init__(self):
                    self.results = []
                    self.search_strategy = "tavily_web_search"
            return SearchResponse()
    
    async def search_with_metadata(
        self, 
        query: str, 
        top_k: int = 10,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform search with metadata.
        
        Args:
            query: Search query
            top_k: Maximum number of results
            include_metadata: Whether to include metadata
            
        Returns:
            List of search results with metadata
        """
        results = await self.search(query, top_k)
        
        if include_metadata:
            for result in results:
                result['metadata'] = {
                    'search_timestamp': self._get_timestamp(),
                    'search_tool': 'tavily',
                    'topic': self.topic,
                    'max_results': self.max_results
                }
        
        return results
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.utcnow().isoformat()
    
    def is_available(self) -> bool:
        """Check if the search service is available."""
        return bool(os.getenv("TAVILY_API_KEY"))
