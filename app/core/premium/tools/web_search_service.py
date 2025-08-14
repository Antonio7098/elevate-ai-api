"""
Real Web Search Service for Premium Tools
Integrates with Tavily API for live web search and real-time information.
"""

import aiohttp
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import json
import os
from urllib.parse import quote_plus

@dataclass
class SearchResult:
    """Individual search result"""
    title: str
    url: str
    content: str
    score: float
    source: str
    published_date: Optional[str] = None
    language: Optional[str] = None

@dataclass
class SearchResponse:
    """Complete search response"""
    query: str
    results: List[SearchResult]
    total_results: int
    search_time: float
    search_depth: str
    timestamp: datetime
    answer: Optional[str] = None
    related_questions: List[str] = None

@dataclass
class RealtimeData:
    """Real-time data result"""
    data_type: str
    value: Any
    timestamp: datetime
    source: str
    confidence: float
    unit: Optional[str] = None

class TavilySearchService:
    """Web search service using Tavily API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("Tavily API key is required")
        
        self.base_url = "https://api.tavily.com"
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Rate limiting
        self.requests_per_minute = 60
        self.request_times: List[datetime] = []
        
        # Cache for search results
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 300  # 5 minutes
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def search(self, query: str, search_type: str = "basic", 
                    max_results: int = 10, include_answer: bool = True) -> SearchResponse:
        """
        Perform web search using Tavily API
        
        Args:
            query: Search query
            search_type: Search depth (basic, advanced)
            max_results: Maximum number of results
            include_answer: Whether to include AI-generated answer
            
        Returns:
            SearchResponse with search results
        """
        start_time = asyncio.get_event_loop().time()
        
        # Check rate limiting
        await self._check_rate_limit()
        
        # Check cache
        cache_key = f"{query}:{search_type}:{max_results}"
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        try:
            # Prepare request parameters
            params = {
                "api_key": self.api_key,
                "query": query,
                "search_depth": search_type,
                "include_answer": include_answer,
                "include_raw_content": False,
                "max_results": max_results
            }
            
            # Make API request
            async with self.session.get(f"{self.base_url}/search", params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Tavily API error: {response.status} - {error_text}")
                
                data = await response.json()
                
                # Parse results
                results = []
                for item in data.get("results", []):
                    result = SearchResult(
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        content=item.get("content", ""),
                        score=item.get("score", 0.0),
                        source=item.get("source", ""),
                        published_date=item.get("published_date"),
                        language=item.get("language")
                    )
                    results.append(result)
                
                # Create response
                search_response = SearchResponse(
                    query=query,
                    results=results,
                    total_results=data.get("total_results", len(results)),
                    search_time=asyncio.get_event_loop().time() - start_time,
                    search_depth=search_type,
                    answer=data.get("answer"),
                    related_questions=data.get("related_questions", []),
                    timestamp=datetime.utcnow()
                )
                
                # Cache result
                self._cache_result(cache_key, search_response)
                
                # Update rate limiting
                self._update_rate_limit()
                
                return search_response
                
        except Exception as e:
            # Return fallback response
            return SearchResponse(
                query=query,
                results=[],
                total_results=0,
                search_time=asyncio.get_event_loop().time() - start_time,
                search_depth=search_type,
                answer=None,
                related_questions=[],
                timestamp=datetime.utcnow()
            )
    
    async def get_realtime_data(self, data_type: str, params: Dict[str, Any]) -> RealtimeData:
        """
        Fetch real-time data (weather, stocks, news, etc.)
        
        Args:
            data_type: Type of data to fetch
            params: Parameters for the data request
            
        Returns:
            RealtimeData with the requested information
        """
        try:
            if data_type == "weather":
                return await self._get_weather_data(params)
            elif data_type == "stocks":
                return await self._get_stock_data(params)
            elif data_type == "news":
                return await self._get_news_data(params)
            elif data_type == "sports":
                return await self._get_sports_data(params)
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
                
        except Exception as e:
            return RealtimeData(
                data_type=data_type,
                value=None,
                timestamp=datetime.utcnow(),
                source="error",
                confidence=0.0
            )
    
    async def _get_weather_data(self, params: Dict[str, Any]) -> RealtimeData:
        """Get weather data for a location"""
        location = params.get("location", "New York")
        query = f"current weather in {location}"
        
        search_response = await self.search(query, search_type="basic", max_results=3)
        
        if search_response.results:
            # Extract weather information from search results
            weather_info = search_response.results[0].content
            return RealtimeData(
                data_type="weather",
                value=weather_info,
                timestamp=datetime.utcnow(),
                source="Tavily Search",
                confidence=0.8
            )
        
        return RealtimeData(
            data_type="weather",
            value="Weather information not available",
            timestamp=datetime.utcnow(),
            source="Tavily Search",
            confidence=0.0
        )
    
    async def _get_stock_data(self, params: Dict[str, Any]) -> RealtimeData:
        """Get stock market data"""
        symbol = params.get("symbol", "AAPL")
        query = f"current stock price {symbol}"
        
        search_response = await self.search(query, search_type="basic", max_results=3)
        
        if search_response.results:
            stock_info = search_response.results[0].content
            return RealtimeData(
                data_type="stocks",
                value=stock_info,
                timestamp=datetime.utcnow(),
                source="Tavily Search",
                confidence=0.8
            )
        
        return RealtimeData(
            data_type="stocks",
            value="Stock information not available",
            timestamp=datetime.utcnow(),
            source="Tavily Search",
            confidence=0.0
        )
    
    async def _get_news_data(self, params: Dict[str, Any]) -> RealtimeData:
        """Get latest news for a topic"""
        topic = params.get("topic", "technology")
        query = f"latest news {topic}"
        
        search_response = await self.search(query, search_type="basic", max_results=5)
        
        if search_response.results:
            news_summary = "\n".join([f"• {result.title}" for result in search_response.results[:3]])
            return RealtimeData(
                data_type="news",
                value=news_summary,
                timestamp=datetime.utcnow(),
                source="Tavily Search",
                confidence=0.9
            )
        
        return RealtimeData(
            data_type="news",
            value="News information not available",
            timestamp=datetime.utcnow(),
            source="Tavily Search",
            confidence=0.0
        )
    
    async def _get_sports_data(self, params: Dict[str, Any]) -> RealtimeData:
        """Get sports scores and updates"""
        sport = params.get("sport", "football")
        team = params.get("team", "")
        
        if team:
            query = f"latest {sport} scores {team}"
        else:
            query = f"latest {sport} scores"
        
        search_response = await self.search(query, search_type="basic", max_results=3)
        
        if search_response.results:
            sports_info = search_response.results[0].content
            return RealtimeData(
                data_type="sports",
                value=sports_info,
                timestamp=datetime.utcnow(),
                source="Tavily Search",
                confidence=0.8
            )
        
        return RealtimeData(
            data_type="sports",
            value="Sports information not available",
            timestamp=datetime.utcnow(),
            source="Tavily Search",
            confidence=0.0
        )
    
    async def _check_rate_limit(self):
        """Check if we're within rate limits"""
        now = datetime.utcnow()
        
        # Remove old requests from tracking
        self.request_times = [t for t in self.request_times 
                            if (now - t).total_seconds() < 60]
        
        if len(self.request_times) >= self.requests_per_minute:
            # Wait until we can make another request
            wait_time = 60 - (now - self.request_times[0]).total_seconds()
            if wait_time > 0:
                await asyncio.sleep(wait_time)
    
    def _update_rate_limit(self):
        """Update rate limiting tracking"""
        self.request_times.append(datetime.utcnow())
    
    def _get_cached_result(self, cache_key: str) -> Optional[SearchResponse]:
        """Get cached search result"""
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            cache_age = (datetime.utcnow() - cached["timestamp"]).total_seconds()
            
            if cache_age < self.cache_ttl:
                return cached["data"]
            else:
                # Remove expired cache entry
                del self.cache[cache_key]
        
        return None
    
    def _cache_result(self, cache_key: str, result: SearchResponse):
        """Cache search result"""
        self.cache[cache_key] = {
            "data": result,
            "timestamp": datetime.utcnow()
        }
        
        # Limit cache size
        if len(self.cache) > 100:
            # Remove oldest entries
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest_key]
    
    async def get_search_suggestions(self, query: str) -> List[str]:
        """Get search suggestions for a query"""
        try:
            # Use Tavily's search to get related queries
            search_response = await self.search(query, search_type="basic", max_results=5)
            
            suggestions = []
            for result in search_response.results:
                # Extract potential suggestions from titles
                title_words = result.title.split()
                if len(title_words) > 2:
                    suggestion = " ".join(title_words[:3])
                    if suggestion not in suggestions:
                        suggestions.append(suggestion)
            
            return suggestions[:5]
            
        except Exception:
            return []
    
    async def get_search_analytics(self) -> Dict[str, Any]:
        """Get search service analytics"""
        return {
            "total_searches": len(self.request_times),
            "cache_size": len(self.cache),
            "cache_hit_rate": 0.0,  # Would need to track hits/misses
            "average_response_time": 0.0,  # Would need to track response times
            "rate_limit_status": {
                "requests_this_minute": len(self.request_times),
                "limit": self.requests_per_minute
            }
        }
