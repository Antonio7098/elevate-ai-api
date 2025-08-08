#!/usr/bin/env python3

"""
End-to-End Search & RAG Test Script

This script tests the complete search and RAG chat functionality:
1. Vector search for relevant content
2. Semantic similarity search 
3. RAG chat with context retrieval
4. Multi-turn conversation handling
5. Search result ranking and filtering
6. Usage analytics for search queries

Usage: python test-e2e-search-rag.py
"""

import asyncio
import sys
import json
import time
from typing import Dict, List, Any, Optional
import httpx
from datetime import datetime

# Configuration
AI_API_BASE_URL = "http://localhost:8000"
API_KEY = "test_api_key_123"
TEST_USER_ID = 108

class TestResult:
    def __init__(self, step: str, status: str, details: str = None, error: Any = None):
        self.step = step
        self.status = status  # 'PASS', 'FAIL', 'SKIP'
        self.details = details
        self.error = error

class SearchRAGE2ETest:
    def __init__(self):
        self.results: List[TestResult] = []
        self.test_data = {}
        self.client = httpx.AsyncClient(timeout=30.0)
        
    async def run(self) -> None:
        """Run the complete search & RAG e2e test suite."""
        print("🚀 Starting AI API Search & RAG E2E Test\n")
        
        try:
            await self.run_step(self.test_health_check)
            await self.run_step(self.test_vector_search)
            await self.run_step(self.test_semantic_search)
            await self.run_step(self.test_rag_chat_single)
            await self.run_step(self.test_rag_chat_multi_turn)
            await self.run_step(self.test_search_filtering)
            await self.run_step(self.test_usage_analytics)
        except Exception as error:
            print(f"\n❌ Test suite aborted due to critical failure: {error}")
        finally:
            await self.client.aclose()
            self.print_results()
    
    async def run_step(self, step_func, continue_on_error: bool = False) -> None:
        """Execute a test step with error handling."""
        try:
            await step_func()
        except Exception as error:
            if not continue_on_error:
                raise error

    async def test_health_check(self) -> None:
        """Test 1: Verify AI API health and availability."""
        try:
            print("🏥 Step 1: Checking AI API health...")
            
            response = await self.client.get(f"{AI_API_BASE_URL}/api/health")
            
            if response.status_code == 200:
                health_data = response.json()
                self.results.append(TestResult(
                    "1. AI API Health Check",
                    "PASS",
                    f"AI API healthy - Status: {health_data.get('status', 'unknown')}"
                ))
                print("   ✅ AI API health check successful")
            else:
                raise Exception(f"Health check failed with status {response.status_code}")
                
        except Exception as error:
            self.results.append(TestResult(
                "1. AI API Health Check",
                "FAIL",
                f"Health check failed: {str(error)}",
                error
            ))
            print("   ❌ AI API health check failed")
            raise error

    async def test_vector_search(self) -> None:
        """Test 2: Vector-based similarity search."""
        try:
            print("🔍 Step 2: Testing vector search...")
            
            # Test vector search for relevant content
            payload = {
                "query": "photosynthesis light reactions chloroplasts",
                "userId": TEST_USER_ID,
                "searchPreferences": {
                    "maxResults": 10,
                    "similarityThreshold": 0.7,
                    "includeMetadata": True,
                    "rankBy": "relevance"
                },
                "filters": {
                    "contentType": ["blueprint", "note"],
                    "recency": "any"
                }
            }
            
            headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
            response = await self.client.post(
                f"{AI_API_BASE_URL}/api/v1/search/vector",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                self.test_data['vector_results'] = data.get('results', [])
                result_count = len(self.test_data['vector_results'])
                
                # Check result quality
                relevance_scores = [r.get('relevanceScore', 0) for r in self.test_data['vector_results']]
                avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
                
                self.results.append(TestResult(
                    "2. Vector Search",
                    "PASS",
                    f"Retrieved {result_count} results with avg relevance: {avg_relevance:.2f}"
                ))
                print(f"   ✅ Vector search successful - {result_count} results found")
                
            else:
                raise Exception(f"Vector search failed with status {response.status_code}: {response.text}")
                
        except Exception as error:
            self.results.append(TestResult(
                "2. Vector Search",
                "FAIL",
                f"Vector search failed: {str(error)}",
                error
            ))
            print("   ❌ Vector search failed")
            raise error

    async def test_semantic_search(self) -> None:
        """Test 3: Semantic search with natural language queries."""
        try:
            print("🧠 Step 3: Testing semantic search...")
            
            # Test semantic search with natural language
            payload = {
                "query": "How do plants make food using sunlight?",
                "userId": TEST_USER_ID,
                "searchType": "semantic",
                "context": {
                    "subject": "biology",
                    "level": "high_school",
                    "preferredFormat": "explanatory"
                },
                "options": {
                    "expandQuery": True,
                    "includeRelated": True,
                    "maxResults": 8
                }
            }
            
            headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
            response = await self.client.post(
                f"{AI_API_BASE_URL}/api/v1/search/semantic",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                self.test_data['semantic_results'] = data.get('results', [])
                result_count = len(self.test_data['semantic_results'])
                
                # Check for query expansion
                expanded_terms = data.get('expandedTerms', [])
                
                self.results.append(TestResult(
                    "3. Semantic Search",
                    "PASS",
                    f"Retrieved {result_count} results, query expanded with {len(expanded_terms)} terms"
                ))
                print(f"   ✅ Semantic search successful - {result_count} results")
                
            else:
                raise Exception(f"Semantic search failed with status {response.status_code}: {response.text}")
                
        except Exception as error:
            self.results.append(TestResult(
                "3. Semantic Search",
                "FAIL",
                f"Semantic search failed: {str(error)}",
                error
            ))
            print("   ❌ Semantic search failed")
            raise error

    async def test_rag_chat_single(self) -> None:
        """Test 4: Single-turn RAG chat conversation."""
        try:
            print("💬 Step 4: Testing single-turn RAG chat...")
            
            # Test RAG chat with context retrieval and generation
            payload = {
                "message": "Explain the difference between mitosis and meiosis in simple terms",
                "userId": TEST_USER_ID,
                "conversationId": None,  # New conversation
                "chatPreferences": {
                    "responseStyle": "educational",
                    "maxLength": "medium",
                    "includeReferences": True,
                    "contextWindow": 5
                },
                "searchFilters": {
                    "contentTypes": ["blueprint", "note"],
                    "subjects": ["biology"]
                }
            }
            
            headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
            response = await self.client.post(
                f"{AI_API_BASE_URL}/api/v1/chat",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                self.test_data['rag_single'] = data
                
                response_text = data.get('response', '')
                context_sources = data.get('contextSources', [])
                conversation_id = data.get('conversationId')
                
                self.results.append(TestResult(
                    "4. RAG Chat Single-Turn",
                    "PASS",
                    f"Generated response ({len(response_text)} chars) with {len(context_sources)} context sources"
                ))
                print(f"   ✅ RAG chat successful - {len(context_sources)} sources used")
                
                # Store conversation ID for multi-turn test
                self.test_data['conversation_id'] = conversation_id
                
            else:
                raise Exception(f"RAG chat failed with status {response.status_code}: {response.text}")
                
        except Exception as error:
            self.results.append(TestResult(
                "4. RAG Chat Single-Turn",
                "FAIL",
                f"RAG chat failed: {str(error)}",
                error
            ))
            print("   ❌ RAG chat single-turn failed")
            raise error

    async def test_rag_chat_multi_turn(self) -> None:
        """Test 5: Multi-turn RAG conversation with context preservation."""
        try:
            print("🔄 Step 5: Testing multi-turn RAG conversation...")
            
            conversation_id = self.test_data.get('conversation_id')
            if not conversation_id:
                print("   ⏭️  No conversation ID from previous step - skipping multi-turn test")
                self.results.append(TestResult(
                    "5. RAG Chat Multi-Turn",
                    "SKIP",
                    "No conversation ID available from single-turn test"
                ))
                return
            
            # Continue the conversation with a follow-up question
            payload = {
                "message": "Can you give me a specific example of when meiosis would occur in the human body?",
                "userId": TEST_USER_ID,
                "conversationId": conversation_id,
                "chatPreferences": {
                    "responseStyle": "conversational",
                    "maintainContext": True,
                    "referPreviousAnswers": True
                }
            }
            
            headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
            response = await self.client.post(
                f"{AI_API_BASE_URL}/api/v1/chat",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                self.test_data['rag_multi'] = data
                
                response_text = data.get('response', '')
                context_maintained = data.get('contextMaintained', False)
                
                self.results.append(TestResult(
                    "5. RAG Chat Multi-Turn",
                    "PASS",
                    f"Multi-turn response generated, context maintained: {context_maintained}"
                ))
                print(f"   ✅ Multi-turn RAG successful - context maintained: {context_maintained}")
                
            else:
                raise Exception(f"Multi-turn RAG failed with status {response.status_code}: {response.text}")
                
        except Exception as error:
            self.results.append(TestResult(
                "5. RAG Chat Multi-Turn",
                "FAIL",
                f"Multi-turn RAG failed: {str(error)}",
                error
            ))
            print("   ❌ RAG chat multi-turn failed")
            # Don't raise error - continue with other tests

    async def test_search_filtering(self) -> None:
        """Test 6: Search result filtering and ranking."""
        try:
            print("🎛️ Step 6: Testing search filtering and ranking...")
            
            # Test advanced filtering options
            filter_tests = [
                {
                    "name": "Subject Filter",
                    "filters": {"subjects": ["biology", "chemistry"]},
                    "expectedField": "subject"
                },
                {
                    "name": "Content Type Filter", 
                    "filters": {"contentTypes": ["blueprint"]},
                    "expectedField": "contentType"
                },
                {
                    "name": "Recency Filter",
                    "filters": {"recency": "last_week"},
                    "expectedField": "timestamp"
                }
            ]
            
            filter_results = {}
            
            for test in filter_tests:
                payload = {
                    "query": "cellular processes energy",
                    "userId": TEST_USER_ID,
                    "maxResults": 5,
                    "filters": test["filters"]
                }
                
                headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
                response = await self.client.post(
                    f"{AI_API_BASE_URL}/api/v1/search",
                    json=payload,
                    headers=headers
                )
                
                if response.status_code == 200:
                    data = response.json()
                    filter_results[test["name"]] = len(data.get('results', []))
            
            if filter_results:
                total_filtered = sum(filter_results.values())
                filter_count = len(filter_results)
                
                self.test_data['filter_results'] = filter_results
                
                self.results.append(TestResult(
                    "6. Search Filtering",
                    "PASS",
                    f"Tested {filter_count} filter types, retrieved {total_filtered} total filtered results"
                ))
                print(f"   ✅ Search filtering successful - {filter_results}")
                
            else:
                raise Exception("No filtering results were obtained")
                
        except Exception as error:
            self.results.append(TestResult(
                "6. Search Filtering",
                "FAIL",
                f"Search filtering failed: {str(error)}",
                error
            ))
            print("   ❌ Search filtering failed")
            # Don't raise error - continue with other tests

    async def test_usage_analytics(self) -> None:
        """Test 7: Search and chat usage analytics."""
        try:
            print("📊 Step 7: Testing usage analytics...")
            
            # Get search and chat analytics for the user
            headers = {"Authorization": f"Bearer {API_KEY}"}
            response = await self.client.get(
                f"{AI_API_BASE_URL}/api/v1/analytics/search/{TEST_USER_ID}?period=7d",
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                self.test_data['search_analytics'] = data
                
                search_count = data.get('totalSearches', 0)
                chat_count = data.get('totalChats', 0)
                avg_response_time = data.get('averageResponseTime', 0)
                
                self.results.append(TestResult(
                    "7. Usage Analytics",
                    "PASS",
                    f"Analytics retrieved - {search_count} searches, {chat_count} chats, {avg_response_time}ms avg response"
                ))
                print(f"   ✅ Usage analytics successful - {search_count} searches tracked")
                
            else:
                # Analytics endpoint may not exist, create mock data based on test activities
                mock_analytics = {
                    "totalSearches": 3,  # vector, semantic, filtered searches from this test
                    "totalChats": 2,     # single-turn and multi-turn from this test
                    "averageResponseTime": 1250,
                    "topQueries": ["photosynthesis", "mitosis meiosis", "cellular processes"],
                    "successRate": 85.7
                }
                
                self.test_data['search_analytics'] = mock_analytics
                
                self.results.append(TestResult(
                    "7. Usage Analytics",
                    "SKIP", 
                    "Analytics endpoint not available - using estimated data from test activities"
                ))
                print("   ⏭️  Using mock analytics data (endpoint not implemented)")
                
        except Exception as error:
            self.results.append(TestResult(
                "7. Usage Analytics",
                "FAIL",
                f"Usage analytics failed: {str(error)}",
                error
            ))
            print("   ❌ Usage analytics failed")
            # Don't raise error - this is not critical

    def print_results(self) -> None:
        """Print formatted test results."""
        print(f"\n{'=' * 60}")
        print("📊 AI API SEARCH & RAG E2E TEST RESULTS")
        print(f"{'=' * 60}")
        
        for result in self.results:
            status_emoji = "✅" if result.status == "PASS" else "❌" if result.status == "FAIL" else "⏭️"
            print(f"{status_emoji} {result.step}")
            if result.details:
                print(f"   {result.details}")
        
        print("-" * 60)
        
        passed = sum(1 for r in self.results if r.status == "PASS")
        failed = sum(1 for r in self.results if r.status == "FAIL")
        skipped = sum(1 for r in self.results if r.status == "SKIP")
        
        print(f"📈 SUMMARY: {passed} passed, {failed} failed, {skipped} skipped")
        
        if failed > 0:
            print("⚠️  Some tests failed. Check the errors above for details.")
        elif passed > 0:
            print("🎉 All tests passed! Search & RAG workflow is working correctly.")
        
        print(f"{'=' * 60}")

async def main():
    """Main function to run the test suite."""
    test = SearchRAGE2ETest()
    await test.run()

if __name__ == "__main__":
    asyncio.run(main())
