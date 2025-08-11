#!/usr/bin/env python3
"""
Test script to verify user isolation in the RAG search pipeline.

This test ensures that:
1. Users can only access their own blueprints
2. Cross-user blueprint access is prevented
3. userId filtering works correctly in all search strategies
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

from app.core.vector_store import create_vector_store
from app.core.embeddings import get_embedding_service, initialize_embedding_service
from app.core.rag_search import RAGSearchService, RAGSearchRequest

async def test_user_isolation():
    """Test that users can only access their own blueprints."""
    
    print("=== USER ISOLATION TEST ===")
    
    # Initialize services
    try:
        # Initialize embedding service first
        await initialize_embedding_service(
            service_type="google",
            api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Get environment variables for Pinecone
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
        
        if not pinecone_api_key:
            print("❌ PINECONE_API_KEY not found in environment")
            return False
            
        if not pinecone_env:
            print("❌ PINECONE_ENVIRONMENT not found in environment")
            return False
        
        # Create vector store with proper credentials
        vector_store = create_vector_store(
            store_type="pinecone",
            api_key=pinecone_api_key,
            environment=pinecone_env,
            index_name="blueprint-nodes"  # Use the same index as production
        )
        
        # Initialize the vector store client
        await vector_store.initialize()
        
        # Get embedding service (already initialized globally)
        embedding_service = await get_embedding_service()
        
        # Create RAG service
        rag_service = RAGSearchService(vector_store, embedding_service)
        
        print("✅ Services initialized successfully")
    except Exception as e:
        print(f"❌ Service initialization failed: {e}")
        return False
    
    # Test 1: Search with User 1
    print("\n--- Test 1: Search as User 1 ---")
    try:
        search_request_user1 = RAGSearchRequest(
            query="What is machine learning?",
            user_id="user_1",  # Search as user 1
            top_k=5,
            similarity_threshold=0.7
        )
        
        results_user1 = await rag_service.search(search_request_user1)
        print(f"✅ User 1 search completed: {len(results_user1.results)} results")
        
        # Check that all results have userId = user_1
        user1_filtered_correctly = True
        for result in results_user1.results:
            result_user_id = result.metadata.get('userId')
            if result_user_id != "user_1":
                print(f"❌ User 1 got result from userId: {result_user_id} (should be user_1)")
                user1_filtered_correctly = False
        
        if user1_filtered_correctly:
            print("✅ User 1 only received their own blueprints")
        
    except Exception as e:
        print(f"❌ User 1 search failed: {e}")
        return False
    
    # Test 2: Search with User 2
    print("\n--- Test 2: Search as User 2 ---")
    try:
        search_request_user2 = RAGSearchRequest(
            query="What is machine learning?",
            user_id="user_2",  # Search as user 2
            top_k=5,
            similarity_threshold=0.7
        )
        
        results_user2 = await rag_service.search(search_request_user2)
        print(f"✅ User 2 search completed: {len(results_user2.results)} results")
        
        # Check that all results have userId = user_2
        user2_filtered_correctly = True
        for result in results_user2.results:
            result_user_id = result.metadata.get('userId')
            if result_user_id != "user_2":
                print(f"❌ User 2 got result from userId: {result_user_id} (should be user_2)")
                user2_filtered_correctly = False
        
        if user2_filtered_correctly:
            print("✅ User 2 only received their own blueprints")
        
    except Exception as e:
        print(f"❌ User 2 search failed: {e}")
        return False
    
    # Test 3: Verify cross-user isolation
    print("\n--- Test 3: Cross-User Isolation Verification ---")
    
    # Collect user IDs from both searches
    user1_result_ids = {result.id for result in results_user1.results}
    user2_result_ids = {result.id for result in results_user2.results}
    
    # Check for overlap (there should be none)
    overlap = user1_result_ids.intersection(user2_result_ids)
    
    if not overlap:
        print("✅ No overlap between user results - isolation working correctly")
        isolation_success = True
    else:
        print(f"❌ Found {len(overlap)} overlapping results between users: {overlap}")
        isolation_success = False
    
    # Test 4: Test with non-existent user
    print("\n--- Test 4: Non-existent User ---")
    try:
        search_request_no_user = RAGSearchRequest(
            query="What is machine learning?",
            user_id="non_existent_user",  # User that doesn't exist
            top_k=5,
            similarity_threshold=0.7
        )
        
        results_no_user = await rag_service.search(search_request_no_user)
        print(f"✅ Non-existent user search completed: {len(results_no_user.results)} results")
        
        if len(results_no_user.results) == 0:
            print("✅ Non-existent user correctly got 0 results")
            no_user_success = True
        else:
            print(f"❌ Non-existent user got {len(results_no_user.results)} results (should be 0)")
            no_user_success = False
            
    except Exception as e:
        print(f"❌ Non-existent user search failed: {e}")
        no_user_success = False
    
    # Final results
    print("\n=== FINAL RESULTS ===")
    all_tests_passed = (
        user1_filtered_correctly and 
        user2_filtered_correctly and 
        isolation_success and 
        no_user_success
    )
    
    if all_tests_passed:
        print("🎉 ALL USER ISOLATION TESTS PASSED!")
        print("✅ Users can only access their own blueprints")
        print("✅ Cross-user data access is prevented")
        print("✅ Non-existent users get no results")
        return True
    else:
        print("❌ USER ISOLATION TESTS FAILED!")
        print("🚨 CRITICAL SECURITY VULNERABILITY - Users can access other users' data!")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_user_isolation())
    exit(0 if success else 1)
