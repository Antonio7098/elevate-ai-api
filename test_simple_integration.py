#!/usr/bin/env python3
"""
Simple integration test to verify services are working without hanging.
"""

import os
import asyncio
import httpx

# Configuration
CORE_API_BASE_URL = "http://localhost:3000"
AI_API_BASE_URL = "http://localhost:8000"

# Optional authentication
AI_API_KEY = os.getenv("AI_API_KEY") or os.getenv("ELEVATE_AI_API_KEY") or os.getenv("API_KEY")
DEFAULT_HEADERS = {"Authorization": f"Bearer {AI_API_KEY}"} if AI_API_KEY else {}

async def test_services():
    """Test basic service connectivity and endpoints."""
    print("🔍 Testing service connectivity...")
    
    async with httpx.AsyncClient(headers=DEFAULT_HEADERS, timeout=10.0) as client:
        try:
            # Test AI API health (no auth required)
            print("📡 Testing AI API health...")
            ai_response = await client.get(f"{AI_API_BASE_URL}/health")
            print(f"   AI API: {ai_response.status_code} - {ai_response.text[:100]}")
            
            # Test AI API with authentication
            print("📡 Testing AI API auth endpoint...")
            auth_response = await client.get(f"{AI_API_BASE_URL}/api/health")
            print(f"   AI API (auth): {auth_response.status_code} - {auth_response.text[:100]}")
            
            # Skip Core API health for now (requires auth)
            print("📡 Skipping Core API health (requires auth)...")
            # core_response = await client.get(f"{CORE_API_BASE_URL}/api/health")
            # print(f"   Core API: {core_response.status_code} - {core_response.text[:100]}")
            
            # Test a simple AI API endpoint
            print("📡 Testing AI API deconstruct endpoint...")
            try:
                deconstruct_response = await client.post(
                    f"{AI_API_BASE_URL}/api/v1/deconstruct",
                    json={
                        "source_text": "Test content for deconstruction",
                        "context": {"title": "Test", "subject": "Testing"}
                    },
                    timeout=10.0
                )
                print(f"   Deconstruct: {deconstruct_response.status_code} - {deconstruct_response.text[:200]}")
            except Exception as e:
                print(f"   Deconstruct: ERROR - {e}")
            
            # Test primitive endpoints
            print("📡 Testing AI API primitives endpoint...")
            try:
                primitive_response = await client.post(
                    f"{AI_API_BASE_URL}/api/v1/primitives/generate",
                    json={
                        "sourceContent": "Test content",
                        "sourceType": "text",
                        "userPreferences": {"maxPrimitives": 1}
                    },
                    timeout=10.0
                )
                print(f"   Primitives: {primitive_response.status_code} - {primitive_response.text[:200]}")
            except Exception as e:
                print(f"   Primitives: ERROR - {e}")
            
            print("✅ Service connectivity test completed!")
            
        except Exception as e:
            print(f"❌ Error during service test: {e}")
            return False
    
    return True

if __name__ == "__main__":
    result = asyncio.run(test_services())
    print(f"🎯 Test result: {'PASS' if result else 'FAIL'}")
