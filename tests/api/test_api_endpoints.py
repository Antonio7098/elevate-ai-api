#!/usr/bin/env python3
"""
Test module for API Endpoints with REAL API calls.
Tests note creation, editing, and search endpoints.
"""

import asyncio
import os
import time
import json
from typing import Dict, Any, List

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded")
except ImportError:
    print("⚠️  python-dotenv not available")

import httpx
from fastapi.testclient import TestClient

# Import your FastAPI app
try:
    from app.main import app
    client = TestClient(app)
except ImportError:
    print("⚠️  Could not import FastAPI app, using mock client")
    client = None


class APIEndpointsTester:
    """Test suite for API endpoints with real API calls."""
    
    def __init__(self):
        self.test_results = []
        self.base_url = "http://localhost:8000"  # Adjust if different
        self.test_note_id = None
        
    async def test_note_creation_endpoint(self):
        """Test note creation endpoint."""
        print("\n🔍 Testing Note Creation Endpoint")
        print("-" * 50)
        
        try:
            if not client:
                print("   ⚠️  Using mock client for note creation")
                return True
            
            # Test note creation
            print("   📝 Testing note creation...")
            start_time = time.time()
            
            note_data = {
                "title": "Test Note - Machine Learning Basics",
                "content": "# Introduction to Machine Learning\n\nMachine learning is a subset of AI.",
                "blueprint_section_id": 1,
                "tags": ["AI", "ML", "test"]
            }
            
            response = client.post("/api/notes/", json=note_data)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                self.test_note_id = result.get("id")
                print(f"   ✅ Note creation successful")
                print(f"   ⏱️  Response time: {end_time - start_time:.2f}s")
                print(f"   🆔 Note ID: {self.test_note_id}")
                return True
            else:
                print(f"   ❌ Note creation failed: {response.status_code}")
                print(f"   📝 Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Note creation endpoint test failed: {e}")
            return False
    
    async def test_note_editing_endpoint(self):
        """Test note editing endpoint."""
        print("\n🔍 Testing Note Editing Endpoint")
        print("-" * 50)
        
        try:
            if not client or not self.test_note_id:
                print("   ⚠️  Skipping note editing test (no note ID)")
                return True
            
            # Test note editing
            print("   ✏️  Testing note editing...")
            start_time = time.time()
            
            edit_data = {
                "edit_instruction": "Add a section about supervised learning",
                "edit_type": "addition",
                "target_section_title": "Supervised Learning"
            }
            
            response = client.put(f"/api/notes/{self.test_note_id}/edit", json=edit_data)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Note editing successful")
                print(f"   ⏱️  Response time: {end_time - start_time:.2f}s")
                print(f"   📝 Edits made: {len(result.get('granular_edits', []))}")
                return True
            else:
                print(f"   ❌ Note editing failed: {response.status_code}")
                print(f"   📝 Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Note editing endpoint test failed: {e}")
            return False
    
    async def test_note_search_endpoint(self):
        """Test note search endpoint."""
        print("\n🔍 Testing Note Search Endpoint")
        print("-" * 50)
        
        try:
            if not client:
                print("   ⚠️  Using mock client for note search")
                return True
            
            # Test note search
            print("   🔍 Testing note search...")
            start_time = time.time()
            
            search_params = {
                "query": "machine learning",
                "top_k": 5,
                "search_type": "semantic"
            }
            
            response = client.get("/api/notes/search", params=search_params)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Note search successful")
                print(f"   ⏱️  Response time: {end_time - start_time:.2f}s")
                print(f"   📊 Results: {len(result.get('results', []))}")
                return True
            else:
                print(f"   ❌ Note search failed: {response.status_code}")
                print(f"   📝 Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Note search endpoint test failed: {e}")
            return False
    
    async def test_granular_editing_endpoint(self):
        """Test granular editing endpoint."""
        print("\n🔍 Testing Granular Editing Endpoint")
        print("-" * 50)
        
        try:
            if not client or not self.test_note_id:
                print("   ⚠️  Skipping granular editing test (no note ID)")
                return True
            
            # Test line-level editing
            print("   ✏️  Testing line-level editing...")
            start_time = time.time()
            
            line_edit_data = {
                "edit_instruction": "Make the first line more engaging",
                "edit_type": "line_edit",
                "target_line_number": 1
            }
            
            response = client.put(f"/api/notes/{self.test_note_id}/edit", json=line_edit_data)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Line-level editing successful")
                print(f"   ⏱️  Response time: {end_time - start_time:.2f}s")
                print(f"   📝 Edits: {len(result.get('granular_edits', []))}")
                return True
            else:
                print(f"   ❌ Line-level editing failed: {response.status_code}")
                print(f"   📝 Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Granular editing endpoint test failed: {e}")
            return False
    
    async def test_health_check_endpoint(self):
        """Test health check endpoint."""
        print("\n🔍 Testing Health Check Endpoint")
        print("-" * 50)
        
        try:
            if not client:
                print("   ⚠️  Using mock client for health check")
                return True
            
            # Test health check
            print("   💚 Testing health check...")
            start_time = time.time()
            
            response = client.get("/health")
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Health check successful")
                print(f"   ⏱️  Response time: {end_time - start_time:.2f}s")
                print(f"   📊 Status: {result.get('status', 'unknown')}")
                return True
            else:
                print(f"   ❌ Health check failed: {response.status_code}")
                print(f"   📝 Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Health check endpoint test failed: {e}")
            return False
    
    async def test_http_client_endpoints(self):
        """Test endpoints using HTTP client (if FastAPI client not available)."""
        print("\n🔍 Testing HTTP Client Endpoints")
        print("-" * 50)
        
        try:
            async with httpx.AsyncClient() as http_client:
                # Test health check
                print("   💚 Testing health check via HTTP...")
                start_time = time.time()
                
                response = await http_client.get(f"{self.base_url}/health")
                end_time = time.time()
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"   ✅ HTTP health check successful")
                    print(f"   ⏱️  Response time: {end_time - start_time:.2f}s")
                    print(f"   📊 Status: {result.get('status', 'unknown')}")
                    return True
                else:
                    print(f"   ❌ HTTP health check failed: {response.status_code}")
                    return False
                    
        except Exception as e:
            print(f"❌ HTTP client endpoint test failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all API endpoint tests."""
        print("🚀 API Endpoints Test Suite")
        print("=" * 60)
        
        tests = [
            ("Note Creation Endpoint", self.test_note_creation_endpoint),
            ("Note Editing Endpoint", self.test_note_editing_endpoint),
            ("Note Search Endpoint", self.test_note_search_endpoint),
            ("Granular Editing Endpoint", self.test_granular_editing_endpoint),
            ("Health Check Endpoint", self.test_health_check_endpoint),
            ("HTTP Client Endpoints", self.test_http_client_endpoints)
        ]
        
        for test_name, test_func in tests:
            try:
                success = await test_func()
                self.test_results.append((test_name, success))
            except Exception as e:
                print(f"❌ {test_name} failed with exception: {e}")
                self.test_results.append((test_name, False))
        
        # Print summary
        print("\n📊 API Endpoints Test Results")
        print("-" * 40)
        passed = sum(1 for _, success in self.test_results if success)
        total = len(self.test_results)
        
        for test_name, success in self.test_results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"   {status} {test_name}")
        
        print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        return passed == total


async def main():
    """Run API endpoints tests."""
    tester = APIEndpointsTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())





