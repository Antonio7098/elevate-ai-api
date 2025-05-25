"""
Test script for the /chat endpoint.
"""
import os
import sys
import json
import requests
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
BASE_URL = "http://localhost:8000"
API_KEY = os.getenv("CORE_API_ACCESS_KEY")

def test_chat():
    """Test the /chat endpoint with various conversation scenarios."""
    print("\n=== Testing /chat endpoint ===")
    
    # Test case 1: Simple greeting
    print("\nTest 1: Simple greeting")
    request_data = {
        "message": "Hello, how are you?",
        "conversation": [],
        "context": {},
        "language": "en"
    }
    
    print("Sending request 1...")
    response = make_request("/chat", request_data)
    print_response(response)
    
    # Store the response for the next test
    first_response = response.get('data', {})
    
    # Test case 2: Follow-up question in the conversation
    print("\nTest 2: Follow-up in conversation")
    request_data = {
        "message": "Can you tell me more about that?",
        "conversation": [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": first_response.get('message', '')}
        ],
        "context": {},
        "language": "en"
    }
    
    print("Sending request 2...")
    response = make_request("/chat", request_data)
    print_response(response)
    
    # Test case 3: With context about the subject
    print("\nTest 3: With subject context")
    request_data = {
        "message": "What are the main topics in biology?",
        "conversation": [],
        "context": {
            "subject": "biology",
            "gradeLevel": "high school"
        },
        "language": "en"
    }
    
    print("Sending request 3...")
    response = make_request("/chat", request_data)
    print_response(response)
    
    # Test case 4: In a different language (Spanish)
    print("\nTest 4: Spanish language")
    request_data = {
        "message": "¿Cuál es la capital de España?",
        "conversation": [],
        "context": {},
        "language": "es"
    }
    
    print("Sending request 4...")
    response = make_request("/chat", request_data)
    print_response(response)
    
    # Test case 5: Missing required field
    print("\nTest 5: Missing required field (message)")
    request_data = {
        "conversation": [],
        "context": {},
        "language": "en"
    }
    
    print("Sending request 5...")
    response = make_request("/chat", request_data)
    print_response(response)

def make_request(endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Make a request to the specified endpoint with the given data."""
    url = f"{BASE_URL}{endpoint}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        return {
            "status_code": response.status_code,
            "data": response.json() if response.content else {}
        }
    except requests.exceptions.RequestException as e:
        print(f"Error making request to {endpoint}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            return {
                "status_code": e.response.status_code,
                "data": e.response.text
            }
        return {
            "status_code": 500,
            "data": {"error": str(e)}
        }

def print_response(response: Dict[str, Any]) -> None:
    """Print the response in a formatted way."""
    print(f"Status Code: {response.get('status_code')}")
    print("Response:")
    
    data = response.get('data', {})
    if isinstance(data, dict):
        if 'success' in data and not data['success']:
            print(f"❌ Error: {data.get('error', {}).get('message', 'Unknown error')}")
        else:
            print("✅ Success!")
            if 'message' in data:
                print(f"\nMessage: {data['message']}")
            
            if 'suggestedQuestions' in data and data['suggestedQuestions']:
                print("\nSuggested Questions:")
                for i, question in enumerate(data['suggestedQuestions'], 1):
                    print(f"  {i}. {question}")
            
            if 'references' in data and data['references']:
                print("\nReferences:")
                for ref in data['references']:
                    print(f"  - {ref.get('text', '')} (Source: {ref.get('source', 'N/A')})")
            
            if 'metadata' in data:
                print("\nMetadata:")
                for key, value in data['metadata'].items():
                    print(f"  {key}: {value}")
    else:
        print(data)

def main():
    """Run the tests."""
    if not API_KEY:
        print("Error: CORE_API_ACCESS_KEY environment variable is not set")
        sys.exit(1)
    
    print(f"Testing /chat endpoint with API key: {API_KEY[:5]}...{API_KEY[-5:]}")
    test_chat()
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()
