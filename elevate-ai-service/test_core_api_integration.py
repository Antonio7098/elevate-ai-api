"""
Test script to verify Core API integration with the Python AI Service.
"""
import os
import sys
import json
import requests
from typing import Dict, Any

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configuration
BASE_URL = "http://localhost:8000"
API_KEY = os.getenv("CORE_API_ACCESS_KEY")

def make_request(endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Make a request to the specified endpoint with the given data."""
    url = f"{BASE_URL}{endpoint}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making request to {endpoint}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        sys.exit(1)

def test_evaluate_answer():
    """Test the /evaluate-answer endpoint."""
    print("\n=== Testing /evaluate-answer endpoint ===")
    
    request_data = {
        "questionContext": {
            "questionId": "test-123",
            "questionText": "What is the capital of France?",
            "expectedAnswer": "Paris",
            "questionType": "short-answer"
        },
        "userAnswer": "London"
    }
    
    print("Sending evaluation request...")
    response = make_request("/evaluate-answer", request_data)
    
    print("\nEvaluation Response:")
    print(json.dumps(response, indent=2))
    
    # Validate response structure
    required_fields = ["success", "evaluation", "metadata"]
    for field in required_fields:
        if field not in response:
            print(f"❌ Missing required field in response: {field}")
            return False
    
    print("✅ /evaluate-answer test passed!")
    return True

def test_generate_questions():
    """Test the /generate-questions endpoint."""
    print("\n=== Testing /generate-questions endpoint ===")
    
    request_data = {
        "sourceText": "The mitochondria is the powerhouse of the cell. It generates energy in the form of ATP.",
        "questionCount": 2,
        "questionTypes": ["multiple-choice", "true-false"],
        "difficulty": "medium"
    }
    
    print("Sending question generation request...")
    response = make_request("/generate-questions", request_data)
    
    print("\nQuestion Generation Response:")
    print(json.dumps(response, indent=2))
    
    # Validate response structure
    required_fields = ["success", "questions", "metadata"]
    for field in required_fields:
        if field not in response:
            print(f"❌ Missing required field in response: {field}")
            return False
    
    print("✅ /generate-questions test passed!")
    return True

def test_chat():
    """Test the /chat endpoint."""
    print("\n=== Testing /chat endpoint ===")
    
    request_data = {
        "message": "What is the function of mitochondria?",
        "conversation": [
            {"role": "user", "content": "Hello, I have a question about biology."},
            {"role": "assistant", "content": "I'd be happy to help with your biology question! What would you like to know?"}
        ]
    }
    
    print("Sending chat request...")
    response = make_request("/chat", request_data)
    
    print("\nChat Response:")
    print(json.dumps(response, indent=2))
    
    # Validate response structure
    required_fields = ["success", "response", "metadata"]
    for field in required_fields:
        if field not in response:
            print(f"❌ Missing required field in response: {field}")
            return False
    
    print("✅ /chat test passed!")
    return True

def main():
    """Run all tests."""
    print(f"Testing integration with Python AI Service at {BASE_URL}")
    
    # Test endpoints
    test_evaluate_answer()
    test_generate_questions()
    test_chat()
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()
