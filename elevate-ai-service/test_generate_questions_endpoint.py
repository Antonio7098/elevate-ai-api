"""
Test script for the /generate-questions endpoint.
"""
import os
import sys
import json
import requests
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
BASE_URL = "http://localhost:8000"
API_KEY = os.getenv("CORE_API_ACCESS_KEY")

def test_generate_questions():
    """Test the /generate-questions endpoint with various inputs."""
    print("\n=== Testing /generate-questions endpoint ===")
    
    # Test case 1: Basic test with minimum required fields
    print("\nTest 1: Basic test with minimum required fields")
    request_data = {
        "sourceText": "The mitochondria is the powerhouse of the cell. It generates energy in the form of ATP.",
        "questionCount": 2,
        "questionTypes": ["multiple-choice", "true-false"],
        "difficulty": "medium"
    }
    
    print("Sending request 1...")
    response = make_request("/generate-questions", request_data)
    print_response(response)
    
    # Test case 2: Test with all possible fields
    print("\nTest 2: Test with all possible fields")
    request_data = {
        "sourceText": "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll.",
        "questionCount": 3,
        "questionTypes": ["multiple-choice", "true-false", "short-answer"],
        "difficulty": "hard",
        "topics": ["biology", "photosynthesis"],
        "language": "en"
    }
    
    print("Sending request 2...")
    response = make_request("/generate-questions", request_data)
    print_response(response)
    
    # Test case 3: Test with different language
    print("\nTest 3: Test with Spanish language")
    request_data = {
        "sourceText": "La fotosíntesis es el proceso por el cual las plantas verdes y algunos otros organismos usan la luz solar para sintetizar alimentos con la ayuda de la clorofila.",
        "questionCount": 2,
        "questionTypes": ["multiple-choice", "true-false"],
        "difficulty": "medium",
        "language": "es"
    }
    
    print("Sending request 3...")
    response = make_request("/generate-questions", request_data)
    print_response(response)
    
    # Test case 4: Test with invalid input (missing required field)
    print("\nTest 4: Test with invalid input (missing sourceText)")
    request_data = {
        "questionCount": 2,
        "questionTypes": ["multiple-choice"],
        "difficulty": "easy"
    }
    
    print("Sending request 4...")
    response = make_request("/generate-questions", request_data)
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
            if 'questions' in data:
                print(f"Generated {len(data['questions'])} questions:")
                for i, question in enumerate(data['questions'], 1):
                    print(f"\nQuestion {i}:")
                    print(f"  Text: {question.get('text', 'N/A')}")
                    print(f"  Type: {question.get('questionType', 'N/A')}")
                    print(f"  Answer: {question.get('answer', 'N/A')}")
                    if 'options' in question:
                        print(f"  Options: {', '.join(question['options'])}")
                    if 'explanation' in question:
                        print(f"  Explanation: {question['explanation']}")
            
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
    
    print(f"Testing /generate-questions endpoint with API key: {API_KEY[:5]}...{API_KEY[-5:]}")
    test_generate_questions()
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()
