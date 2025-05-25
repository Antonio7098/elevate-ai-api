"""
Test script for the /evaluate-answer endpoint.
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

def test_evaluate_answer():
    """Test the /evaluate-answer endpoint with various inputs."""
    print("\n=== Testing /evaluate-answer endpoint ===")
    
    # Test case 1: Correct answer (multiple-choice)
    print("\nTest 1: Correct multiple-choice answer")
    request_data = {
        "questionContext": {
            "questionText": "What is the capital of France?",
            "expectedAnswer": "Paris",
            "questionType": "multiple-choice",
            "questionId": "geo-001"
        },
        "userAnswer": "Paris"
    }
    
    print("Sending request 1...")
    response = make_request("/evaluate-answer", request_data)
    print_response(response)
    
    # Test case 2: Incorrect answer (true-false)
    print("\nTest 2: Incorrect true-false answer")
    request_data = {
        "questionContext": {
            "questionText": "The Earth is flat.",
            "expectedAnswer": "false",
            "questionType": "true-false",
            "questionId": "sci-001"
        },
        "userAnswer": "true"
    }
    
    print("Sending request 2...")
    response = make_request("/evaluate-answer", request_data)
    print_response(response)
    
    # Test case 3: Partially correct answer (short-answer)
    print("\nTest 3: Partially correct short answer")
    request_data = {
        "questionContext": {
            "questionText": "Explain the process of photosynthesis.",
            "expectedAnswer": "Photosynthesis is the process by which green plants use sunlight to synthesize foods with the help of chlorophyll, converting carbon dioxide and water into glucose and oxygen.",
            "questionType": "short-answer",
            "questionId": "bio-001"
        },
        "userAnswer": "Photosynthesis is when plants make food using sunlight and air."
    }
    
    print("Sending request 3...")
    response = make_request("/evaluate-answer", request_data)
    print_response(response)
    
    # Test case 4: Missing required field
    print("\nTest 4: Missing required field (questionContext)")
    request_data = {
        "userAnswer": "42"
    }
    
    print("Sending request 4...")
    response = make_request("/evaluate-answer", request_data)
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
            if 'evaluation' in data:
                eval_data = data['evaluation']
                print("\nEvaluation:")
                print(f"  Is Correct: {eval_data.get('isCorrect', 'N/A')}")
                print(f"  Is Partially Correct: {eval_data.get('isPartiallyCorrect', 'N/A')}")
                print(f"  Score: {eval_data.get('score', 'N/A')}")
                print(f"  Feedback: {eval_data.get('feedbackText', 'N/A')}")
                print(f"  Suggested Answer: {eval_data.get('suggestedCorrectAnswer', 'N/A')}")
            
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
    
    print(f"Testing /evaluate-answer endpoint with API key: {API_KEY[:5]}...{API_KEY[-5:]}")
    test_evaluate_answer()
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()
