"""
Test script for the /evaluate-answer endpoint.

This script sends test requests to the /evaluate-answer endpoint
to verify that it's working correctly.
"""

import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_URL = "http://localhost:8000/evaluate-answer"
API_KEY = os.getenv("CORE_API_ACCESS_KEY", "your-core-api-access-key-here")

# Test cases
test_cases = [
    {
        "name": "Multiple Choice - Correct Answer",
        "payload": {
            "questionContext": {
                "questionId": "mc1",
                "questionText": "What is the capital of France?",
                "expectedAnswer": "Paris",
                "questionType": "multiple-choice"
            },
            "userAnswer": "Paris"
        }
    },
    {
        "name": "Multiple Choice - Incorrect Answer",
        "payload": {
            "questionContext": {
                "questionId": "mc2",
                "questionText": "What is the capital of France?",
                "expectedAnswer": "Paris",
                "questionType": "multiple-choice"
            },
            "userAnswer": "London"
        }
    },
    {
        "name": "True/False - Correct Answer",
        "payload": {
            "questionContext": {
                "questionId": "tf1",
                "questionText": "Paris is the capital of France.",
                "expectedAnswer": "true",
                "questionType": "true-false"
            },
            "userAnswer": "true"
        }
    },
    {
        "name": "Short Answer - Partially Correct",
        "payload": {
            "questionContext": {
                "questionId": "sa1",
                "questionText": "Explain the process of photosynthesis.",
                "expectedAnswer": "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll.",
                "questionType": "short-answer"
            },
            "userAnswer": "Photosynthesis is how plants make food using sunlight."
        }
    }
]

def run_tests():
    """Run all test cases and print the results."""
    print("Testing /evaluate-answer endpoint...\n")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    for test_case in test_cases:
        print(f"Test: {test_case['name']}")
        print(f"Sending request...")
        
        try:
            response = requests.post(
                API_URL,
                headers=headers,
                json=test_case['payload'],
                timeout=30  # Longer timeout for LLM processing
            )
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("Success!")
                print(f"Is Correct: {result['evaluation']['isCorrect']}")
                print(f"Is Partially Correct: {result['evaluation']['isPartiallyCorrect']}")
                print(f"Score: {result['evaluation']['score']}")
                print(f"Feedback: {result['evaluation']['feedbackText'][:100]}...")  # Truncate long feedback
                print(f"Processing Time: {result['metadata']['processingTime']}")
            else:
                print(f"Error: {response.text}")
                
        except Exception as e:
            print(f"Exception: {str(e)}")
            
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    run_tests()
