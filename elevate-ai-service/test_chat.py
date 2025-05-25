"""
Test script for the /api/chat endpoint.

This script sends test requests to the /api/chat endpoint
to verify that it's working correctly.
"""

import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_URL = "http://localhost:8000/chat"
API_KEY = os.getenv("CORE_API_ACCESS_KEY", "your-core-api-access-key-here")

# Test case
test_payload = {
    "message": "Can you explain the concept of photosynthesis?",
    "conversation": [
        {
            "role": "user",
            "content": "What is biology?"
        },
        {
            "role": "assistant",
            "content": "Biology is the scientific study of life and living organisms, including their physical structure, chemical processes, molecular interactions, physiological mechanisms, development, and evolution."
        }
    ],
    "context": {
        "questionSets": [
            {
                "id": 1,
                "name": "Biology 101",
                "questions": [
                    {
                        "text": "What is photosynthesis?",
                        "answer": "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll."
                    }
                ]
            }
        ],
        "userLevel": "beginner",
        "preferredLearningStyle": "visual"
    },
    "language": "en"
}

def run_test():
    """Run the test and print the results."""
    print("Testing /api/chat endpoint...\n")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    print(f"Sending request about photosynthesis...")
    
    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json=test_payload,
            timeout=60  # Longer timeout for LLM processing
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Success!")
            
            # Print the response
            response_data = result.get('response', {})
            print("\nAI Response:")
            print(f"{response_data.get('message', '')[:200]}..." if len(response_data.get('message', '')) > 200 else response_data.get('message', ''))
            
            # Print references if any
            references = response_data.get('references', [])
            if references:
                print("\nReferences:")
                for ref in references:
                    print(f"- {ref.get('text', '')[:100]}... (Source: {ref.get('source', '')})")
            
            # Print suggested questions
            suggested_questions = response_data.get('suggestedQuestions', [])
            if suggested_questions:
                print("\nSuggested Questions:")
                for i, question in enumerate(suggested_questions):
                    print(f"{i+1}. {question}")
                
            print(f"\nProcessing Time: {result.get('metadata', {}).get('processingTime')}")
            print(f"Tokens Used: {result.get('metadata', {}).get('tokensUsed')}")
        else:
            print(f"Error: {response.text}")
                
    except Exception as e:
        print(f"Exception: {str(e)}")

if __name__ == "__main__":
    run_test()
