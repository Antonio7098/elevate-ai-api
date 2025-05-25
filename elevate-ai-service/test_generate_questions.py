"""
Test script for the /api/generate-questions endpoint.

This script sends test requests to the /api/generate-questions endpoint
to verify that it's working correctly.
"""

import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_URL = "http://localhost:8000/generate-questions"
API_KEY = os.getenv("CORE_API_ACCESS_KEY", "your-core-api-access-key-here")

# Test case
test_payload = {
    "sourceText": "The water cycle, also known as the hydrologic cycle, describes the continuous movement of water on, above, and below the surface of the Earth. Water can change states among liquid, vapor, and ice at various places in the water cycle. Although the balance of water on Earth remains fairly constant over time, individual water molecules can come and go. The water moves from one reservoir to another, such as from river to ocean, or from the ocean to the atmosphere, by the physical processes of evaporation, condensation, precipitation, infiltration, surface runoff, and subsurface flow. In doing so, the water goes through different forms: liquid, solid (ice) and vapor. The water cycle involves the exchange of energy, which leads to temperature changes. When water evaporates, it takes up energy from its surroundings and cools the environment. When it condenses, it releases energy and warms the environment. These heat exchanges influence climate.",
    "questionCount": 3,
    "questionTypes": ["multiple-choice", "true-false", "short-answer"],
    "difficulty": "medium",
    "topics": ["water cycle", "earth science"],
    "language": "en"
}

def run_test():
    """Run the test and print the results."""
    print("Testing /api/generate-questions endpoint...\n")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    print(f"Sending request with source text about the water cycle...")
    
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
            print(f"Number of questions generated: {len(result.get('questions', []))}")
            
            # Print a sample of each question type
            questions = result.get('questions', [])
            for i, question in enumerate(questions):
                print(f"\nQuestion {i+1}:")
                print(f"Type: {question.get('questionType')}")
                print(f"Text: {question.get('text')}")
                if question.get('questionType') == 'multiple-choice':
                    print(f"Options: {', '.join(question.get('options', []))}")
                print(f"Answer: {question.get('answer')}")
                print(f"Explanation: {question.get('explanation')[:100]}..." if len(question.get('explanation', '')) > 100 else question.get('explanation'))
                
            print(f"\nProcessing Time: {result.get('metadata', {}).get('processingTime')}")
        else:
            print(f"Error: {response.text}")
                
    except Exception as e:
        print(f"Exception: {str(e)}")

if __name__ == "__main__":
    run_test()
