#!/usr/bin/env python3
"""Debug script to investigate API endpoint 400 errors with mocked blueprint data"""

import sys
import os
sys.path.append('/home/antonio/programming/elevate/core_and_ai/elevate-ai-api')

from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from app.main import app

client = TestClient(app)

# Mock blueprint data
mock_blueprint_data = {
    "source_id": "test-blueprint",
    "source_text": "Photosynthesis is the process by which plants convert sunlight into energy.",
    "source_title": "Test Photosynthesis Blueprint",
    "knowledge_primitives": {
        "key_propositions_and_facts": [
            {
                "id": "prop_1",
                "statement": "Photosynthesis converts sunlight into chemical energy",
                "supporting_evidence": ["Light absorption", "Energy conversion"],
                "sections": ["sec_1"]
            }
        ]
    }
}

# Test with mocked blueprint data
with patch('app.core.indexing._get_blueprint_data', new_callable=AsyncMock) as mock_retrieval:
    mock_retrieval.return_value = mock_blueprint_data
    
    headers = {"Authorization": "Bearer test_api_key_123"}
    response = client.post("/api/v1/ai-rag/learning-blueprints/test-blueprint/question-sets", 
                         json={
                             "name": "Test Question Set",
                             "folder_id": 1,
                             "question_options": {"scope": "KeyConcepts"}
                         },
                         headers=headers)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response Headers: {response.headers}")
    print(f"Response Content: {response.text}")
    
    if response.status_code != 200:
        try:
            error_data = response.json()
            print(f"Error JSON: {error_data}")
        except:
            print("Could not parse response as JSON")
