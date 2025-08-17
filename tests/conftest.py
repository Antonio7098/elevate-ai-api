"""
Pytest configuration and fixtures for Elevate AI API tests
"""

import pytest
import os
import sys
import asyncio
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure pytest-asyncio
pytest_plugins = ['pytest_asyncio']

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def test_config():
    """Load test configuration"""
    config_path = project_root / "tests" / "e2e" / "e2e_test_config.json"
    
    if config_path.exists():
        import json
        with open(config_path) as f:
            return json.load(f)
    else:
        return {
            "api_base_url": "http://localhost:8000",
            "core_api_url": "http://localhost:3000",
            "test_queries": [
                "Explain the concept of machine learning in simple terms",
                "What are the key differences between supervised and unsupervised learning?",
                "How does a neural network work? Explain with examples",
                "What is the impact of AI on modern healthcare?",
                "Explain the concept of transfer learning in deep learning"
            ]
        }

def get_test_config():
    """Get test configuration for non-fixture usage"""
    config_path = project_root / "tests" / "e2e" / "e2e_test_config.json"
    
    if config_path.exists():
        import json
        with open(config_path) as f:
            return json.load(f)
    else:
        return {
            "api_base_url": "http://localhost:8000",
            "core_api_url": "http://localhost:3000",
            "test_queries": [
                "Explain the concept of machine learning in simple terms",
                "What are the key differences between supervised and unsupervised learning?",
                "How does a neural network work? Explain with examples",
                "What is the impact of AI on modern healthcare?",
                "Explain the concept of transfer learning in deep learning"
            ]
        }

@pytest.fixture(scope="session")
def api_client():
    """Create API client for testing"""
    import httpx
    
    base_url = os.getenv("TEST_API_URL", "http://localhost:8000")
    
    with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        yield client

@pytest.fixture(scope="session")
def test_user():
    """Test user data"""
    return {
        "user_id": "test-premium-user-123",
        "user_tier": "premium",
        "learning_style": "VISUAL",
        "cognitive_approach": "BALANCED"
    }

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment variables"""
    # Set test environment
    os.environ["TESTING"] = "true"
    
    # Ensure required API keys are available for testing
    if not os.getenv("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = "test-key"
    if not os.getenv("OPENROUTER_API_KEY"):
        os.environ["OPENROUTER_API_KEY"] = "test-key"
    
    yield
    
    # Cleanup
    if os.getenv("TESTING"):
        del os.environ["TESTING"] 