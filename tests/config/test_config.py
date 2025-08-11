#!/usr/bin/env python3
"""
Centralized test configuration for Elevate AI API
Provides configuration management for all test types
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class TestConfig:
    """Test configuration dataclass"""
    # Test execution settings
    timeout: float = 60.0
    retry_attempts: int = 3
    retry_delay: float = 2.0
    test_iterations: int = 3
    concurrent_requests: int = 5
    
    # Performance thresholds
    performance_threshold: float = 30.0
    cost_threshold: float = 1.0
    success_rate_threshold: float = 0.95
    
    # Response time targets (seconds)
    simple_query_response_time: float = 5.0
    medium_query_response_time: float = 15.0
    complex_query_response_time: float = 30.0
    concurrent_response_time: float = 20.0
    cost_per_query: float = 0.01
    
    # Load testing
    concurrent_users: list = None
    test_duration: int = 300
    ramp_up_time: int = 60
    steady_state_time: int = 180
    ramp_down_time: int = 60
    
    # Monitoring
    enable_metrics_collection: bool = True
    metrics_interval: int = 10
    enable_alerting: bool = True
    
    # Reporting
    generate_html_report: bool = True
    generate_json_report: bool = True
    include_performance_graphs: bool = True
    save_screenshots: bool = False
    report_directory: str = "./e2e_test_reports"
    
    def __post_init__(self):
        if self.concurrent_users is None:
            self.concurrent_users = [1, 5, 10, 20]

@dataclass
class APIConfig:
    """API endpoint configuration"""
    ai_api_base_url: str = "http://localhost:8000"
    core_api_base_url: str = "http://localhost:3000"
    ai_api_health: str = "/api/health"
    premium_api_health: str = "/api/v1/premium/health"
    core_api_health: str = "/health"

@dataclass
class TestQueries:
    """Test query configurations"""
    simple: list = None
    medium: list = None
    complex: list = None
    
    def __post_init__(self):
        if self.simple is None:
            self.simple = [
                "Hello, how are you?",
                "What is the weather like?",
                "Explain a simple concept"
            ]
        if self.medium is None:
            self.medium = [
                "Explain the basics of machine learning",
                "What are the key differences between supervised and unsupervised learning?",
                "How does a neural network work? Explain with examples."
            ]
        if self.complex is None:
            self.complex = [
                "Analyze the impact of AI on healthcare and provide detailed recommendations with examples",
                "Explain quantum mechanics with examples and mathematical formulations",
                "Provide a comprehensive analysis of blockchain technology and its applications"
            ]

class TestConfigurationManager:
    """Manages test configuration loading and access"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or self._get_default_config_path()
        self._config = None
        self._api_config = None
        self._queries = None
        
    def _get_default_config_path(self) -> Path:
        """Get the default configuration file path"""
        return Path(__file__).parent / "e2e_test_config.json"
    
    def load_config(self) -> TestConfig:
        """Load configuration from file or use defaults"""
        if self._config is not None:
            return self._config
            
        if self.config_file and self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                
                # Extract test configuration
                test_config_data = config_data.get('test_configuration', {})
                self._config = TestConfig(**test_config_data)
                
                # Extract API configuration
                api_config_data = config_data.get('api_endpoints', {})
                self._api_config = APIConfig(**api_config_data)
                
                # Extract test queries
                queries_data = config_data.get('test_queries', {})
                self._queries = TestQueries(**queries_data)
                
                print(f"✅ Loaded configuration from {self.config_file}")
                
            except Exception as e:
                print(f"⚠️  Failed to load config from {self.config_file}: {e}")
                print("Using default configuration")
                self._config = TestConfig()
                self._api_config = APIConfig()
                self._queries = TestQueries()
        else:
            print("📝 No config file found, using default configuration")
            self._config = TestConfig()
            self._api_config = APIConfig()
            self._queries = TestQueries()
        
        return self._config
    
    def get_api_config(self) -> APIConfig:
        """Get API configuration"""
        if self._api_config is None:
            self.load_config()
        return self._api_config
    
    def get_queries(self) -> TestQueries:
        """Get test queries"""
        if self._queries is None:
            self.load_config()
        return self._queries
    
    def get_config(self) -> TestConfig:
        """Get test configuration"""
        if self._config is None:
            self.load_config()
        return self._config
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values"""
        config = self.get_config()
        for key, value in updates.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                print(f"⚠️  Unknown config key: {key}")
    
    def save_config(self, filepath: Optional[str] = None) -> None:
        """Save current configuration to file"""
        filepath = filepath or self.config_file
        
        config_data = {
            "test_configuration": asdict(self.get_config()),
            "api_endpoints": asdict(self.get_api_config()),
            "test_queries": asdict(self.get_queries())
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(config_data, f, indent=2)
            print(f"💾 Configuration saved to {filepath}")
        except Exception as e:
            print(f"❌ Failed to save configuration: {e}")
    
    def get_environment_overrides(self) -> Dict[str, Any]:
        """Get configuration overrides from environment variables"""
        overrides = {}
        
        # Test execution overrides
        if os.getenv('TEST_TIMEOUT'):
            overrides['timeout'] = float(os.getenv('TEST_TIMEOUT'))
        if os.getenv('TEST_ITERATIONS'):
            overrides['test_iterations'] = int(os.getenv('TEST_ITERATIONS'))
        if os.getenv('TEST_CONCURRENT_REQUESTS'):
            overrides['concurrent_requests'] = int(os.getenv('TEST_CONCURRENT_REQUESTS'))
        
        # Performance overrides
        if os.getenv('TEST_COST_THRESHOLD'):
            overrides['cost_threshold'] = float(os.getenv('TEST_COST_THRESHOLD'))
        if os.getenv('TEST_PERFORMANCE_THRESHOLD'):
            overrides['performance_threshold'] = float(os.getenv('TEST_PERFORMANCE_THRESHOLD'))
        
        # API endpoint overrides
        if os.getenv('AI_API_BASE_URL'):
            overrides['ai_api_base_url'] = os.getenv('AI_API_BASE_URL')
        if os.getenv('CORE_API_BASE_URL'):
            overrides['core_api_base_url'] = os.getenv('CORE_API_BASE_URL')
        
        return overrides

# Global configuration instance
config_manager = TestConfigurationManager()

def get_test_config() -> TestConfig:
    """Get the global test configuration"""
    return config_manager.get_config()

def get_api_config() -> APIConfig:
    """Get the global API configuration"""
    return config_manager.get_api_config()

def get_test_queries() -> TestQueries:
    """Get the global test queries"""
    return config_manager.get_queries()

def update_test_config(updates: Dict[str, Any]) -> None:
    """Update the global test configuration"""
    config_manager.update_config(updates)

# Environment-based configuration
def load_environment_config() -> None:
    """Load configuration overrides from environment variables"""
    overrides = config_manager.get_environment_overrides()
    if overrides:
        config_manager.update_config(overrides)
        print(f"🔧 Applied environment overrides: {overrides}")

# Initialize configuration on import
load_environment_config()

