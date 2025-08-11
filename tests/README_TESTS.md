# Elevate AI API Test Suite

This directory contains the comprehensive test suite for the Elevate AI API, including unit tests, integration tests, E2E tests, and performance tests.

## 🏗️ Test Structure

```
tests/
├── config/                          # Test configuration management
│   ├── __init__.py                 # Package initialization
│   ├── test_config.py              # Centralized test configuration
│   └── e2e_test_config.json        # E2E test configuration file
├── e2e/                            # End-to-end tests
│   ├── __init__.py
│   ├── test_e2e_real_llm_performance.py
│   └── e2e_test_config.json
├── unit/                           # Unit tests
├── integration/                    # Integration tests
├── performance/                    # Performance tests
├── contract/                       # Contract tests
├── utils/                          # Test utilities
│   ├── __init__.py
│   ├── run_e2e_tests.py           # E2E test runner
│   └── run_tests.sh               # Shell script runner
├── conftest.py                     # Pytest configuration
├── pytest.ini                     # Pytest settings
├── requirements-test.txt           # Test dependencies
├── run_all_tests.py               # Main test runner
├── run_tests.py                   # Simple test runner
└── README.md                       # This file
```

## 🚀 Quick Start

### 1. Install Test Dependencies

```bash
# Using pip
pip install -r tests/requirements-test.txt

# Using poetry
poetry install --with dev
```

### 2. Run Tests

```bash
# Run all tests
python tests/run_tests.py

# Run only E2E tests
python tests/run_tests.py --e2e

# Run with coverage
python tests/run_tests.py --coverage

# Run specific test type
python tests/run_tests.py --type unit
```

### 3. Using Make Commands

```bash
# Run all tests
make test-all

# Run E2E tests only
make test-e2e

# Run with custom iterations
make test-e2e ITERATIONS=5

# Discover available tests
make test-discover

# Run with coverage
make test-coverage
```

## ⚙️ Configuration

### Test Configuration

The test suite uses a centralized configuration system located in `tests/config/test_config.py`. This provides:

- **Test execution settings**: timeout, retries, iterations
- **Performance thresholds**: response time targets, cost limits
- **API endpoints**: service URLs and health check endpoints
- **Test queries**: predefined test data for different complexity levels
- **Environment-specific settings**: development, staging, production

### Environment Variables

You can override configuration values using environment variables:

```bash
export TEST_TIMEOUT=120
export TEST_ITERATIONS=5
export TEST_CONCURRENT_REQUESTS=10
export AI_API_BASE_URL=http://localhost:8000
export CORE_API_BASE_URL=http://localhost:3000
```

### Configuration File

The main configuration file is `tests/config/e2e_test_config.json`. You can modify this file to customize:

- Test parameters
- API endpoints
- Performance targets
- Load testing settings
- Reporting options

## 🧪 Test Types

### Unit Tests (`tests/unit/`)
- Test individual functions and classes in isolation
- Fast execution, no external dependencies
- Use mocks and stubs for external services

### Integration Tests (`tests/integration/`)
- Test interactions between components
- May use test databases or mock services
- Moderate execution time

### E2E Tests (`tests/e2e/`)
- Test complete user workflows
- Require running services (API, database)
- Longer execution time, realistic scenarios

### Performance Tests (`tests/performance/`)
- Measure response times and throughput
- Load testing and stress testing
- Performance regression detection

### Contract Tests (`tests/contract/`)
- Verify API contracts and schemas
- Ensure backward compatibility
- Validate data formats

## 📊 Test Execution

### Test Discovery

The test runner automatically discovers tests based on naming conventions:

- Files: `test_*.py` or `*_test.py`
- Classes: `Test*`
- Functions: `test_*`

### Running Specific Tests

```bash
# Run specific test file
pytest tests/unit/test_embeddings.py

# Run specific test function
pytest tests/unit/test_embeddings.py::test_embedding_generation

# Run tests matching pattern
pytest -k "embedding"

# Run tests with specific marker
pytest -m "slow"
```

### Test Markers

The test suite uses pytest markers to categorize tests:

```python
import pytest

@pytest.mark.unit
def test_unit_function():
    pass

@pytest.mark.integration
def test_integration_function():
    pass

@pytest.mark.e2e
def test_e2e_workflow():
    pass

@pytest.mark.slow
def test_slow_operation():
    pass
```

## 📈 Coverage and Reporting

### Coverage Reports

```bash
# Generate HTML coverage report
make test-coverage

# View coverage in terminal
pytest --cov=app --cov-report=term-missing
```

### Test Reports

```bash
# Generate HTML test report
make test-report

# Generate JSON report
pytest --json-report=test_reports/report.json
```

## 🔧 Advanced Configuration

### Custom Test Configuration

```python
from tests.config.test_config import update_test_config

# Update configuration programmatically
update_test_config({
    'timeout': 120.0,
    'test_iterations': 5,
    'concurrent_requests': 10
})
```

### Environment-Specific Settings

```python
from tests.config.test_config import get_test_config, get_api_config

# Get current configuration
config = get_test_config()
api_config = get_api_config()

# Access specific values
timeout = config.timeout
base_url = api_config.ai_api_base_url
```

### Test Categories

The configuration supports different test categories:

- **Smoke**: Basic functionality tests (fast)
- **Regression**: Comprehensive tests (moderate)
- **Performance**: Performance and load tests (slow)
- **Stress**: High-load stress tests (very slow)

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the project root
2. **Configuration Not Found**: Check that `tests/config/` directory exists
3. **Service Unavailable**: Verify that required services are running
4. **Permission Errors**: Ensure test directories are writable

### Debug Mode

```bash
# Enable verbose output
python tests/run_tests.py --verbose

# Check environment
python tests/run_tests.py --check-env

# Discover tests only
python tests/run_tests.py --discover
```

### Logs and Reports

- Test reports: `test_reports/`
- Coverage reports: `htmlcov/`
- E2E test reports: `e2e_test_reports/`

## 🤝 Contributing

### Adding New Tests

1. Follow the naming convention: `test_*.py`
2. Use appropriate markers: `@pytest.mark.unit`
3. Add to the appropriate test directory
4. Update this README if adding new test types

### Test Best Practices

- Write isolated, independent tests
- Use descriptive test names
- Clean up test data after each test
- Mock external dependencies
- Test both success and failure cases

### Configuration Updates

- Update `tests/config/e2e_test_config.json` for new settings
- Add new configuration classes in `test_config.py`
- Document new environment variables
- Update this README

## 📚 Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [Coverage.py](https://coverage.readthedocs.io/)
- [Test Configuration Best Practices](https://docs.pytest.org/en/stable/how-to/configuration.html)

