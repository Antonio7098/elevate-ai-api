# Test Organization Structure

This document describes the organized test structure for the Elevate AI API project.

## Overview

The test files have been organized into a clear structure that separates different types of tests and makes them easier to manage and run.

## Directory Structure

```
tests/
├── pytest/           # Pytest-compatible test files
├── standalone/       # Standalone test scripts (not pytest-compatible)
├── debug/           # Debug and utility scripts
├── config/          # Test configuration files
├── utils/           # Test utility scripts
└── conftest.py      # Pytest configuration
```

## Test Categories

### 1. Pytest Tests (`tests/pytest/`)
- **Purpose**: Standard pytest-compatible test files
- **Naming**: Files starting with `test_`
- **Execution**: Can be run with `pytest` or the test runner
- **Count**: 11 files
- **Examples**: 
  - `test_api_endpoints.py`
  - `test_blueprint_ingestion.py`
  - `test_embeddings.py`

### 2. Standalone Tests (`tests/standalone/`)
- **Purpose**: Test scripts designed to run independently
- **Naming**: Files starting with `test_` or `test-`
- **Execution**: Must be run individually as they're not pytest-compatible
- **Count**: 31 files
- **Examples**:
  - `test_e2e_real_llm_performance.py`
  - `test_sprint38_cost_optimization.py`
  - `test_e2e_premium_chat_workflow.py`

### 3. Debug Scripts (`tests/debug/`)
- **Purpose**: Debugging and utility scripts
- **Naming**: Files starting with `debug_`
- **Execution**: Run manually as needed
- **Count**: 14 files

## Running Tests

### Using the Test Runner

```bash
# Discover all available tests
python run_tests.py discover

# Run all tests
python run_tests.py all

# Run only pytest tests
python run_tests.py pytest

# Run only standalone tests (lists available tests)
python run_tests.py standalone

# Run with verbose output
python run_tests.py pytest --verbose
```

### Using Make Commands

```bash
# Discover tests
make test-discover

# Run all tests
make test-all

# Run pytest tests
make test-pytest

# Run standalone tests
make test-standalone

# Clean test artifacts
make test-clean
```

### Direct Execution

```bash
# Run pytest tests directly
python -m pytest tests/pytest/

# Run a specific test file
python -m pytest tests/pytest/test_api_endpoints.py

# Run standalone tests individually
python tests/standalone/test_e2e_real_llm_performance.py
```

## Test File Organization

### Pytest Tests
- Follow standard pytest conventions
- Use `test_` prefix for functions
- Can use fixtures and pytest plugins
- Support for async testing with `pytest-asyncio`

### Standalone Tests
- Designed as complete test scripts
- Often include their own test runners
- May have dependencies on external services
- Can be run independently of pytest

## Benefits of This Organization

1. **Clear Separation**: Pytest vs standalone tests are clearly separated
2. **Easier Management**: Related tests are grouped together
3. **Better Discovery**: Easy to find and run specific test types
4. **Reduced Conflicts**: No more pytest collection warnings from standalone scripts
5. **Flexible Execution**: Can run tests individually or as groups

## Migration Notes

- All test files have been moved from the root directory to organized subdirectories
- The `pytest.ini` configuration file remains in the root for global pytest settings
- Test runner scripts have been updated to work with the new structure
- Makefile commands have been updated to reflect the new organization

## Future Improvements

- Add test categories and tags for better organization
- Implement test result aggregation for standalone tests
- Add performance benchmarking for test execution
- Create test dependency management system
