# Elevate AI API Makefile
# Provides easy commands for development and testing

.PHONY: help install test test-e2e test-unit test-integration test-all clean lint format run-server check-env test-setup test-discover test-coverage test-report

# Default target
help:
	@echo "Elevate AI API - Available Commands:"
	@echo ""
	@echo "Development:"
	@echo "  install      Install dependencies"
	@echo "  run-server   Start the development server"
	@echo "  check-env    Check environment variables"
	@echo ""
	@echo "Testing:"
	@echo "  test         Run all tests"
	@echo "  test-e2e     Run E2E tests only"
	@echo "  test-unit    Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-all     Run comprehensive test suite"
	@echo "  test-discover Discover available tests"
	@echo "  test-coverage Run tests with coverage report"
	@echo "  test-report  Generate test reports"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint         Run linting checks"
	@echo "  format       Format code with black"
	@echo "  clean        Clean up generated files"
	@echo ""
	@echo "Examples:"
	@echo "  make test-e2e ITERATIONS=3"
	@echo "  make test-all VERBOSE=1"
	@echo "  make test-discover"
	@echo "  make test-coverage"

# Development
install:
	@echo "📦 Installing dependencies..."
	poetry install

run-server:
	@echo "🚀 Starting development server..."
	@echo "Make sure you have set GOOGLE_API_KEY and OPENROUTER_API_KEY"
	poetry run python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

check-env:
	@echo "🔍 Checking environment variables..."
	@if [ -z "$$GOOGLE_API_KEY" ]; then echo "❌ GOOGLE_API_KEY not set"; else echo "✅ GOOGLE_API_KEY set"; fi
	@if [ -z "$$OPENROUTER_API_KEY" ]; then echo "❌ OPENROUTER_API_KEY not set"; else echo "✅ OPENROUTER_API_KEY set"; fi

# Testing
test: test-all

test-all:
	@echo "🎯 Running All Tests..."
	python run_tests.py all

test-pytest:
	@echo "🧪 Running Pytest Tests..."
	python run_tests.py pytest

test-standalone:
	@echo "🚀 Running Standalone Tests..."
	python run_tests.py standalone

test-debug:
	@echo "🔧 Debug Scripts Available..."
	python run_tests.py debug

test-discover:
	@echo "🔍 Discovering Available Tests..."
	python run_tests.py discover

test-clean:
	@echo "🧹 Cleaning Test Artifacts..."
	rm -rf .pytest_cache/
	rm -rf tests/__pycache__/
	rm -rf tests/*/__pycache__/
	rm -rf tests/*/*/__pycache__/
	find tests/ -name "*.pyc" -delete
	find tests/ -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

test-check-env:
	@echo "🔍 Checking Test Environment..."
	poetry run python tests/run_all_tests.py --check-env

test-coverage:
	@echo "📊 Running Tests with Coverage..."
	poetry run pytest tests/ --cov=app --cov-report=html:htmlcov --cov-report=term-missing --cov-fail-under=80

test-report:
	@echo "📋 Generating Test Reports..."
	@mkdir -p test_reports
	poetry run pytest tests/ --html=test_reports/report.html --self-contained-html --json-report=test_reports/report.json

# Alternative test runners
test-e2e-alt:
	@echo "🧪 Running E2E Tests (Alternative)..."
	poetry run python tests/e2e/test_e2e_real_llm_performance.py --test-type basic --iterations $(or $(ITERATIONS),1) $(if $(VERBOSE),--verbose)

test-e2e-shell:
	@echo "🧪 Running E2E Tests via Shell Script..."
	@chmod +x tests/utils/run_tests.sh
	./tests/utils/run_tests.sh --test-type basic --iterations $(or $(ITERATIONS),1) $(if $(VERBOSE),--verbose)

# Test configuration management
test-config-show:
	@echo "🔧 Showing Test Configuration..."
	poetry run python -c "from tests.config.test_config import get_test_config, get_api_config; print('Test Config:', get_test_config()); print('API Config:', get_api_config())"

test-config-update:
	@echo "🔧 Updating Test Configuration..."
	@if [ -z "$(KEY)" ] || [ -z "$(VALUE)" ]; then \
		echo "Usage: make test-config-update KEY=timeout VALUE=120"; \
		exit 1; \
	fi
	poetry run python -c "from tests.config.test_config import update_test_config; update_test_config({'$(KEY)': $(VALUE)}); print('Updated $(KEY) to $(VALUE)')"

# Code Quality
lint:
	@echo "🔍 Running linting checks..."
	poetry run flake8 app/ tests/
	poetry run black --check app/ tests/
	poetry run isort --check-only app/ tests/

format:
	@echo "🎨 Formatting code..."
	poetry run black app/ tests/
	poetry run isort app/ tests/

clean:
	@echo "🧹 Cleaning up..."
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf test_reports/
	rm -rf e2e_test_reports/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Quick test commands
quick-test:
	@echo "⚡ Running Quick Test..."
	poetry run python tests/run_all_tests.py --test-type e2e --iterations 1

performance-test:
	@echo "📊 Running Performance Test..."
	poetry run python tests/run_all_tests.py --test-type e2e --iterations 5 --verbose

smoke-test:
	@echo "💨 Running Smoke Tests..."
	poetry run python tests/run_all_tests.py --test-type e2e --iterations 1

regression-test:
	@echo "🔄 Running Regression Tests..."
	poetry run python tests/run_all_tests.py --test-type all --iterations 3

# Development helpers
dev-setup: install check-env
	@echo "✅ Development environment setup complete!"
	@echo "Run 'make run-server' to start the server"
	@echo "Run 'make test-e2e' to run tests"

test-setup: install
	@echo "📦 Installing test dependencies..."
	pip install -r tests/requirements-test.txt
	@echo "✅ Test environment setup complete!"

# Test environment setup
test-env-setup: test-setup
	@echo "🔧 Setting up test environment..."
	@mkdir -p test_reports
	@mkdir -p e2e_test_reports
	@mkdir -p htmlcov
	@echo "✅ Test environment setup complete!"

# CI/CD helpers
ci-test:
	@echo "🚀 Running CI Test Suite..."
	poetry run python tests/run_all_tests.py --test-type all --iterations 1
	poetry run pytest tests/ --cov=app --cov-report=xml --cov-report=term-missing

ci-lint:
	@echo "🔍 Running CI Linting..."
	poetry run flake8 app/ tests/
	poetry run black --check app/ tests/
	poetry run isort --check-only app/ tests/




