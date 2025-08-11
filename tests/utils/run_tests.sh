#!/bin/bash

# E2E Test Runner Script
# Provides easy execution of E2E tests with common configurations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
TEST_TYPE="all"
ITERATIONS=3
CONCURRENT_REQUESTS=5
TIMEOUT=60
COST_THRESHOLD=1.0
VERBOSE=false
SAVE_RESULTS=false
RESULTS_FILE="e2e_test_results.json"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "E2E Test Runner Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -t, --test-type TYPE       Test type: all, chat, cascade, concurrent, cost-optimization, core-integration"
    echo "  -i, --iterations NUM       Number of test iterations (default: 3)"
    echo "  -c, --concurrent NUM       Number of concurrent requests (default: 5)"
    echo "  --timeout SECONDS          Timeout for API calls (default: 60)"
    echo "  --cost-threshold DOLLARS   Maximum cost threshold (default: 1.0)"
    echo "  -v, --verbose              Enable verbose output"
    echo "  -s, --save-results         Save test results to file"
    echo "  --results-file FILE        Results file name (default: e2e_test_results.json)"
    echo "  -h, --help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Run all tests with defaults"
    echo "  $0 -t chat                           # Run only chat tests"
    echo "  $0 -i 5 -c 10                       # 5 iterations, 10 concurrent requests"
    echo "  $0 --timeout 120 --cost-threshold 2.0 # Custom timeout and cost threshold"
    echo "  $0 -v -s                             # Verbose output and save results"
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check if required files exist
    if [[ ! -f "test_e2e_real_llm_performance.py" ]]; then
        print_error "test_e2e_real_llm_performance.py not found in current directory"
        exit 1
    fi
    
    if [[ ! -f "run_e2e_tests.py" ]]; then
        print_error "run_e2e_tests.py not found in current directory"
        exit 1
    fi
    
    # Check if services are accessible
    print_status "Checking service availability..."
    
    # Check AI API
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        print_success "AI API is accessible"
    else
        print_warning "AI API is not accessible at http://localhost:8000/health"
        print_warning "Make sure the AI API is running on port 8000"
    fi
    
    # Check Core API
    if curl -s http://localhost:3000/health > /dev/null 2>&1; then
        print_success "Core API is accessible"
    else
        print_warning "Core API is not accessible at http://localhost:3000/health"
        print_warning "Make sure the Core API is running on port 3000"
    fi
    
    print_success "Prerequisites check completed"
}

# Function to run tests
run_tests() {
    print_status "Starting E2E tests..."
    
    # Build command
    CMD="python3 run_e2e_tests.py"
    CMD="$CMD --test-type $TEST_TYPE"
    CMD="$CMD --iterations $ITERATIONS"
    CMD="$CMD --concurrent-requests $CONCURRENT_REQUESTS"
    CMD="$CMD --timeout $TIMEOUT"
    CMD="$CMD --cost-threshold $COST_THRESHOLD"
    
    if [[ "$VERBOSE" == true ]]; then
        CMD="$CMD --verbose"
    fi
    
    if [[ "$SAVE_RESULTS" == true ]]; then
        CMD="$CMD --save-results"
        CMD="$CMD --results-file $RESULTS_FILE"
    fi
    
    print_status "Executing: $CMD"
    echo ""
    
    # Execute command
    eval $CMD
    
    # Check exit code
    EXIT_CODE=$?
    if [[ $EXIT_CODE -eq 0 ]]; then
        print_success "All tests passed!"
    elif [[ $EXIT_CODE -eq 1 ]]; then
        print_warning "Most tests passed, some issues detected"
    else
        print_error "Multiple tests failed"
    fi
    
    return $EXIT_CODE
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--test-type)
            TEST_TYPE="$2"
            shift 2
            ;;
        -i|--iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        -c|--concurrent)
            CONCURRENT_REQUESTS="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --cost-threshold)
            COST_THRESHOLD="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -s|--save-results)
            SAVE_RESULTS=true
            shift
            ;;
        --results-file)
            RESULTS_FILE="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate test type
VALID_TEST_TYPES=("all" "chat" "cascade" "concurrent" "cost-optimization" "core-integration")
if [[ ! " ${VALID_TEST_TYPES[@]} " =~ " ${TEST_TYPE} " ]]; then
    print_error "Invalid test type: $TEST_TYPE"
    print_error "Valid types: ${VALID_TEST_TYPES[*]}"
    exit 1
fi

# Validate numeric parameters
if ! [[ "$ITERATIONS" =~ ^[0-9]+$ ]] || [[ "$ITERATIONS" -lt 1 ]]; then
    print_error "Iterations must be a positive integer"
    exit 1
fi

if ! [[ "$CONCURRENT_REQUESTS" =~ ^[0-9]+$ ]] || [[ "$CONCURRENT_REQUESTS" -lt 1 ]]; then
    print_error "Concurrent requests must be a positive integer"
    exit 1
fi

if ! [[ "$TIMEOUT" =~ ^[0-9]+(\.[0-9]+)?$ ]] || (( $(echo "$TIMEOUT <= 0" | bc -l) )); then
    print_error "Timeout must be a positive number"
    exit 1
fi

if ! [[ "$COST_THRESHOLD" =~ ^[0-9]+(\.[0-9]+)?$ ]] || (( $(echo "$COST_THRESHOLD <= 0" | bc -l) )); then
    print_error "Cost threshold must be a positive number"
    exit 1
fi

# Main execution
echo "🚀 E2E Test Runner Script"
echo "=========================="
echo "Test Type: $TEST_TYPE"
echo "Iterations: $ITERATIONS"
echo "Concurrent Requests: $CONCURRENT_REQUESTS"
echo "Timeout: ${TIMEOUT}s"
echo "Cost Threshold: $${COST_THRESHOLD}"
echo "Verbose: $VERBOSE"
echo "Save Results: $SAVE_RESULTS"
if [[ "$SAVE_RESULTS" == true ]]; then
    echo "Results File: $RESULTS_FILE"
fi
echo ""

# Check prerequisites
check_prerequisites

# Run tests
run_tests
EXIT_CODE=$?

echo ""
if [[ $EXIT_CODE -eq 0 ]]; then
    print_success "Test execution completed successfully"
else
    print_warning "Test execution completed with issues (exit code: $EXIT_CODE)"
fi

exit $EXIT_CODE




