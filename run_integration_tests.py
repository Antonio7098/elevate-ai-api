#!/usr/bin/env python3
"""
Integration Test Runner for Cost Optimization with Real LLM Calls.
Run this script to test the cost optimization system with actual Gemini API calls.
"""

import os
import sys
import asyncio
import pytest
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️  python-dotenv not available, using system environment variables")

def check_environment():
    """Check if required environment variables are set"""
    required_vars = ["GOOGLE_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these variables in your .env file and try again.")
        return False
    
    print("✅ Environment check passed")
    return True

def run_specific_test(test_name):
    """Run a specific integration test"""
    test_file = "tests/test_cost_optimization_integration.py"
    
    if not Path(test_file).exists():
        print(f"❌ Test file not found: {test_file}")
        return False
    
    print(f"🧪 Running specific test: {test_name}")
    
    # Run the specific test
    result = pytest.main([
        test_file,
        f"TestCostOptimizationRealLLM::{test_name}",
        "-v", "-s", "--tb=short"
    ])
    
    return result == 0

def run_all_integration_tests():
    """Run all integration tests"""
    test_file = "tests/test_cost_optimization_integration.py"
    
    if not Path(test_file).exists():
        print(f"❌ Test file not found: {test_file}")
        return False
    
    print("🧪 Running all integration tests...")
    
    # Run all tests in the integration test file
    result = pytest.main([
        test_file,
        "-v", "-s", "--tb=short"
    ])
    
    return result == 0

def run_performance_tests():
    """Run performance tests only"""
    test_file = "tests/test_cost_optimization_integration.py"
    
    if not Path(test_file).exists():
        print(f"❌ Test file not found: {test_file}")
        return False
    
    print("⚡ Running performance tests...")
    
    # Run only performance tests
    result = pytest.main([
        test_file,
        "TestCostOptimizationPerformance",
        "-v", "-s", "--tb=short"
    ])
    
    return result == 0

def show_available_tests():
    """Show available integration tests"""
    print("\n📋 Available Integration Tests:")
    print("=" * 50)
    
    print("\n🔍 Real LLM Tests:")
    print("   - test_real_llm_cost_analysis")
    print("   - test_real_model_cost_comparison")
    print("   - test_real_optimization_recommendations")
    print("   - test_real_budget_compliance")
    print("   - test_real_cost_efficiency_scoring")
    print("   - test_real_end_to_end_workflow")
    
    print("\n⚡ Performance Tests:")
    print("   - test_cost_analysis_performance")
    
    print("\n💡 Usage Examples:")
    print("   python run_integration_tests.py --test test_real_llm_cost_analysis")
    print("   python run_integration_tests.py --performance")
    print("   python run_integration_tests.py --all")

def main():
    """Main function to run integration tests"""
    print("🚀 Cost Optimization Integration Test Runner")
    print("=" * 50)
    
    # Check environment first
    if not check_environment():
        sys.exit(1)
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test" and len(sys.argv) > 2:
            test_name = sys.argv[2]
            success = run_specific_test(test_name)
        elif sys.argv[1] == "--performance":
            success = run_performance_tests()
        elif sys.argv[1] == "--all":
            success = run_all_integration_tests()
        elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
            show_available_tests()
            return
        else:
            print("❌ Invalid arguments. Use --help for usage information.")
            return
    else:
        # Default: show available tests
        show_available_tests()
        return
    
    # Report results
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
