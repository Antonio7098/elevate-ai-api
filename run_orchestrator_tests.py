#!/usr/bin/env python3
"""
Test runner for the Generation Orchestrator

This script runs all the tests for the sequential generation workflow
and provides a summary of the results.
"""

import subprocess
import sys
import os

def run_tests():
    """Run the orchestrator tests and display results."""
    print("🧪 Running Generation Orchestrator Tests")
    print("=" * 50)
    
    # Run the orchestrator tests
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        "tests/test_generation_orchestrator.py", 
        "-v", "--tb=short"
    ], capture_output=True, text=True)
    
    # Display results
    if result.returncode == 0:
        print("✅ All orchestrator tests passed!")
        print("\n📊 Test Summary:")
        # Extract test summary from output
        lines = result.stdout.split('\n')
        for line in lines:
            if "passed" in line and "warnings" in line:
                print(f"   {line.strip()}")
                break
    else:
        print("❌ Some tests failed!")
        print("\n🔍 Test Output:")
        print(result.stdout)
        print("\n❌ Error Output:")
        print(result.stderr)
    
    print("\n" + "=" * 50)
    return result.returncode == 0

def run_endpoint_tests():
    """Run the endpoint tests if they exist."""
    endpoint_test_file = "tests/test_orchestrator_endpoints.py"
    
    if not os.path.exists(endpoint_test_file):
        print("⚠️  Endpoint tests not found, skipping...")
        return True
    
    print("\n🌐 Running Orchestrator Endpoint Tests")
    print("=" * 50)
    
    # Run the endpoint tests
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        endpoint_test_file, 
        "-v", "--tb=short"
    ], capture_output=True, text=True)
    
    # Display results
    if result.returncode == 0:
        print("✅ All endpoint tests passed!")
        print("\n📊 Test Summary:")
        # Extract test summary from output
        lines = result.stdout.split('\n')
        for line in lines:
            if "passed" in line and "warnings" in line:
                print(f"   {line.strip()}")
                break
    else:
        print("❌ Some endpoint tests failed!")
        print("\n🔍 Test Output:")
        print(result.stdout)
        print("\n❌ Error Output:")
        print(result.stderr)
    
    print("\n" + "=" * 50)
    return result.returncode == 0

def main():
    """Main test runner function."""
    print("🚀 Elevate AI API - Sequential Generation Workflow Test Suite")
    print("=" * 60)
    
    # Run orchestrator tests
    orchestrator_success = run_tests()
    
    # Run endpoint tests
    endpoint_success = run_endpoint_tests()
    
    # Final summary
    print("\n🎯 Final Test Results")
    print("=" * 30)
    print(f"Orchestrator Tests: {'✅ PASSED' if orchestrator_success else '❌ FAILED'}")
    print(f"Endpoint Tests:     {'✅ PASSED' if endpoint_success else '❌ FAILED'}")
    
    if orchestrator_success and endpoint_success:
        print("\n🎉 All tests passed! The sequential generation workflow is working correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())



