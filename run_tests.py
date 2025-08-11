#!/usr/bin/env python3
"""
Simple test runner for Elevate AI API
Organizes and runs tests from the organized test structure
"""
import os
import sys
import subprocess
from pathlib import Path

def discover_tests():
    """Discover all test files in the organized structure"""
    test_dirs = {
        'pytest': 'tests/pytest',
        'standalone': 'tests/standalone',
        'debug': 'tests/debug',
        'unit': 'tests/unit'
    }
    
    results = {}
    for test_type, test_dir in test_dirs.items():
        if os.path.exists(test_dir):
            test_files = []
            for file in os.listdir(test_dir):
                if file.endswith('.py') and (file.startswith('test_') or file.startswith('test-')):
                    test_files.append(os.path.join(test_dir, file))
            results[test_type] = test_files
    
    return results

def run_pytest_tests(test_type=None, verbose=False):
    """Run pytest-compatible tests"""
    if test_type and test_type == 'pytest':
        test_path = "tests/pytest"
    else:
        test_path = "tests/pytest"
    
    if not os.path.exists(test_path):
        print(f"❌ Test directory {test_path} not found")
        return False
    
    cmd = ['python', '-m', 'pytest', test_path]
    if verbose:
        cmd.append('-v')
    
    print(f"🚀 Running pytest tests: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0

def run_standalone_tests(test_type=None, verbose=False):
    """Run standalone test scripts"""
    if test_type and test_type == 'standalone':
        test_path = "tests/standalone"
    else:
        test_path = "tests/standalone"
    
    if not os.path.exists(test_path):
        print(f"❌ Test directory {test_path} not found")
        return False
    
    print(f"🚀 Running standalone tests from: {test_path}")
    print("Note: Standalone tests need to be run individually as they are not pytest-compatible")
    
    # List available standalone tests
    test_files = [f for f in os.listdir(test_path) if f.endswith('.py') and f.startswith('test_')]
    print(f"Available standalone tests ({len(test_files)}):")
    for test_file in test_files:
        print(f"  - {test_file}")
    
    return True

def run_tests(test_type=None, verbose=False):
    """Run tests of a specific type or all tests"""
    if test_type == 'pytest':
        return run_pytest_tests(test_type, verbose)
    elif test_type == 'standalone':
        return run_standalone_tests(test_type, verbose)
    elif test_type == 'debug':
        print("🔧 Debug scripts available in tests/debug/")
        return True
    else:
        # Run all test types
        print("🎯 Running all test types...")
        pytest_success = run_pytest_tests(None, verbose)
        standalone_success = run_standalone_tests(None, verbose)
        return pytest_success and standalone_success

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_tests.py [test_type] [--verbose]")
        print("\nAvailable test types:")
        tests = discover_tests()
        for test_type, test_files in tests.items():
            print(f"  {test_type}: {len(test_files)} files")
        print("\nExamples:")
        print("  python run_tests.py pytest")
        print("  python run_tests.py standalone")
        print("  python run_tests.py all")
        return
    
    test_type = sys.argv[1]
    verbose = '--verbose' in sys.argv
    
    if test_type == 'discover':
        print("🔍 Test Discovery Results:")
        tests = discover_tests()
        for test_type, test_files in tests.items():
            print(f"  {test_type}: {len(test_files)} files")
            for test_file in test_files[:5]:  # Show first 5 files
                print(f"    - {test_file}")
            if len(test_files) > 5:
                print(f"    ... and {len(test_files) - 5} more")
        return
    
    if test_type == 'all':
        test_type = None
    
    success = run_tests(test_type, verbose)
    if success:
        print("✅ Tests completed successfully!")
    else:
        print("❌ Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
