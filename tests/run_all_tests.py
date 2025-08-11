#!/usr/bin/env python3
"""
Main test runner for Elevate AI API
Discovers and runs all tests in the test suite with proper configuration management
"""

import sys
import os
import argparse
import subprocess
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import test configuration
try:
    from tests.config.test_config import get_test_config, get_api_config, update_test_config
except ImportError:
    print("⚠️  Test configuration not found, using defaults")
    get_test_config = lambda: None
    get_api_config = lambda: None
    update_test_config = lambda x: None

class TestRunner:
    """Main test runner class"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.project_root = project_root
        self.test_results = {}
        
    def discover_tests(self) -> Dict[str, List[Path]]:
        """Discover all available tests in the test suite"""
        tests = {
            'unit': [],
            'integration': [],
            'e2e': [],
            'performance': [],
            'contract': []
        }
        
        test_dirs = {
            'unit': self.project_root / 'tests' / 'unit',
            'integration': self.project_root / 'tests' / 'integration',
            'e2e': self.project_root / 'tests' / 'e2e',
            'performance': self.project_root / 'tests' / 'performance',
            'contract': self.project_root / 'tests' / 'contract'
        }
        
        for test_type, test_dir in test_dirs.items():
            if test_dir.exists():
                for test_file in test_dir.glob('test_*.py'):
                    tests[test_type].append(test_file)
                for test_file in test_dir.glob('*_test.py'):
                    tests[test_type].append(test_file)
        
        # Also check for test files in the root tests directory
        root_test_dir = self.project_root / 'tests'
        for test_file in root_test_dir.glob('test_*.py'):
            if test_file.name not in ['conftest.py', 'run_all_tests.py']:
                tests['integration'].append(test_file)
        
        return tests
    
    def print_test_discovery(self, tests: Dict[str, List[Path]]) -> None:
        """Print discovered tests"""
        print("\n🔍 Test Discovery Results:")
        print("=" * 50)
        
        total_tests = 0
        for test_type, test_files in tests.items():
            count = len(test_files)
            total_tests += count
            print(f"{test_type.capitalize()} Tests: {count}")
            
            if self.verbose and test_files:
                for test_file in test_files:
                    print(f"  📄 {test_file.name}")
        
        print(f"\n📊 Total Test Files: {total_tests}")
        print("=" * 50)
    
    def run_pytest_tests(self, test_paths: List[Path], test_type: str) -> bool:
        """Run tests using pytest"""
        if not test_paths:
            print(f"⚠️  No {test_type} tests found")
            return True
        
        print(f"\n🧪 Running {test_type.capitalize()} Tests with pytest...")
        print(f"Found {len(test_paths)} test files")
        
        # Build pytest command
        cmd = [
            sys.executable, '-m', 'pytest',
            '--tb=short',
            '--strict-markers',
            '--disable-warnings'
        ]
        
        if self.verbose:
            cmd.append('-v')
        
        # Add test paths
        for test_path in test_paths:
            cmd.append(str(test_path))
        
        print(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=not self.verbose,
                text=True
            )
            
            if result.returncode == 0:
                print(f"✅ {test_type.capitalize()} tests passed")
                return True
            else:
                print(f"❌ {test_type.capitalize()} tests failed")
                if not self.verbose and result.stdout:
                    print("Output:", result.stdout)
                if result.stderr:
                    print("Errors:", result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ Error running {test_type} tests: {e}")
            return False
    
    def run_e2e_tests(self, test_paths: List[Path], iterations: int = 1) -> bool:
        """Run E2E tests using the specialized runner"""
        if not test_paths:
            print("⚠️  No E2E tests found")
            return True
        
        print(f"\n🚀 Running E2E Tests...")
        print(f"Found {len(test_paths)} E2E test files")
        
        # Use the E2E test runner for each test file
        success = True
        
        for test_file in test_paths:
            print(f"\n📋 Running {test_file.name}...")
            
            # Check if it's a Python test file that can be run directly
            if test_file.suffix == '.py':
                try:
                    # Try to run the test file directly
                    cmd = [
                        sys.executable, str(test_file),
                        '--iterations', str(iterations)
                    ]
                    
                    if self.verbose:
                        cmd.append('--verbose')
                    
                    result = subprocess.run(
                        cmd,
                        cwd=self.project_root,
                        capture_output=not self.verbose,
                        text=True
                    )
                    
                    if result.returncode == 0:
                        print(f"✅ {test_file.name} passed")
                    else:
                        print(f"❌ {test_file.name} failed")
                        success = False
                        
                except Exception as e:
                    print(f"❌ Error running {test_file.name}: {e}")
                    success = False
        
        return success
    
    def run_tests(self, test_type: str = "all", iterations: int = 1) -> bool:
        """Run tests based on type"""
        
        # Discover tests
        tests = self.discover_tests()
        self.print_test_discovery(tests)
        
        # Update configuration if available
        if get_test_config:
            config = get_test_config()
            if config:
                update_test_config({
                    'test_iterations': iterations
                })
                print(f"🔧 Updated test configuration: {iterations} iterations")
        
        if test_type == "e2e":
            return self.run_e2e_tests(tests['e2e'], iterations)
            
        elif test_type == "unit":
            return self.run_pytest_tests(tests['unit'], 'unit')
            
        elif test_type == "integration":
            return self.run_pytest_tests(tests['integration'], 'integration')
            
        elif test_type == "performance":
            return self.run_pytest_tests(tests['performance'], 'performance')
            
        elif test_type == "contract":
            return self.run_pytest_tests(tests['contract'], 'contract')
            
        elif test_type == "all":
            print("\n🎯 Running All Tests...")
            success = True
            
            # Run unit tests
            if tests['unit']:
                if not self.run_pytest_tests(tests['unit'], 'unit'):
                    success = False
            
            # Run integration tests
            if tests['integration']:
                if not self.run_pytest_tests(tests['integration'], 'integration'):
                    success = False
            
            # Run E2E tests
            if tests['e2e']:
                if not self.run_e2e_tests(tests['e2e'], iterations):
                    success = False
            
            # Run performance tests
            if tests['performance']:
                if not self.run_pytest_tests(tests['performance'], 'performance'):
                    success = False
            
            # Run contract tests
            if tests['contract']:
                if not self.run_pytest_tests(tests['contract'], 'contract'):
                    success = False
            
            return success
        
        else:
            print(f"❌ Unknown test type: {test_type}")
            return False
    
    def check_environment(self) -> bool:
        """Check if the test environment is properly set up"""
        print("\n🔍 Checking Test Environment...")
        
        # Check if we're in the right directory
        if not (self.project_root / 'tests').exists():
            print("❌ Tests directory not found")
            return False
        
        # Check if pytest is available
        try:
            import pytest
            print("✅ pytest available")
        except ImportError:
            print("❌ pytest not available")
            return False
        
        # Check if test configuration is available
        if get_test_config:
            try:
                config = get_test_config()
                if config:
                    print("✅ Test configuration loaded")
                else:
                    print("⚠️  Test configuration not available")
            except Exception as e:
                print(f"⚠️  Test configuration error: {e}")
        
        # Check if server is running (optional)
        if get_api_config:
            try:
                api_config = get_api_config()
                if api_config:
                    print(f"✅ API configuration: {api_config.ai_api_base_url}")
            except Exception as e:
                print(f"⚠️  API configuration error: {e}")
        
        print("✅ Environment check complete")
        return True

def main():
    parser = argparse.ArgumentParser(description="Run Elevate AI API tests")
    parser.add_argument(
        "--test-type", 
        choices=["all", "e2e", "unit", "integration", "performance", "contract"], 
        default="all", 
        help="Type of tests to run"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Verbose output"
    )
    parser.add_argument(
        "--iterations", "-i", 
        type=int, 
        default=1, 
        help="Number of test iterations for E2E tests"
    )
    parser.add_argument(
        "--discover-only", 
        action="store_true", 
        help="Only discover tests, don't run them"
    )
    parser.add_argument(
        "--check-env", 
        action="store_true", 
        help="Check test environment only"
    )
    
    args = parser.parse_args()
    
    print("🧪 Elevate AI API Test Suite")
    print("=" * 50)
    
    # Create test runner
    runner = TestRunner(verbose=args.verbose)
    
    # Check environment if requested
    if args.check_env:
        runner.check_environment()
        return
    
    # Check environment
    if not runner.check_environment():
        print("❌ Environment check failed!")
        sys.exit(1)
    
    # Discover tests
    tests = runner.discover_tests()
    runner.print_test_discovery(tests)
    
    # Exit if only discovering
    if args.discover_only:
        return
    
    # Run tests
    success = runner.run_tests(args.test_type, args.iterations)
    
    if success:
        print("\n✅ All tests completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
