#!/usr/bin/env python3
"""
Simple test runner script for Elevate AI API
Provides easy access to the test suite with common options
"""

import sys
import argparse
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    parser = argparse.ArgumentParser(
        description="Elevate AI API Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  python tests/run_tests.py
  
  # Run only E2E tests
  python tests/run_tests.py --e2e
  
  # Run tests with coverage
  python tests/run_tests.py --coverage
  
  # Run specific test type
  python tests/run_tests.py --type unit
  
  # Run with custom iterations
  python tests/run_tests.py --e2e --iterations 5
        """
    )
    
    parser.add_argument(
        '--e2e', 
        action='store_true',
        help='Run E2E tests only'
    )
    
    parser.add_argument(
        '--unit', 
        action='store_true',
        help='Run unit tests only'
    )
    
    parser.add_argument(
        '--integration', 
        action='store_true',
        help='Run integration tests only'
    )
    
    parser.add_argument(
        '--performance', 
        action='store_true',
        help='Run performance tests only'
    )
    
    parser.add_argument(
        '--type', 
        choices=['all', 'e2e', 'unit', 'integration', 'performance', 'contract'],
        default='all',
        help='Type of tests to run'
    )
    
    parser.add_argument(
        '--iterations', '-i',
        type=int,
        default=1,
        help='Number of test iterations for E2E tests'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--coverage',
        action='store_true',
        help='Run tests with coverage reporting'
    )
    
    parser.add_argument(
        '--discover',
        action='store_true',
        help='Only discover tests, don\'t run them'
    )
    
    parser.add_argument(
        '--check-env',
        action='store_true',
        help='Check test environment only'
    )
    
    args = parser.parse_args()
    
    # Determine test type based on flags
    if args.e2e:
        test_type = 'e2e'
    elif args.unit:
        test_type = 'unit'
    elif args.integration:
        test_type = 'integration'
    elif args.performance:
        test_type = 'performance'
    else:
        test_type = args.type
    
    # Build command for the main test runner
    cmd = [
        sys.executable, 
        str(project_root / 'tests' / 'run_all_tests.py'),
        '--test-type', test_type
    ]
    
    if args.iterations > 1:
        cmd.extend(['--iterations', str(args.iterations)])
    
    if args.verbose:
        cmd.append('--verbose')
    
    if args.discover:
        cmd.append('--discover-only')
    
    if args.check_env:
        cmd.append('--check-env')
    
    # Run the command
    print(f"🚀 Running: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        import subprocess
        result = subprocess.run(cmd, cwd=project_root)
        
        if args.coverage and result.returncode == 0:
            print("\n📊 Running coverage report...")
            coverage_cmd = [
                sys.executable, '-m', 'pytest',
                'tests/',
                '--cov=app',
                '--cov-report=html:htmlcov',
                '--cov-report=term-missing'
            ]
            subprocess.run(coverage_cmd, cwd=project_root)
        
        sys.exit(result.returncode)
        
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error running tests: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

