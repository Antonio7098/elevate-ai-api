#!/usr/bin/env python3
"""
Simple runner script for E2E LLM performance tests.
Provides easy execution with different configurations and options.
"""

import asyncio
import sys
import argparse
import os
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from test_e2e_real_llm_performance import RealLLMPerformanceTester, REAL_LLM_CONFIG

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="E2E LLM Performance Test Runner")
    
    parser.add_argument(
        "--test-type",
        choices=["all", "chat", "cascade", "concurrent", "cost-optimization", "core-integration"],
        default="all",
        help="Type of test to run"
    )
    
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of test iterations per query"
    )
    
    parser.add_argument(
        "--concurrent-requests",
        type=int,
        default=5,
        help="Number of concurrent requests for load testing"
    )
    
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Timeout for API calls in seconds"
    )
    
    parser.add_argument(
        "--cost-threshold",
        type=float,
        default=1.0,
        help="Maximum cost threshold in dollars"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save test results to file"
    )
    
    parser.add_argument(
        "--results-file",
        type=str,
        default="e2e_test_results.json",
        help="File to save test results"
    )
    
    return parser.parse_args()

def update_config(args):
    """Update test configuration based on arguments"""
    REAL_LLM_CONFIG.update({
        "test_iterations": args.iterations,
        "concurrent_requests": args.concurrent_requests,
        "timeout": args.timeout,
        "cost_threshold": args.cost_threshold
    })
    
    if args.verbose:
        print(f"🔧 Updated configuration: {REAL_LLM_CONFIG}")

async def run_specific_test(tester: RealLLMPerformanceTester, test_type: str):
    """Run a specific test type"""
    if test_type == "chat":
        return [await tester.test_real_llm_chat_performance()]
    elif test_type == "cascade":
        return [await tester.test_model_cascading_performance()]
    elif test_type == "concurrent":
        return [await tester.test_concurrent_llm_performance()]
    elif test_type == "cost-optimization":
        return [await tester.test_cost_optimization_workflow()]
    elif test_type == "core-integration":
        return [await tester.test_core_api_integration()]
    else:
        return await tester.run_all_tests()

async def main():
    """Main runner function"""
    args = parse_arguments()
    
    print("🚀 E2E LLM Performance Test Runner")
    print("=" * 50)
    
    # Update configuration
    update_config(args)
    
    # Create tester instance
    tester = RealLLMPerformanceTester()
    
    try:
        # Validate environment
        if not await tester.validate_environment():
            print("❌ Environment validation failed. Please check your services.")
            sys.exit(1)
        
        # Run tests
        print(f"\n🧪 Running {args.test_type} tests...")
        results = await run_specific_test(tester, args.test_type)
        
        # Print summary
        await tester.print_detailed_summary(results)
        
        # Save results if requested
        if args.save_results:
            import json
            from datetime import datetime
            
            results_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "configuration": REAL_LLM_CONFIG,
                "results": [
                    {
                        "test_name": result.test_name,
                        "success": result.success,
                        "success_rate": result.success_rate,
                        "avg_response_time": result.avg_response_time,
                        "total_cost": result.total_cost,
                        "error_details": result.error_details
                    }
                    for result in results
                ]
            }
            
            with open(args.results_file, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            print(f"\n💾 Results saved to: {args.results_file}")
        
        # Exit with appropriate code
        passed = sum(1 for result in results if result.success)
        total = len(results)
        
        if passed == total:
            print("\n🎉 All tests passed!")
            sys.exit(0)
        elif passed >= total * 0.8:
            print("\n⚠️  Most tests passed. Some issues detected.")
            sys.exit(1)
        else:
            print("\n❌ Multiple tests failed. Performance needs attention.")
            sys.exit(2)
            
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(3)
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    asyncio.run(main())




