#!/usr/bin/env python3
"""
Runner script for Real LLM Performance E2E Tests
This script sets up the environment and runs the comprehensive LLM performance tests.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

async def main():
    """Main runner function"""
    print("🚀 Real LLM Performance E2E Test Runner")
    print("=" * 60)
    
    # Check if required services are running
    print("🔍 Checking environment...")
    
    # Import the test module
    try:
        from test_e2e_real_llm_performance import RealLLMPerformanceTester
        print("✅ Test module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import test module: {e}")
        print("Make sure you're in the correct directory and all dependencies are installed")
        return 1
    
    # Check environment variables
    required_env_vars = [
        "GOOGLE_API_KEY",
        "OPENROUTER_API_KEY", 
        "OPENAI_API_KEY"
    ]
    
    print("\n🔑 Checking API keys...")
    missing_keys = []
    for key in required_env_vars:
        if not os.getenv(key) or os.getenv(key) == f"your_{key.lower()}_here":
            missing_keys.append(key)
            print(f"  ⚠️  {key}: Not configured")
        else:
            print(f"  ✅ {key}: Configured")
    
    if missing_keys:
        print(f"\n⚠️  Warning: {len(missing_keys)} API keys are not configured:")
        for key in missing_keys:
            print(f"    - {key}")
        print("\nTests may fail if these services are required.")
        
        response = input("\nContinue anyway? (y/N): ").strip().lower()
        if response != 'y':
            print("Aborting tests.")
            return 1
    
    # Run the tests
    print("\n🧪 Starting Real LLM Performance Tests...")
    print("This will make actual API calls to LLM services and may incur costs.")
    
    try:
        tester = RealLLMPerformanceTester()
        results = await tester.run_all_tests()
        
        # Print final summary
        passed = sum(1 for result in results if result.success)
        total = len(results)
        
        print(f"\n🎯 Final Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! Your LLM performance is excellent.")
            return 0
        elif passed >= total * 0.8:
            print("⚠️  Most tests passed. Some issues detected but performance is good.")
            return 1
        else:
            print("❌ Multiple tests failed. LLM performance needs attention.")
            return 2
            
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Unexpected error during testing: {e}")
        return 3
    finally:
        if 'tester' in locals():
            await tester.cleanup()

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(4)
