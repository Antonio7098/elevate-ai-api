#!/usr/bin/env python3
"""
Test runner for the Granular Editing System.
Run this script to test all the new granular editing capabilities with real LLM calls.
"""

import asyncio
import sys
import os

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️  python-dotenv not available, using system environment variables")

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from test_granular_editing_system import GranularEditingSystemTester


async def main():
    """Run the comprehensive granular editing test suite."""
    print("🚀 Starting Granular Editing System Tests")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists('app'):
        print("❌ Error: Please run this script from the elevate-ai-api directory")
        print("   Current directory:", os.getcwd())
        print("   Expected to find 'app' directory")
        return
    
    # Initialize and run tests
    tester = GranularEditingSystemTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()



