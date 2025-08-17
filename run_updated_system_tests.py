#!/usr/bin/env python3
"""
Simple test runner for the updated note editing system.
Run this script to test all the new functionality with real LLM calls.
"""

import asyncio
import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from test_updated_note_editing_system import NoteEditingSystemTester


async def main():
    """Run the comprehensive test suite."""
    print("🚀 Starting Updated Note Editing System Tests")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists('app'):
        print("❌ Error: Please run this script from the elevate-ai-api directory")
        print("   Current directory:", os.getcwd())
        print("   Expected to find 'app' directory")
        return
    
    # Initialize and run tests
    tester = NoteEditingSystemTester()
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










