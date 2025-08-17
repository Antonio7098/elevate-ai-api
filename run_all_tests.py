#!/usr/bin/env python3
"""
Main Test Runner for All Services with REAL API calls.
Allows running all tests or individual modules in a modular fashion.
"""

import asyncio
import sys
import time
from typing import List, Dict, Any

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded")
except ImportError:
    print("⚠️  python-dotenv not available")


class MasterTestRunner:
    """Master test runner that orchestrates all test modules."""
    
    def __init__(self):
        self.test_modules = {
            "llm": {
                "name": "LLM Services",
                "file": "tests/services/test_llm_services.py",
                "description": "Tests Google Gemini, OpenRouter, and Mock LLM services"
            },
            "vector": {
                "name": "Vector Stores",
                "file": "tests/services/test_vector_stores.py",
                "description": "Tests Pinecone and ChromaDB vector stores"
            },
            "rag": {
                "name": "RAG & Search",
                "file": "tests/services/test_rag_search.py",
                "description": "Tests RAG, search, and knowledge retrieval services"
            },
            "notes": {
                "name": "Note Services",
                "file": "tests/services/test_note_services.py",
                "description": "Tests note editing, granular editing, and orchestration"
            },
            "premium": {
                "name": "Premium Agents",
                "file": "tests/services/test_premium_agents.py",
                "description": "Tests content curation, explanation, and context agents"
            },
            "api": {
                "name": "API Endpoints",
                "file": "tests/api/test_api_endpoints.py",
                "description": "Tests note creation, editing, and search endpoints"
            }
        }
        self.test_results = {}
        
    def print_help(self):
        """Print help information."""
        print("\n🚀 Elevate AI API - Master Test Runner")
        print("=" * 60)
        print("Run all tests or individual modules with REAL API calls!")
        print("\n📋 Available Test Modules:")
        print("-" * 40)
        
        for key, module in self.test_modules.items():
            print(f"  {key:8} - {module['name']}")
            print(f"           {module['description']}")
            print()
        
        print("🔧 Usage:")
        print("  python run_all_tests.py                    # Run all tests")
        print("  python run_all_tests.py llm               # Run only LLM services")
        print("  python run_all_tests.py vector rag        # Run vector stores and RAG")
        print("  python run_all_tests.py --help            # Show this help")
        print("\n⚠️  Make sure your .env file has the required API keys:")
        print("   - GOOGLE_API_KEY")
        print("   - PINECONE_API_KEY")
        print("   - OPENROUTER_API_KEY")
        print()
    
    async def run_test_module(self, module_key: str) -> bool:
        """Run a single test module."""
        if module_key not in self.test_modules:
            print(f"❌ Unknown test module: {module_key}")
            return False
        
        module = self.test_modules[module_key]
        print(f"\n🚀 Running {module['name']} Tests")
        print("=" * 60)
        
        try:
            # Import and run the test module
            module_name = module['file'].replace('/', '.').replace('.py', '')
            print(f"📁 Module: {module_name}")
            print(f"📝 Description: {module['description']}")
            print()
            
            # Execute the test module
            start_time = time.time()
            
            # Use subprocess to run the test module
            import subprocess
            result = subprocess.run([
                sys.executable, module['file']
            ], capture_output=True, text=True)
            
            end_time = time.time()
            
            # Print output
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print("⚠️  Warnings/Errors:")
                print(result.stderr)
            
            success = result.returncode == 0
            self.test_results[module_key] = {
                "success": success,
                "duration": end_time - start_time,
                "return_code": result.returncode
            }
            
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"\n🎯 {module['name']}: {status} ({end_time - start_time:.2f}s)")
            
            return success
            
        except Exception as e:
            print(f"❌ Failed to run {module['name']}: {e}")
            self.test_results[module_key] = {
                "success": False,
                "duration": 0,
                "return_code": -1,
                "error": str(e)
            }
            return False
    
    async def run_all_tests(self) -> bool:
        """Run all test modules."""
        print("🚀 Running ALL Test Modules")
        print("=" * 60)
        
        all_success = True
        total_start_time = time.time()
        
        for module_key in self.test_modules.keys():
            success = await self.run_test_module(module_key)
            if not success:
                all_success = False
            
            # Add a small delay between modules
            await asyncio.sleep(1)
        
        total_end_time = time.time()
        
        # Print final summary
        self.print_final_summary(total_end_time - total_start_time)
        
        return all_success
    
    def print_final_summary(self, total_duration: float):
        """Print final test summary."""
        print("\n" + "=" * 60)
        print("📊 FINAL TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for result in self.test_results.values() if result["success"])
        total = len(self.test_results)
        
        print(f"⏱️  Total Duration: {total_duration:.2f}s")
        print(f"🎯 Overall Result: {passed}/{total} modules passed ({passed/total*100:.1f}%)")
        print()
        
        print("📋 Module Results:")
        print("-" * 40)
        
        for module_key, result in self.test_results.items():
            module_name = self.test_modules[module_key]["name"]
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            duration = f"{result['duration']:.2f}s"
            
            if result["success"]:
                print(f"   {status} {module_name} ({duration})")
            else:
                error_info = f" - {result.get('error', f'Return code: {result.get('return_code', 'Unknown')}')}"
                print(f"   {status} {module_name} ({duration}){error_info}")
        
        print()
        
        if passed == total:
            print("🎉 All test modules passed successfully!")
        else:
            print(f"⚠️  {total - passed} test module(s) failed. Check the output above for details.")
    
    async def run_specific_modules(self, module_keys: List[str]) -> bool:
        """Run specific test modules."""
        print(f"🚀 Running Specific Test Modules: {', '.join(module_keys)}")
        print("=" * 60)
        
        all_success = True
        total_start_time = time.time()
        
        for module_key in module_keys:
            if module_key in self.test_modules:
                success = await self.run_test_module(module_key)
                if not success:
                    all_success = False
            else:
                print(f"❌ Unknown test module: {module_key}")
                all_success = False
        
        total_end_time = time.time()
        
        # Print summary for selected modules
        if len(module_keys) > 1:
            self.print_final_summary(total_end_time - total_start_time)
        
        return all_success


async def main():
    """Main entry point."""
    runner = MasterTestRunner()
    
    # Parse command line arguments
    args = sys.argv[1:]
    
    if not args or "--help" in args or "-h" in args:
        runner.print_help()
        return
    
    if "all" in args:
        await runner.run_all_tests()
    else:
        await runner.run_specific_modules(args)


if __name__ == "__main__":
    asyncio.run(main())
