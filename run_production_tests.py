#!/usr/bin/env python3
"""
Master Production Readiness Test Runner.
Orchestrates all test modules and generates comprehensive reports.
"""

import asyncio
import os
import sys
import time
import subprocess
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ProductionTestRunner:
    def __init__(self):
        self.test_modules = {
            "blueprint": {
                "name": "Blueprint Services",
                "file": "test_blueprint_services.py",
                "description": "Primitive generation, section creation, parsing, lifecycle"
            },
            "mastery": {
                "name": "Mastery & Learning Services", 
                "file": "test_mastery_services.py",
                "description": "Criteria generation, question generation, mapping"
            },
            "notes": {
                "name": "Note Services",
                "file": "test_note_services.py", 
                "description": "Note generation, editing, granular editing"
            },
            "rag": {
                "name": "RAG & Search Services",
                "file": "test_rag_services.py",
                "description": "RAG engine, GraphRAG, search, knowledge retrieval"
            },
            "chat": {
                "name": "Chat & Interaction Services",
                "file": "test_chat_services.py",
                "description": "Chat, context assembly, response generation"
            },
            "integration": {
                "name": "Integration Workflows",
                "file": "test_integration_workflows.py",
                "description": "Complete workflows, agent orchestration"
            },
            "vector": {
                "name": "Vector Store Operations",
                "file": "test_vector_store_operations.py",
                "description": "Blueprint/primitive/criterion operations for RAG and GraphRAG"
            }
        }
        
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    def print_banner(self):
        """Print the production test banner."""
        print("=" * 100)
        print("🚀 ELEVATE AI API - PRODUCTION READINESS TEST SUITE")
        print("=" * 100)
        print("🧪 Testing all services with REAL LLM calls for production validation")
        print("🔑 Requires: GOOGLE_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT")
        print("=" * 100)
        print()
    
    def print_help(self):
        """Print help information."""
        print("📚 PRODUCTION TEST SUITE HELP")
        print("-" * 60)
        print("Available test modules:")
        print()
        
        for key, module in self.test_modules.items():
            print(f"  {key:12} - {module['name']}")
            print(f"              {module['description']}")
            print()
        
        print("Usage examples:")
        print("  python run_production_tests.py                    # Run all tests")
        print("  python run_production_tests.py blueprint         # Run blueprint tests only")
        print("  python run_production_tests.py mastery notes     # Run mastery and note tests")
        print("  python run_production_tests.py --help            # Show this help")
        print()
        print("Test execution:")
        print("  - Each module runs independently with real LLM calls")
        print("  - Tests use Gemini 2.5 Flash for optimal performance")
        print("  - Vector stores use Pinecone (production) or ChromaDB (fallback)")
        print("  - All services are tested with real dependencies")
        print()
    
    async def run_test_module(self, module_key: str) -> bool:
        """Run a single test module."""
        if module_key not in self.test_modules:
            print(f"❌ Unknown test module: {module_key}")
            return False
        
        module = self.test_modules[module_key]
        print(f"\n🧪 Running: {module['name']}")
        print(f"📄 File: {module['file']}")
        print(f"📝 Description: {module['description']}")
        print("-" * 80)
        
        try:
            # Run the test module as a subprocess
            result = subprocess.run([
                sys.executable, module['file']
            ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
            
            if result.returncode == 0:
                print(f"✅ {module['name']} - PASSED")
                self.results[module_key] = {
                    "status": "PASSED",
                    "return_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
                return True
            else:
                print(f"❌ {module['name']} - FAILED (exit code: {result.returncode})")
                if result.stderr:
                    print(f"   Error output: {result.stderr[:200]}...")
                self.results[module_key] = {
                    "status": "FAILED",
                    "return_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {module['name']} - TIMEOUT (exceeded 5 minutes)")
            self.results[module_key] = {
                "status": "TIMEOUT",
                "return_code": -1,
                "stdout": "",
                "stderr": "Test execution timed out"
            }
            return False
        except Exception as e:
            print(f"💥 {module['name']} - ERROR: {e}")
            self.results[module_key] = {
                "status": "ERROR",
                "return_code": -1,
                "stdout": "",
                "stderr": str(e)
            }
            return False
    
    async def run_all_tests(self) -> bool:
        """Run all test modules."""
        print("🚀 Starting comprehensive production readiness test suite...")
        print()
        
        self.start_time = time.time()
        
        # Check environment variables
        print("🔍 Checking environment configuration...")
        google_key = os.getenv("GOOGLE_API_KEY")
        pinecone_key = os.getenv("PINECONE_API_KEY")
        pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
        
        if not google_key:
            print("⚠️  Warning: GOOGLE_API_KEY not found - some tests may fail")
        else:
            print("✅ GOOGLE_API_KEY found")
            
        if not pinecone_key or not pinecone_env:
            print("⚠️  Warning: Pinecone credentials not found - will fallback to ChromaDB")
        else:
            print("✅ Pinecone credentials found")
        
        print()
        
        # Run all test modules
        results = []
        for module_key in self.test_modules.keys():
            result = await self.run_test_module(module_key)
            results.append((module_key, result))
        
        self.end_time = time.time()
        
        # Generate summary
        await self.print_final_summary(results)
        
        return all(result for _, result in results)
    
    async def run_specific_modules(self, module_keys: list) -> bool:
        """Run specific test modules."""
        print(f"🎯 Running specific test modules: {', '.join(module_keys)}")
        print()
        
        self.start_time = time.time()
        
        # Validate module keys
        valid_keys = [key for key in module_keys if key in self.test_modules]
        invalid_keys = [key for key in module_keys if key not in self.test_modules]
        
        if invalid_keys:
            print(f"❌ Invalid module keys: {', '.join(invalid_keys)}")
            print("   Use --help to see available modules")
            return False
        
        # Run specified modules
        results = []
        for module_key in valid_keys:
            result = await self.run_test_module(module_key)
            results.append((module_key, result))
        
        self.end_time = time.time()
        
        # Generate summary
        await self.print_final_summary(results)
        
        return all(result for _, result in results)
    
    async def print_final_summary(self, results: list):
        """Print the final test summary."""
        print("\n" + "=" * 100)
        print("📊 PRODUCTION READINESS TEST SUITE - FINAL SUMMARY")
        print("=" * 100)
        
        # Calculate statistics
        total_tests = len(results)
        passed_tests = sum(1 for _, result in results if result)
        failed_tests = total_tests - passed_tests
        
        # Calculate duration
        duration = self.end_time - self.start_time if self.start_time and self.end_time else 0
        
        # Print overall results
        print(f"🎯 Overall Results: {passed_tests}/{total_tests} test modules passed")
        print(f"⏱️  Total Duration: {duration:.2f} seconds")
        print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Print detailed results
        print("📋 Detailed Results:")
        print("-" * 80)
        
        for module_key, result in results:
            module = self.test_modules[module_key]
            status_icon = "✅" if result else "❌"
            status_text = "PASSED" if result else "FAILED"
            
            print(f"{status_icon} {module['name']:<30} - {status_text}")
            
            if not result and module_key in self.results:
                result_info = self.results[module_key]
                if result_info.get("stderr"):
                    print(f"    Error: {result_info['stderr'][:100]}...")
        
        print()
        
        # Print recommendations
        if failed_tests == 0:
            print("🎉 ALL TESTS PASSED! Your AI API is production-ready!")
            print("   - All services are functioning correctly")
            print("   - LLM integration is working")
            print("   - Vector stores are operational")
            print("   - Agent orchestration is functional")
        else:
            print("⚠️  SOME TESTS FAILED - Review the output above")
            print("   - Check error messages for specific issues")
            print("   - Verify environment variables are set correctly")
            print("   - Ensure all dependencies are installed")
            print("   - Review service configurations")
        
        print()
        print("📚 Next Steps:")
        print("   - Review individual test outputs for detailed information")
        print("   - Fix any failing tests before production deployment")
        print("   - Run specific test modules to isolate issues")
        print("   - Check logs and error messages for debugging")
        
        print("=" * 100)
    
    def generate_report_file(self):
        """Generate a detailed report file."""
        if not self.results:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"production_test_report_{timestamp}.txt"
        
        try:
            with open(report_file, 'w') as f:
                f.write("ELEVATE AI API - PRODUCTION READINESS TEST REPORT\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Duration: {self.end_time - self.start_time:.2f} seconds\n\n")
                
                for module_key, result_info in self.results.items():
                    module = self.test_modules[module_key]
                    f.write(f"Module: {module['name']}\n")
                    f.write(f"Status: {result_info['status']}\n")
                    f.write(f"Return Code: {result_info['return_code']}\n")
                    
                    if result_info.get('stdout'):
                        f.write("Output:\n")
                        f.write(result_info['stdout'])
                        f.write("\n")
                    
                    if result_info.get('stderr'):
                        f.write("Errors:\n")
                        f.write(result_info['stderr'])
                        f.write("\n")
                    
                    f.write("-" * 40 + "\n\n")
            
            print(f"📄 Detailed report saved to: {report_file}")
            
        except Exception as e:
            print(f"⚠️  Failed to generate report file: {e}")

async def main():
    """Main function."""
    runner = ProductionTestRunner()
    
    # Parse command line arguments
    args = sys.argv[1:]
    
    if "--help" in args or "-h" in args:
        runner.print_banner()
        runner.print_help()
        return
    
    runner.print_banner()
    
    try:
        if not args:
            # Run all tests
            print("🎯 Running all production readiness tests...")
            success = await runner.run_all_tests()
        else:
            # Run specific modules
            success = await runner.run_specific_modules(args)
        
        # Generate report file
        runner.generate_report_file()
        
        # Exit with appropriate code
        if success:
            print("\n🎉 Production test suite completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Some production tests failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
