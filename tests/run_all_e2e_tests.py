#!/usr/bin/env python3
"""
Master E2E Test Runner
Runs all end-to-end tests for the premium API workflows.
"""

import asyncio
import sys
import os
import subprocess
import json
from datetime import datetime
from typing import List, Dict, Any

# Test scripts to run
TEST_SCRIPTS = [
    "test_e2e_premium_chat_workflow.py",
    "test_e2e_search_workflow.py", 
    "test_e2e_cost_optimization_workflow.py"
]

class E2ETestRunner:
    """Master test runner for all e2e tests"""
    
    def __init__(self):
        self.results = []
        self.start_time = datetime.utcnow()
    
    async def run_test_script(self, script_name: str) -> Dict[str, Any]:
        """Run a single test script"""
        print(f"\n{'='*60}")
        print(f"🚀 Running: {script_name}")
        print(f"{'='*60}")
        
        try:
            # Run the test script
            result = subprocess.run(
                [sys.executable, script_name],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per script
            )
            
            success = result.returncode == 0
            
            # Parse output for summary
            output_lines = result.stdout.split('\n')
            test_summary = "Unknown"
            
            for line in output_lines:
                if "Results:" in line:
                    test_summary = line.strip()
                    break
            
            return {
                "script": script_name,
                "success": success,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "summary": test_summary,
                "timestamp": datetime.utcnow()
            }
            
        except subprocess.TimeoutExpired:
            return {
                "script": script_name,
                "success": False,
                "return_code": -1,
                "stdout": "",
                "stderr": "Test timed out after 5 minutes",
                "summary": "TIMEOUT",
                "timestamp": datetime.utcnow()
            }
        except Exception as e:
            return {
                "script": script_name,
                "success": False,
                "return_code": -1,
                "stdout": "",
                "stderr": str(e),
                "summary": "EXCEPTION",
                "timestamp": datetime.utcnow()
            }
    
    async def run_all_tests(self):
        """Run all e2e test scripts"""
        print("🎯 MASTER E2E TEST RUNNER")
        print(f"📅 Started at: {self.start_time}")
        print(f"📋 Test scripts: {len(TEST_SCRIPTS)}")
        
        for script in TEST_SCRIPTS:
            if os.path.exists(script):
                result = await self.run_test_script(script)
                self.results.append(result)
            else:
                print(f"⚠️  Test script not found: {script}")
                self.results.append({
                    "script": script,
                    "success": False,
                    "return_code": -1,
                    "stdout": "",
                    "stderr": f"Script not found: {script}",
                    "summary": "NOT_FOUND",
                    "timestamp": datetime.utcnow()
                })
        
        # Print final summary
        self.print_final_summary()
        
        # Save results
        self.save_results()
    
    def print_final_summary(self):
        """Print final test summary"""
        print(f"\n{'='*60}")
        print("📊 MASTER E2E TEST SUMMARY")
        print(f"{'='*60}")
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r['success'])
        failed_tests = total_tests - passed_tests
        
        print(f"🎯 Overall Results: {passed_tests}/{total_tests} test suites passed")
        print(f"⏱️  Total runtime: {(datetime.utcnow() - self.start_time).total_seconds():.1f} seconds")
        
        print(f"\n📋 Detailed Results:")
        for result in self.results:
            status = "✅ PASSED" if result['success'] else "❌ FAILED"
            print(f"   {status} {result['script']}")
            if not result['success']:
                print(f"      Error: {result['stderr'][:100]}...")
        
        if passed_tests == total_tests:
            print(f"\n🎉 ALL TEST SUITES PASSED! E2E testing complete.")
        else:
            print(f"\n⚠️  {failed_tests} test suite(s) failed. Please review the implementation.")
    
    def save_results(self):
        """Save test results to file"""
        results_file = "e2e_master_results.json"
        
        with open(results_file, "w") as f:
            json.dump({
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.utcnow().isoformat(),
                "total_tests": len(self.results),
                "passed_tests": sum(1 for r in self.results if r['success']),
                "results": self.results
            }, f, indent=2, default=str)
        
        print(f"\n📄 Results saved to: {results_file}")

async def main():
    """Main test runner"""
    runner = E2ETestRunner()
    await runner.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
