#!/usr/bin/env python3
"""
Performance Testing for Blueprint Section Operations
Tests performance of section-aware operations with hierarchical data structures.
"""

import asyncio
import time
import statistics
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from app.models.blueprint_centric import (
    BlueprintSection
)
from app.api.schemas import BlueprintSectionTreeResponse
from app.services.blueprint_section_service import BlueprintSectionService
from app.services.content_aggregator import ContentAggregator
from app.services.knowledge_graph_traversal import KnowledgeGraphTraversal

@dataclass
class PerformanceResult:
    """Performance test result"""
    operation: str
    duration: float
    success: bool
    data_size: int
    error_message: Optional[str] = None

@dataclass
class PerformanceBenchmark:
    """Performance benchmark results"""
    operation: str
    iterations: int
    avg_duration: float
    min_duration: float
    max_duration: float
    p95_duration: float
    p99_duration: float
    success_rate: float
    throughput: float  # operations per second

class BlueprintSectionPerformanceTester:
    """Performance tester for blueprint section operations"""
    
    def __init__(self):
        self.section_service = BlueprintSectionService()
        self.content_aggregator = ContentAggregator()
        self.graph_traversal = KnowledgeGraphTraversal()
        self.results: List[PerformanceResult] = []
        
    async def generate_test_sections(self, blueprint_id: str, num_sections: int, max_depth: int = 5) -> List[str]:
        """Generate test sections for performance testing"""
        section_ids = []
        
        for i in range(num_sections):
            # Create sections with varying depths
            depth = min(i % max_depth, max_depth - 1)
            parent_id = section_ids[depth - 1] if depth > 0 and section_ids else None
            
            section_data = {
                "title": f"Test Section {i+1}",
                "description": f"Performance test section {i+1}",
                "blueprint_id": blueprint_id,
                "parent_section_id": parent_id,
                "depth": depth,
                "order_index": i,
                "difficulty": "BEGINNER",
                "user_id": 1
            }
            
            try:
                section = await self.section_service.create_section(section_data)
                section_ids.append(section.id)
            except Exception as e:
                print(f"Failed to create test section {i+1}: {e}")
                
        return section_ids
    
    async def test_section_tree_construction(self, blueprint_id: str, num_sections: int) -> PerformanceBenchmark:
        """Test section tree construction performance"""
        print(f"🧪 Testing section tree construction with {num_sections} sections...")
        
        # Generate test sections
        section_ids = await self.generate_test_sections(blueprint_id, num_sections)
        
        durations = []
        successes = 0
        
        for _ in range(10):  # 10 iterations for reliable metrics
            try:
                start_time = time.time()
                
                # Build section tree
                tree = await self.section_service.get_section_tree(blueprint_id)
                
                end_time = time.time()
                duration = end_time - start_time
                durations.append(duration)
                successes += 1
                
            except Exception as e:
                print(f"Section tree construction failed: {e}")
                durations.append(0)
        
        # Calculate statistics
        successful_durations = [d for d in durations if d > 0]
        if successful_durations:
            avg_duration = statistics.mean(successful_durations)
            min_duration = min(successful_durations)
            max_duration = max(successful_durations)
            p95_duration = statistics.quantiles(successful_durations, n=20)[18]  # 95th percentile
            p99_duration = statistics.quantiles(successful_durations, n=100)[98]  # 99th percentile
        else:
            avg_duration = min_duration = max_duration = p95_duration = p99_duration = 0
        
        success_rate = successes / len(durations)
        throughput = 1.0 / avg_duration if avg_duration > 0 else 0
        
        return PerformanceBenchmark(
            operation="section_tree_construction",
            iterations=len(durations),
            avg_duration=avg_duration,
            min_duration=min_duration,
            max_duration=max_duration,
            p95_duration=p95_duration,
            p99_duration=p99_duration,
            success_rate=success_rate,
            throughput=throughput
        )
    
    async def test_content_aggregation_performance(self, section_id: str, num_iterations: int = 10) -> PerformanceBenchmark:
        """Test content aggregation performance"""
        print(f"🧪 Testing content aggregation for section {section_id}...")
        
        durations = []
        successes = 0
        
        for _ in range(num_iterations):
            try:
                start_time = time.time()
                
                # Aggregate section content
                content = await self.content_aggregator.aggregate_section_content(section_id)
                
                end_time = time.time()
                duration = end_time - start_time
                durations.append(duration)
                successes += 1
                
            except Exception as e:
                print(f"Content aggregation failed: {e}")
                durations.append(0)
        
        # Calculate statistics
        successful_durations = [d for d in durations if d > 0]
        if successful_durations:
            avg_duration = statistics.mean(successful_durations)
            min_duration = min(successful_durations)
            max_duration = max(successful_durations)
            p95_duration = statistics.quantiles(successful_durations, n=20)[18]
            p99_duration = statistics.quantiles(successful_durations, n=100)[98]
        else:
            avg_duration = min_duration = max_duration = p95_duration = p99_duration = 0
        
        success_rate = successes / len(durations)
        throughput = 1.0 / avg_duration if avg_duration > 0 else 0
        
        return PerformanceBenchmark(
            operation="content_aggregation",
            iterations=len(durations),
            avg_duration=avg_duration,
            min_duration=min_duration,
            max_duration=max_duration,
            p95_duration=p95_duration,
            p99_duration=p99_duration,
            success_rate=success_rate,
            throughput=throughput
        )
    
    async def test_graph_traversal_performance(self, start_node_id: str, max_depth: int = 3, num_iterations: int = 10) -> PerformanceBenchmark:
        """Test knowledge graph traversal performance"""
        print(f"🧪 Testing graph traversal from node {start_node_id} with max depth {max_depth}...")
        
        durations = []
        successes = 0
        
        for _ in range(num_iterations):
            try:
                start_time = time.time()
                
                # Traverse knowledge graph
                result = await self.graph_traversal.traverse_graph(start_node_id, max_depth)
                
                end_time = time.time()
                duration = end_time - start_time
                durations.append(duration)
                successes += 1
                
            except Exception as e:
                print(f"Graph traversal failed: {e}")
                durations.append(0)
        
        # Calculate statistics
        successful_durations = [d for d in durations if d > 0]
        if successful_durations:
            avg_duration = statistics.mean(successful_durations)
            min_duration = min(successful_durations)
            max_duration = max(successful_durations)
            p95_duration = statistics.quantiles(successful_durations, n=20)[18]
            p99_duration = statistics.quantiles(successful_durations, n=100)[98]
        else:
            avg_duration = min_duration = max_duration = p95_duration = p99_duration = 0
        
        success_rate = successes / len(durations)
        throughput = 1.0 / avg_duration if avg_duration > 0 else 0
        
        return PerformanceBenchmark(
            operation="graph_traversal",
            iterations=len(durations),
            avg_duration=avg_duration,
            min_duration=min_duration,
            max_duration=max_duration,
            p95_duration=p95_duration,
            p99_duration=p99_duration,
            success_rate=success_rate,
            throughput=throughput
        )
    
    async def test_concurrent_section_operations(self, blueprint_id: str, num_concurrent: int = 10) -> PerformanceBenchmark:
        """Test concurrent section operations performance"""
        print(f"🧪 Testing {num_concurrent} concurrent section operations...")
        
        async def single_operation():
            """Single concurrent operation"""
            try:
                start_time = time.time()
                
                # Simulate section operation (read tree)
                tree = await self.section_service.get_section_tree(blueprint_id)
                
                end_time = time.time()
                return end_time - start_time, True
                
            except Exception as e:
                return 0, False
        
        # Run concurrent operations
        start_time = time.time()
        tasks = [single_operation() for _ in range(num_concurrent)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Process results
        durations = []
        successes = 0
        
        for result in results:
            if isinstance(result, tuple):
                duration, success = result
                durations.append(duration)
                if success:
                    successes += 1
            else:
                durations.append(0)
        
        # Calculate statistics
        successful_durations = [d for d in durations if d > 0]
        if successful_durations:
            avg_duration = statistics.mean(successful_durations)
            min_duration = min(successful_durations)
            max_duration = max(successful_durations)
            p95_duration = statistics.quantiles(successful_durations, n=20)[18]
            p99_duration = statistics.quantiles(successful_durations, n=100)[98]
        else:
            avg_duration = min_duration = max_duration = p95_duration = p99_duration = 0
        
        success_rate = successes / len(durations)
        throughput = num_concurrent / total_time if total_time > 0 else 0
        
        return PerformanceBenchmark(
            operation="concurrent_section_operations",
            iterations=len(durations),
            avg_duration=avg_duration,
            min_duration=min_duration,
            max_duration=max_duration,
            p95_duration=p95_duration,
            p99_duration=p99_duration,
            success_rate=success_rate,
            throughput=throughput
        )
    
    async def run_all_performance_tests(self, blueprint_id: str) -> Dict[str, PerformanceBenchmark]:
        """Run all performance tests"""
        print("🚀 Starting Blueprint Section Performance Tests")
        print("=" * 60)
        
        benchmarks = {}
        
        # Test 1: Section tree construction with different sizes
        for num_sections in [10, 50, 100, 500]:
            benchmark = await self.test_section_tree_construction(blueprint_id, num_sections)
            benchmarks[f"tree_construction_{num_sections}_sections"] = benchmark
            
            # Performance target: <200ms for section navigation
            if benchmark.avg_duration < 0.2:
                print(f"✅ Tree construction with {num_sections} sections: {benchmark.avg_duration*1000:.1f}ms (Target: <200ms)")
            else:
                print(f"⚠️  Tree construction with {num_sections} sections: {benchmark.avg_duration*1000:.1f}ms (Target: <200ms)")
        
        # Test 2: Content aggregation
        if benchmarks:
            # Use first section for testing
            first_section_id = "test-section-1"  # This would be the actual ID from test data
            benchmark = await self.test_content_aggregation_performance(first_section_id)
            benchmarks["content_aggregation"] = benchmark
            
            # Performance target: <300ms for context assembly
            if benchmark.avg_duration < 0.3:
                print(f"✅ Content aggregation: {benchmark.avg_duration*1000:.1f}ms (Target: <300ms)")
            else:
                print(f"⚠️  Content aggregation: {benchmark.avg_duration*1000:.1f}ms (Target: <300ms)")
        
        # Test 3: Graph traversal
        if benchmarks:
            start_node_id = "test-node-1"  # This would be the actual ID from test data
            benchmark = await self.test_graph_traversal_performance(start_node_id)
            benchmarks["graph_traversal"] = benchmark
            
            # Performance target: <500ms for graph traversal
            if benchmark.avg_duration < 0.5:
                print(f"✅ Graph traversal: {benchmark.avg_duration*1000:.1f}ms (Target: <500ms)")
            else:
                print(f"⚠️  Graph traversal: {benchmark.avg_duration*1000:.1f}ms (Target: <500ms)")
        
        # Test 4: Concurrent operations
        benchmark = await self.test_concurrent_section_operations(blueprint_id, 50)
        benchmarks["concurrent_operations"] = benchmark
        
        # Performance target: <1s total for concurrent operations
        if benchmark.avg_duration < 1.0:
            print(f"✅ Concurrent operations: {benchmark.avg_duration*1000:.1f}ms (Target: <1s)")
        else:
            print(f"⚠️  Concurrent operations: {benchmark.avg_duration*1000:.1f}ms (Target: <1s)")
        
        return benchmarks
    
    def print_performance_summary(self, benchmarks: Dict[str, PerformanceBenchmark]):
        """Print comprehensive performance summary"""
        print("\n📊 PERFORMANCE TEST SUMMARY")
        print("=" * 60)
        
        for name, benchmark in benchmarks.items():
            print(f"\n🔍 {name.replace('_', ' ').title()}")
            print(f"   Iterations: {benchmark.iterations}")
            print(f"   Avg Duration: {benchmark.avg_duration*1000:.1f}ms")
            print(f"   Min Duration: {benchmark.min_duration*1000:.1f}ms")
            print(f"   Max Duration: {benchmark.max_duration*1000:.1f}ms")
            print(f"   P95 Duration: {benchmark.p95_duration*1000:.1f}ms")
            print(f"   P99 Duration: {benchmark.p99_duration*1000:.1f}ms")
            print(f"   Success Rate: {benchmark.success_rate*100:.1f}%")
            print(f"   Throughput: {benchmark.throughput:.2f} ops/sec")
        
        # Overall performance assessment
        print("\n🎯 OVERALL PERFORMANCE ASSESSMENT")
        print("=" * 60)
        
        all_avg_durations = [b.avg_duration for b in benchmarks.values()]
        overall_avg = statistics.mean(all_avg_durations)
        
        print(f"Overall Average Response Time: {overall_avg*1000:.1f}ms")
        
        # Check against performance targets
        targets_met = 0
        total_targets = 0
        
        for name, benchmark in benchmarks.items():
            if "tree_construction" in name:
                target = 0.2  # 200ms
                total_targets += 1
                if benchmark.avg_duration < target:
                    targets_met += 1
            elif "content_aggregation" in name:
                target = 0.3  # 300ms
                total_targets += 1
                if benchmark.avg_duration < target:
                    targets_met += 1
            elif "graph_traversal" in name:
                target = 0.5  # 500ms
                total_targets += 1
                if benchmark.avg_duration < target:
                    targets_met += 1
            elif "concurrent_operations" in name:
                target = 1.0  # 1s
                total_targets += 1
                if benchmark.avg_duration < target:
                    targets_met += 1
        
        print(f"Performance Targets Met: {targets_met}/{total_targets}")
        
        if targets_met == total_targets:
            print("🎉 All performance targets met! Excellent performance.")
        elif targets_met >= total_targets * 0.8:
            print("⚠️  Most performance targets met. Good performance with room for improvement.")
        else:
            print("❌ Multiple performance targets missed. Performance needs optimization.")

async def main():
    """Main performance testing function"""
    print("🚀 Blueprint Section Performance Testing")
    print("=" * 60)
    
    # Initialize tester
    tester = BlueprintSectionPerformanceTester()
    
    # Test blueprint ID (this would be created or provided)
    test_blueprint_id = 12345
    
    try:
        # Run all performance tests
        benchmarks = await tester.run_all_performance_tests(test_blueprint_id)
        
        # Print comprehensive summary
        tester.print_performance_summary(benchmarks)
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"performance_results_{timestamp}.json"
        
        # Convert benchmarks to serializable format
        results_data = {}
        for name, benchmark in benchmarks.items():
            results_data[name] = {
                "operation": benchmark.operation,
                "iterations": benchmark.iterations,
                "avg_duration": benchmark.avg_duration,
                "min_duration": benchmark.min_duration,
                "max_duration": benchmark.max_duration,
                "p95_duration": benchmark.p95_duration,
                "p99_duration": benchmark.p99_duration,
                "success_rate": benchmark.success_rate,
                "throughput": benchmark.throughput
            }
        
        import json
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n💾 Performance results saved to: {results_file}")
        
    except Exception as e:
        print(f"❌ Performance testing failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
