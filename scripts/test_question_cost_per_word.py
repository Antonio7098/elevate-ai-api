#!/usr/bin/env python3
"""
Test script to measure question generation API costs per question for different settings.
"""
import asyncio
import json
import time
from typing import Dict, List
import httpx
from pathlib import Path

API_URL = "http://127.0.0.1:8000/api/v1/ai-rag/learning-blueprints/{blueprint_id}/question-sets"
API_KEY = "test_api_key_123"
DECONSTRUCTIONS_DIR = Path("deconstructions")

# You may want to update this to use a real blueprint_id from your deconstructions
DEFAULT_BLUEPRINT_ID = None


def get_any_blueprint_id():
    """Get a blueprint_id from any deconstruction file."""
    files = sorted([f for f in DECONSTRUCTIONS_DIR.glob("*.json")])
    for f in files:
        with open(f, "r") as file:
            data = json.load(file)
            blueprint_id = data.get("blueprint_id")
            if blueprint_id:
                return blueprint_id
    return None


def calculate_auto_question_count(blueprint_data):
    """Calculate appropriate number of questions to cover as many concepts as possible efficiently."""
    kp = blueprint_data.get("blueprint_json", {}).get("knowledge_primitives", {})
    
    # Count different types of knowledge primitives
    propositions = len(kp.get('key_propositions_and_facts', []))
    entities = len(kp.get('key_entities_and_definitions', []))
    processes = len(kp.get('described_processes_and_steps', []))
    relationships = len(kp.get('identified_relationships', []))
    
    total_primitives = propositions + entities + processes + relationships
    
    if total_primitives == 0:
        return 3  # Minimum for basic content
    
    # Strategy: Cover multiple concepts per question when possible
    # Start with a base that aims to cover most concepts efficiently
    base_count = max(3, min(12, total_primitives * 0.8))
    
    # Adjust based on content type:
    # - More questions for processes (they're complex)
    # - Fewer questions for simple entity lists
    # - Relationships can often be tested in combination with other concepts
    
    if processes > 0:
        # Processes need dedicated questions, but try to combine with related concepts
        base_count += min(processes, 2)  # Cap process questions
    if relationships > 0:
        # Relationships can often be tested alongside other concepts
        base_count += min(relationships // 2, 2)  # Every 2 relationships = 1 question
    
    # Ensure we don't exceed reasonable limits
    return int(min(15, base_count))  # Cap at 15 questions


async def test_question_generation_cost(blueprint_id: str, question_count: int, test_type: str = "fixed") -> Dict:
    """Test question generation cost for a given blueprint and question count."""
    print(f"\n🧪 Generating {question_count} questions ({test_type}) for blueprint {blueprint_id}...")
    payload = {
        "name": f"Cost Test ({question_count} questions - {test_type})",
        "question_options": {
            "count": question_count
        }
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    url = API_URL.format(blueprint_id=blueprint_id)
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        start_time = time.time()
        try:
            response = await client.post(url, headers=headers, json=payload)
            end_time = time.time()
            if response.status_code == 200:
                result = response.json()
                questions = result.get("questions", [])
                print(f"✅ Generated {len(questions)} questions in {end_time - start_time:.2f}s")
                # Cost tracking would go here if available
                return {
                    "question_count": question_count,
                    "total_questions": len(questions),
                    "processing_time": end_time - start_time,
                    "test_type": test_type,
                    "status": "success"
                }
            else:
                print(f"❌ Generation failed: {response.status_code} - {response.text}")
                return {
                    "question_count": question_count,
                    "total_questions": 0,
                    "processing_time": end_time - start_time,
                    "test_type": test_type,
                    "status": "failed"
                }
        except httpx.TimeoutException:
            end_time = time.time()
            print(f"⏰ Request timed out after {end_time - start_time:.2f}s")
            return {
                "question_count": question_count,
                "total_questions": 0,
                "processing_time": end_time - start_time,
                "test_type": test_type,
                "status": "timeout"
            }
        except Exception as e:
            end_time = time.time()
            print(f"❌ Request failed with error: {str(e)}")
            return {
                "question_count": question_count,
                "total_questions": 0,
                "processing_time": end_time - start_time,
                "test_type": test_type,
                "status": "error"
            }


async def test_auto_question_generation(blueprint_id: str) -> Dict:
    """Test auto-calculated question generation."""
    # Load blueprint data to calculate auto count
    files = sorted([f for f in DECONSTRUCTIONS_DIR.glob("*.json")])
    for f in files:
        with open(f, "r") as file:
            data = json.load(file)
            if data.get("blueprint_id") == blueprint_id:
                auto_count = calculate_auto_question_count(data)
                print(f"\n🧠 Auto-calculated question count: {auto_count}")
                return await test_question_generation_cost(blueprint_id, auto_count, "auto")
    
    print(f"❌ No blueprint data found for {blueprint_id}")
    return {
        "question_count": 0,
        "total_questions": 0,
        "processing_time": 0,
        "test_type": "auto",
        "status": "error"
    }


async def main():
    print("🚀 Starting Question Generation Cost Test")
    print("=" * 60)
    blueprint_id = DEFAULT_BLUEPRINT_ID or get_any_blueprint_id()
    if not blueprint_id:
        print("[red]No blueprint_id found in deconstructions/. Run batch_deconstruct_and_view.py first.[/red]")
        return
    
    # Test fixed question counts
    fixed_question_counts = [1, 5, 10, 15]
    results = []
    
    print("\n📊 Testing Fixed Question Counts:")
    print("-" * 40)
    for count in fixed_question_counts:
        result = await test_question_generation_cost(blueprint_id, count, "fixed")
        results.append(result)
        await asyncio.sleep(1)
    
    # Test auto-calculated question count
    print("\n🧠 Testing Auto-Calculated Question Count:")
    print("-" * 40)
    auto_result = await test_auto_question_generation(blueprint_id)
    results.append(auto_result)
    
    # Display results
    print("\n" + "=" * 60)
    print("📊 QUESTION GENERATION COST TEST RESULTS")
    print("=" * 60)
    
    print("\n📝 Fixed Count Tests:")
    for result in results[:-1]:  # All except the last (auto) result
        if result['status'] == 'success':
            print(f"  • {result['question_count']} requested: {result['total_questions']} generated in {result['processing_time']:.2f}s")
        else:
            print(f"  • {result['question_count']} requested: {result['status'].upper()} after {result['processing_time']:.2f}s")
    
    print("\n🧠 Auto-Calculated Test:")
    auto_result = results[-1]
    if auto_result['status'] == 'success':
        print(f"  • {auto_result['question_count']} auto-calculated: {auto_result['total_questions']} generated in {auto_result['processing_time']:.2f}s")
    else:
        print(f"  • Auto-calculated: {auto_result['status'].upper()} after {auto_result['processing_time']:.2f}s")
    
    # Summary statistics
    successful_results = [r for r in results if r['status'] == 'success']
    if successful_results:
        avg_time = sum(r['processing_time'] for r in successful_results) / len(successful_results)
        avg_questions = sum(r['total_questions'] for r in successful_results) / len(successful_results)
        print(f"\n📈 Summary:")
        print(f"  • Average processing time: {avg_time:.2f}s")
        print(f"  • Average questions generated: {avg_questions:.1f}")
        print(f"  • Success rate: {len(successful_results)}/{len(results)} ({len(successful_results)/len(results)*100:.1f}%)")


if __name__ == "__main__":
    asyncio.run(main()) 