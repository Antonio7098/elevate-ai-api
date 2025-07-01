#!/usr/bin/env python3
"""
Test script to measure LLM API costs per word for different source lengths.
"""

import asyncio
import json
import time
from typing import Dict, List
import httpx
from app.core.usage_tracker import usage_tracker


def generate_test_text(word_count: int) -> str:
    """Generate test text with specified word count."""
    base_text = """
    Photosynthesis is the process by which plants convert light energy into chemical energy. 
    This fundamental biological process occurs in chloroplasts, specialized organelles found in plant cells. 
    The process involves two main stages: the light-dependent reactions and the Calvin cycle.
    
    During the light-dependent reactions, chlorophyll molecules in the thylakoid membranes capture photons of light. 
    This energy is used to split water molecules, releasing oxygen as a byproduct and generating high-energy electrons. 
    These electrons are then transferred through an electron transport chain, creating a proton gradient across the thylakoid membrane.
    
    The energy from this proton gradient is used to synthesize ATP through a process called chemiosmosis. 
    Additionally, the electrons are used to reduce NADP+ to NADPH, which carries high-energy electrons to the next stage.
    
    The Calvin cycle, also known as the light-independent reactions, takes place in the stroma of the chloroplast. 
    This cycle uses the ATP and NADPH produced in the light reactions to convert carbon dioxide into organic molecules. 
    The process begins with carbon fixation, where CO2 is combined with a five-carbon sugar called ribulose bisphosphate (RuBP).
    
    This reaction is catalyzed by the enzyme rubisco, which is the most abundant protein on Earth. 
    The resulting six-carbon compound immediately splits into two molecules of 3-phosphoglycerate (3-PGA). 
    These molecules are then phosphorylated using ATP and reduced using NADPH to form glyceraldehyde-3-phosphate (G3P).
    
    Some of the G3P molecules are used to synthesize glucose and other carbohydrates, while others are recycled to regenerate RuBP. 
    This regeneration process requires additional ATP and completes the cycle. For every three molecules of CO2 that enter the cycle, 
    one molecule of G3P is produced, which can be used to make glucose or other organic compounds.
    
    The overall equation for photosynthesis is: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2. 
    This process is essential for life on Earth as it provides the oxygen we breathe and the organic compounds that form the base of most food chains.
    
    Several factors affect the rate of photosynthesis, including light intensity, carbon dioxide concentration, temperature, and water availability. 
    Light intensity directly affects the rate of the light-dependent reactions, while CO2 concentration affects the rate of carbon fixation in the Calvin cycle.
    
    Temperature influences the activity of enzymes involved in the process, with optimal rates typically occurring between 20-30°C for most plants. 
    Water is essential for the light-dependent reactions, as it provides the electrons and protons needed for the process.
    
    Different plants have evolved various adaptations to optimize photosynthesis under different environmental conditions. 
    C3 plants, which include most trees and agricultural crops, use the standard Calvin cycle. C4 plants, such as corn and sugarcane, 
    have evolved a more efficient carbon fixation pathway that reduces photorespiration.
    
    CAM plants, including many succulents and cacti, open their stomata at night to take in CO2 and store it as organic acids. 
    During the day, they close their stomata to conserve water and use the stored CO2 for photosynthesis.
    
    Understanding photosynthesis is crucial for agriculture, as it helps farmers optimize growing conditions to maximize crop yields. 
    It also has implications for climate change, as plants play a vital role in removing CO2 from the atmosphere.
    
    Research into photosynthesis continues to reveal new insights into this complex process. Scientists are studying ways to improve 
    photosynthetic efficiency in crops to help feed a growing global population while reducing the environmental impact of agriculture.
    
    The study of photosynthesis also has applications in renewable energy, as researchers explore ways to mimic this process 
    to create artificial photosynthesis systems for producing clean fuels from sunlight, water, and CO2.
    """
    
    # Split into words and repeat/truncate to get desired length
    words = base_text.split()
    if len(words) < word_count:
        # Repeat the text to reach desired length
        repeat_count = (word_count // len(words)) + 1
        words = (words * repeat_count)[:word_count]
    else:
        # Truncate to desired length
        words = words[:word_count]
    
    return " ".join(words)


async def test_deconstruction_cost(source_text: str, word_count: int) -> Dict:
    """Test deconstruction cost for a given source text."""
    print(f"\n🧪 Testing {word_count}-word source...")
    print(f"📝 Text length: {len(source_text)} characters, {word_count} words")
    
    # Get usage before test
    usage_before = usage_tracker.get_usage_summary()
    total_cost_before = sum(provider['cost_usd'] for provider in usage_before['by_provider'].values())
    
    # Make API call with increased timeout
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:  # 5 minute timeout
        start_time = time.time()
        try:
            response = await client.post(
                "http://127.0.0.1:8000/api/v1/deconstruct",
                headers={
                    "Authorization": "Bearer test_api_key_123",
                    "Content-Type": "application/json"
                },
                json={
                    "source_text": source_text,
                    "source_type_hint": "test"
                }
            )
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Deconstruction successful in {end_time - start_time:.2f}s")
                
                # Get usage after test
                usage_after = usage_tracker.get_usage_summary()
                total_cost_after = sum(provider['cost_usd'] for provider in usage_after['by_provider'].values())
                
                # Calculate cost for this test
                cost_increment = total_cost_after - total_cost_before
                
                # Calculate cost per word
                cost_per_word = cost_increment / word_count if word_count > 0 else 0
                
                # Get recent usage to see what operations were performed
                recent_usage = usage_tracker.get_recent_usage(limit=10)
                
                return {
                    "word_count": word_count,
                    "char_count": len(source_text),
                    "total_cost": cost_increment,
                    "cost_per_word": cost_per_word,
                    "cost_per_1000_words": cost_per_word * 1000,
                    "processing_time": end_time - start_time,
                    "operations": [record['operation'] for record in recent_usage[:5]],
                    "blueprint_id": result.get('blueprint_id'),
                    "status": "success"
                }
            else:
                print(f"❌ Deconstruction failed: {response.status_code} - {response.text}")
                return {
                    "word_count": word_count,
                    "char_count": len(source_text),
                    "total_cost": 0,
                    "cost_per_word": 0,
                    "cost_per_1000_words": 0,
                    "processing_time": end_time - start_time,
                    "operations": [],
                    "blueprint_id": None,
                    "status": "failed"
                }
        except httpx.TimeoutException:
            end_time = time.time()
            print(f"⏰ Request timed out after {end_time - start_time:.2f}s")
            return {
                "word_count": word_count,
                "char_count": len(source_text),
                "total_cost": 0,
                "cost_per_word": 0,
                "cost_per_1000_words": 0,
                "processing_time": end_time - start_time,
                "operations": [],
                "blueprint_id": None,
                "status": "timeout"
            }
        except Exception as e:
            end_time = time.time()
            print(f"❌ Request failed with error: {str(e)}")
            return {
                "word_count": word_count,
                "char_count": len(source_text),
                "total_cost": 0,
                "cost_per_word": 0,
                "cost_per_1000_words": 0,
                "processing_time": end_time - start_time,
                "operations": [],
                "blueprint_id": None,
                "status": "error"
            }


async def main():
    """Main test function."""
    print("🚀 Starting Cost Per Word Test")
    print("=" * 50)
    
    # Test different word counts
    word_counts = [100, 500, 1000]
    results = []
    
    for word_count in word_counts:
        # Generate test text
        test_text = generate_test_text(word_count)
        
        # Test deconstruction
        result = await test_deconstruction_cost(test_text, word_count)
        results.append(result)
        
        # Small delay between tests
        await asyncio.sleep(1)
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 COST PER WORD TEST RESULTS")
    print("=" * 50)
    
    for result in results:
        if result['status'] == 'success':
            print(f"\n📄 {result['word_count']} words:")
            print(f"   💰 Total cost: ${result['total_cost']:.6f}")
            print(f"   📊 Cost per word: ${result['cost_per_word']:.8f}")
            print(f"   📈 Cost per 1000 words: ${result['cost_per_1000_words']:.6f}")
            print(f"   ⏱️  Processing time: {result['processing_time']:.2f}s")
            print(f"   🔧 Operations: {', '.join(result['operations'])}")
        elif result['status'] == 'timeout':
            print(f"\n⏰ {result['word_count']} words: TIMEOUT after {result['processing_time']:.2f}s")
        elif result['status'] == 'error':
            print(f"\n❌ {result['word_count']} words: ERROR after {result['processing_time']:.2f}s")
        else:
            print(f"\n❌ {result['word_count']} words: FAILED after {result['processing_time']:.2f}s")
    
    # Calculate averages
    successful_results = [r for r in results if r['status'] == 'success']
    if successful_results:
        avg_cost_per_word = sum(r['cost_per_word'] for r in successful_results) / len(successful_results)
        avg_cost_per_1000_words = sum(r['cost_per_1000_words'] for r in successful_results) / len(successful_results)
        
        print(f"\n📊 AVERAGE COSTS (from {len(successful_results)} successful tests):")
        print(f"   💰 Average cost per word: ${avg_cost_per_word:.8f}")
        print(f"   📈 Average cost per 1000 words: ${avg_cost_per_1000_words:.6f}")
        
        print(f"\n💡 COST ESTIMATES FOR DIFFERENT DOCUMENT SIZES:")
        print(f"   📄 1,000 words: ~${avg_cost_per_1000_words:.6f}")
        print(f"   📄 5,000 words: ~${avg_cost_per_1000_words * 5:.6f}")
        print(f"   📄 10,000 words: ~${avg_cost_per_1000_words * 10:.6f}")
        print(f"   📄 50,000 words: ~${avg_cost_per_1000_words * 50:.6f}")
        print(f"   📄 100,000 words: ~${avg_cost_per_1000_words * 100:.6f}")
    else:
        print(f"\n❌ No successful tests completed. Check server status and LLM configuration.")
        print(f"   Make sure the FastAPI server is running and Google AI API is properly configured.")


if __name__ == "__main__":
    asyncio.run(main())
