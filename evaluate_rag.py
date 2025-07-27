import os
import json
import time
import requests
from sentence_transformers import SentenceTransformer, util

# --- Configuration ---
RAG_API_URL = "http://localhost:8000/rag/chat"  # Your RAG API endpoint
TEST_CASES_FILE = "/home/antonio/programming/elevate/core_and_ai/elevate-ai-api/test_cases.json"
REPORT_FILE = "/home/antonio/programming/elevate/core_and_ai/elevate-ai-api/evaluation_report.json"

# Load the sentence transformer model for semantic similarity
# This might download the model on the first run, which can take a few minutes.
print("Loading sentence transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")

def get_rag_response(query: str) -> dict:
    """
    Sends a query to the RAG API and returns the JSON response.
    """
    try:
        response = requests.post(
            RAG_API_URL,
            json={"query": query}
        )
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling RAG API: {e}")
        return None

def evaluate_semantic_similarity(expected_answer: str, actual_answer: str) -> float:
    """
    Calculates the cosine similarity between the embeddings of the expected and actual answers.
    """
    if not actual_answer:
        return 0.0
    embedding1 = model.encode(expected_answer, convert_to_tensor=True)
    embedding2 = model.encode(actual_answer, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding1, embedding2)
    return similarity.item()

def evaluate_with_llm_judge(query: str, expected_answer: str, actual_answer: str) -> dict:
    """
    Uses a powerful LLM (like a hypothetical Gemini Pro) to judge the quality of the answer.
    NOTE: This is a placeholder and requires a real LLM API call.
    """
    if not actual_answer:
        return {
            "score": 0,
            "reasoning": "No answer provided."
        }
        
    # Simulate a call to a powerful LLM for evaluation
    # This would involve formatting a prompt and sending it to the LLM API
    # For example:
    # prompt = f"Query: {query}\nExpected: {expected_answer}\nActual: {actual_answer}\n\nIs the actual answer a good response to the query? Score it from 1-5."
    # llm_response = call_gemini_api(prompt) 
    
    # Mocked response for demonstration
    score = 0
    reasoning = "Evaluation with a real LLM judge is not implemented in this script."

    # A simple keyword-based check as a stand-in
    expected_words = set(expected_answer.lower().split())
    actual_words = set(actual_answer.lower().split())
    common_words = expected_words.intersection(actual_words)
    
    if len(expected_words) > 0:
        score = round((len(common_words) / len(expected_words)) * 5)

    reasoning = f"Mock evaluation based on keyword overlap. Score: {score}/5"

    return {
        "score": score,
        "reasoning": reasoning
    }


def main():
    """
    Main function to run the RAG evaluation script.
    """
    print("Starting RAG performance evaluation...")

    try:
        with open(TEST_CASES_FILE, 'r') as f:
            test_cases = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{TEST_CASES_FILE}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file '{TEST_CASES_FILE}'.")
        return

    results = []
    total_duration = 0

    for i, test_case in enumerate(test_cases):
        query = test_case["query"]
        expected_answer = test_case["expected_answer"]
        print(f"\n--- Running Test Case {i+1}/{len(test_cases)} ---")
        print(f"Query: {query}")

        start_time = time.time()
        rag_response = get_rag_response(query)
        duration = time.time() - start_time
        total_duration += duration

        if rag_response and "answer" in rag_response:
            actual_answer = rag_response["answer"]
            print(f"Actual Answer: {actual_answer}")

            # Evaluate Semantic Similarity
            similarity_score = evaluate_semantic_similarity(expected_answer, actual_answer)
            
            # Evaluate with LLM Judge
            judge_evaluation = evaluate_with_llm_judge(query, expected_answer, actual_answer)

            results.append({
                "test_case_id": test_case["id"],
                "query": query,
                "expected_answer": expected_answer,
                "actual_answer": actual_answer,
                "semantic_similarity": similarity_score,
                "llm_judge_score": judge_evaluation["score"],
                "llm_judge_reasoning": judge_evaluation["reasoning"],
                "execution_time_seconds": duration
            })
        else:
            print("Failed to get a valid response from the RAG API.")
            results.append({
                "test_case_id": test_case["id"],
                "query": query,
                "expected_answer": expected_answer,
                "actual_answer": None,
                "error": "No valid response from API"
            })

    # --- Reporting ---
    if results:
        # Calculate aggregate metrics
        successful_tests = [r for r in results if "semantic_similarity" in r]
        avg_similarity = sum(r["semantic_similarity"] for r in successful_tests) / len(successful_tests) if successful_tests else 0
        avg_judge_score = sum(r["llm_judge_score"] for r in successful_tests) / len(successful_tests) if successful_tests else 0
        avg_time = total_duration / len(results)

        # Console Output
        print("\n\n--- Evaluation Summary ---")
        print(f"Total Test Cases: {len(results)}")
        print(f"Successful Runs: {len(successful_tests)}")
        print(f"Average Semantic Similarity: {avg_similarity:.4f}")
        print(f"Average LLM Judge Score: {avg_judge_score:.2f}/5")
        print(f"Average Execution Time: {avg_time:.2f} seconds")
        print("--------------------------")

        # Save detailed report to JSON
        with open(REPORT_FILE, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Detailed report saved to {REPORT_FILE}")

if __name__ == "__main__":
    main()