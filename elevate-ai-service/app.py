"""
Elevate AI Service

This Flask application implements the AI Service for the Elevate platform.
It provides endpoints for evaluating user answers using LLM technology.
"""

from flask import Flask, request, jsonify
import time
import os
import logging
import json
from typing import Dict, List, Any, Optional, Union
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
CORE_API_ACCESS_KEY = os.environ.get("CORE_API_ACCESS_KEY", "your-core-api-access-key-here")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
MODEL_NAME = os.environ.get("AI_MODEL", "gemini-1.5-flash-latest")

# Configure Gemini API
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not found in environment variables")
else:
    logger.info(f"Configuring Gemini API with key: {GEMINI_API_KEY[:5]}...{GEMINI_API_KEY[-5:]}")
genai.configure(api_key=GEMINI_API_KEY)

# Middleware for API key verification
@app.before_request
def verify_api_key():
    if request.endpoint != 'health':  # Skip auth for health check
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({"success": False, "error": {"code": "unauthorized", "message": "No API key provided"}}), 401
        
        try:
            scheme, token = auth_header.split()
            if scheme.lower() != 'bearer':
                return jsonify({"success": False, "error": {"code": "unauthorized", "message": "Invalid authentication scheme"}}), 401
            
            if token != CORE_API_ACCESS_KEY:
                return jsonify({"success": False, "error": {"code": "unauthorized", "message": "Invalid API key"}}), 401
        except ValueError:
            return jsonify({"success": False, "error": {"code": "unauthorized", "message": "Invalid authorization header format"}}), 401

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "version": "v1"}), 200

# Evaluate Answer endpoint
@app.route('/evaluate-answer', methods=['POST'])
def evaluate_answer():
    start_time = time.time()
    
    try:
        data = request.json
        
        # Validate required fields
        if not data.get('questionContext'):
            return jsonify({
                "success": False,
                "error": {
                    "code": "invalid_request",
                    "message": "Question context is required"
                }
            }), 400
        
        if 'userAnswer' not in data:
            return jsonify({
                "success": False,
                "error": {
                    "code": "invalid_request",
                    "message": "User answer is required"
                }
            }), 400
        
        # Extract request parameters
        question_context = data.get('questionContext', {})
        user_answer = data.get('userAnswer', '')
        
        # Extract question details
        question_id = question_context.get('questionId', '')
        question_text = question_context.get('questionText', '')
        expected_answer = question_context.get('expectedAnswer', '')
        question_type = question_context.get('questionType', '')
        
        # Log the incoming request
        logger.info(f"Evaluating answer for question ID: {question_id}")
        
        # Create prompt for the LLM
        prompt = create_evaluation_prompt(question_text, expected_answer, user_answer, question_type)
        
        # Call the LLM to evaluate the answer
        evaluation = call_llm_for_evaluation(prompt)
        
        processing_time = time.time() - start_time
        
        return jsonify({
            "success": True,
            "evaluation": evaluation,
            "metadata": {
                "processingTime": f"{processing_time:.2f}s",
                "model": MODEL_NAME,
                "questionId": question_id
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error evaluating answer: {str(e)}")
        return jsonify({
            "success": False,
            "error": {
                "code": "internal_error",
                "message": "An internal error occurred while processing your request"
            }
        }), 500

def create_evaluation_prompt(question_text: str, expected_answer: str, user_answer: str, question_type: str) -> str:
    """
    Create a prompt for the LLM to evaluate the user's answer.
    
    Args:
        question_text: The original question text
        expected_answer: The expected or ideal answer
        user_answer: The user's submitted answer
        question_type: The type of question (multiple-choice, true-false, short-answer)
        
    Returns:
        A formatted prompt string for the LLM
    """
    return f"""You are an educational AI assistant tasked with evaluating student answers.

QUESTION: {question_text}

EXPECTED ANSWER: {expected_answer}

USER'S ANSWER: {user_answer}

QUESTION TYPE: {question_type}

Please evaluate the user's answer based on the expected answer and provide the following in JSON format:
1. isCorrect: A boolean indicating if the answer is completely correct (true) or not (false)
2. isPartiallyCorrect: A boolean indicating if the answer has some correct elements but is not fully correct
3. score: A number between 0.0 and 1.0 representing the correctness of the answer
4. feedbackText: Detailed feedback explaining what was correct and/or incorrect about the user's answer
5. suggestedCorrectAnswer: The ideal answer, potentially expanded from the expected answer for clarity

Your evaluation should be fair, educational, and helpful. For multiple-choice and true-false questions, 
be strict about exact matches. For short-answer questions, focus on key concepts and semantic meaning 
rather than exact wording.

Return your response in the following JSON format:
{{
  "isCorrect": true/false,
  "isPartiallyCorrect": true/false,
  "score": 0.0-1.0,
  "feedbackText": "Your detailed feedback here",
  "suggestedCorrectAnswer": "The ideal answer here"
}}
"""

def call_llm_for_evaluation(prompt: str) -> Dict[str, Any]:
    """
    Call the LLM to evaluate the user's answer using Google's Gemini API.
    
    Args:
        prompt: The formatted prompt for the LLM
        
    Returns:
        A dictionary containing the evaluation results
    """
    try:
        # Create a system prompt to instruct Gemini
        system_prompt = "You are an educational AI assistant that evaluates student answers. You MUST return your response in valid JSON format. The JSON must include isCorrect (boolean), isPartiallyCorrect (boolean), score (number between 0 and 1), feedbackText (string), and suggestedCorrectAnswer (string)."
        
        # Instantiate the Gemini model
        model = genai.GenerativeModel(MODEL_NAME)
        
        # Configure generation parameters
        generation_config = genai.GenerationConfig(
            temperature=0.2,  # Lower temperature for more consistent evaluations
            max_output_tokens=1000
        )
        
        # Combine system prompt and user prompt
        full_prompt = f"{system_prompt}\n\n{prompt}"
        
        # Call Gemini API
        response = model.generate_content(
            full_prompt,
            generation_config=generation_config
        )
        
        # Extract the content from the response
        content = response.text
        logger.info(f"Raw Gemini response: {content[:100]}...")
        
        # Try to extract JSON from the response text
        # Look for JSON content between curly braces
        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(0)
            # Parse the JSON response
            evaluation = json.loads(json_str)
        else:
            # If no JSON found, create a structured response manually
            logger.warning("No JSON found in Gemini response, creating structured response manually")
            evaluation = {
                "isCorrect": False,
                "isPartiallyCorrect": True,
                "score": 0.5,
                "feedbackText": content,
                "suggestedCorrectAnswer": ""
            }
        
        # Ensure all required fields are present
        required_fields = ["isCorrect", "isPartiallyCorrect", "score", "feedbackText", "suggestedCorrectAnswer"]
        for field in required_fields:
            if field not in evaluation:
                evaluation[field] = "" if field in ["feedbackText", "suggestedCorrectAnswer"] else False if field in ["isCorrect", "isPartiallyCorrect"] else 0.0
        
        return evaluation
        
    except Exception as e:
        logger.error(f"Error calling LLM: {str(e)}")
        # Return a default evaluation in case of error
        return {
            "isCorrect": False,
            "isPartiallyCorrect": False,
            "score": 0.0,
            "feedbackText": "Sorry, we encountered an error while evaluating your answer. Please try again.",
            "suggestedCorrectAnswer": ""
        }

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port, debug=True)
