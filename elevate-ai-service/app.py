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

# Generate Questions endpoint
@app.route('/api/generate-questions', methods=['POST'])
@app.route('/generate-questions', methods=['POST'])
def generate_questions():
    start_time = time.time()
    
    try:
        data = request.json
        
        # Validate required fields
        if not data.get('sourceText'):
            return jsonify({
                "success": False,
                "error": {
                    "code": "invalid_request",
                    "message": "Source text is required"
                }
            }), 400
        
        if not data.get('questionCount') or not isinstance(data.get('questionCount'), int):
            return jsonify({
                "success": False,
                "error": {
                    "code": "invalid_request",
                    "message": "Question count must be a positive integer"
                }
            }), 400
        
        # Extract request parameters
        source_text = data.get('sourceText')
        question_count = min(data.get('questionCount'), 10)  # Limit to 10 questions
        question_types = data.get('questionTypes', ["multiple-choice", "true-false", "short-answer"])
        difficulty = data.get('difficulty', 'medium')
        topics = data.get('topics', [])
        language = data.get('language', 'en')
        
        # Check if source text is too short
        if len(source_text) < 50:
            return jsonify({
                "success": False,
                "error": {
                    "code": "text_too_short",
                    "message": "Source text is too short to generate meaningful questions"
                }
            }), 400
        
        # Create prompt for the LLM
        prompt = create_questions_prompt(source_text, question_count, question_types, difficulty, topics, language)
        
        # Call the LLM to generate questions
        questions = call_llm_for_questions(prompt)
        
        processing_time = time.time() - start_time
        
        return jsonify({
            "success": True,
            "questions": questions,
            "metadata": {
                "processingTime": f"{processing_time:.2f}s",
                "model": MODEL_NAME,
                "sourceTextLength": len(source_text)
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error generating questions: {str(e)}")
        return jsonify({
            "success": False,
            "error": {
                "code": "internal_error",
                "message": "An internal error occurred while processing your request"
            }
        }), 500

# Chat endpoint
@app.route('/api/chat', methods=['POST'])
@app.route('/chat', methods=['POST'])
def chat():
    start_time = time.time()
    
    try:
        data = request.json
        
        # Validate required fields
        if not data.get('message'):
            return jsonify({
                "success": False,
                "error": {
                    "code": "invalid_request",
                    "message": "Message is required"
                }
            }), 400
        
        # Extract request parameters
        message = data.get('message')
        conversation = data.get('conversation', [])
        context = data.get('context', {})
        language = data.get('language', 'en')
        
        # Check if context is too large
        if context and len(str(context)) > 10000:  # Arbitrary limit for demonstration
            return jsonify({
                "success": False,
                "error": {
                    "code": "context_too_large",
                    "message": "The provided context is too large. Please reduce the amount of context data."
                }
            }), 400
        
        # Create prompt for the LLM
        prompt = create_chat_prompt(message, conversation, context, language)
        
        # Call the LLM to generate a response
        response = call_llm_for_chat(prompt, context)
        
        processing_time = time.time() - start_time
        
        return jsonify({
            "success": True,
            "response": response,
            "metadata": {
                "processingTime": f"{processing_time:.2f}s",
                "model": MODEL_NAME,
                "tokensUsed": len(prompt.split()) + len(str(response).split())  # Simplified token calculation
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error generating chat response: {str(e)}")
        return jsonify({
            "success": False,
            "error": {
                "code": "internal_error",
                "message": "An internal error occurred while processing your request"
            }
        }), 500

# Helper functions for generating questions
def create_questions_prompt(source_text: str, question_count: int, question_types: List[str], 
                          difficulty: str, topics: List[str], language: str) -> str:
    """Create a prompt for generating questions from source text."""
    topics_str = ", ".join(topics) if topics else "any relevant topics"
    question_types_str = ", ".join(question_types)
    
    return f"""You are an educational AI assistant tasked with generating questions from source text.

SOURCE TEXT:
{source_text}

GENERATE {question_count} QUESTIONS with the following parameters:
- Question types: {question_types_str}
- Difficulty level: {difficulty}
- Topics to focus on: {topics_str}
- Language: {language}

For each question, provide:
1. The question text
2. The question type (one of: {question_types_str})
3. The correct answer
4. For multiple-choice questions, provide 4 options including the correct answer
5. A brief explanation of the answer

Return your response as a valid JSON array of question objects, with each object having the following structure:
- For multiple-choice questions:
  {{"text": "question text", "questionType": "multiple-choice", "answer": "correct answer", "options": ["option1", "option2", "option3", "option4"], "explanation": "explanation text"}}
- For true-false questions:
  {{"text": "statement text", "questionType": "true-false", "answer": "true or false", "explanation": "explanation text"}}
- For short-answer questions:
  {{"text": "question text", "questionType": "short-answer", "answer": "correct answer", "explanation": "explanation text"}}

Ensure that the questions are diverse, educational, and directly based on the content of the source text.
"""

def call_llm_for_questions(prompt: str) -> List[Dict[str, Any]]:
    """Call the LLM to generate questions from source text."""
    try:
        # Create a system prompt to instruct Gemini
        system_prompt = "You are an educational AI assistant that generates questions from source text. You MUST return your response as a valid JSON array of question objects."
        
        # Instantiate the Gemini model
        model = genai.GenerativeModel(MODEL_NAME)
        
        # Configure generation parameters
        generation_config = genai.GenerationConfig(
            temperature=0.7,  # Higher temperature for more creative questions
            max_output_tokens=2000
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
        logger.info(f"Raw Gemini response for questions: {content[:100]}...")
        
        # Try to extract JSON from the response text
        # Look for JSON content between square brackets
        import re
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(0)
            # Parse the JSON response
            questions = json.loads(json_str)
        else:
            # If no JSON found, create a structured response manually
            logger.warning("No JSON found in Gemini response for questions, creating structured response manually")
            questions = []
        
        # Ensure we have the requested number of questions
        return questions
        
    except Exception as e:
        logger.error(f"Error calling LLM for questions: {str(e)}")
        # Return an empty list in case of error
        return []

# Helper functions for chat
def create_chat_prompt(message: str, conversation: List[Dict[str, str]], 
                     context: Dict[str, Any], language: str) -> str:
    """Create a prompt for the chat functionality."""
    # Format conversation history
    conversation_str = ""
    for msg in conversation:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        conversation_str += f"{role.upper()}: {content}\n"
    
    # Format context information
    context_str = ""
    if context:
        if 'questionSets' in context:
            context_str += "RELEVANT INFORMATION:\n"
            for question_set in context['questionSets']:
                context_str += f"From {question_set.get('name', 'Unknown')}:\n"
                for question in question_set.get('questions', []):
                    context_str += f"- Question: {question.get('text', '')}\n"
                    context_str += f"  Answer: {question.get('answer', '')}\n"
        
        if 'userLevel' in context:
            context_str += f"\nUser knowledge level: {context['userLevel']}\n"
        
        if 'preferredLearningStyle' in context:
            context_str += f"User preferred learning style: {context['preferredLearningStyle']}\n"
    
    return f"""You are an educational AI assistant engaged in a conversation with a user.

CONVERSATION HISTORY:
{conversation_str}

CONTEXT INFORMATION:
{context_str}

CURRENT USER MESSAGE:
{message}

Please provide a helpful, educational response in {language} language.

Your response should be informative, accurate, and tailored to the user's knowledge level and learning style if specified.

Also suggest 3 follow-up questions the user might want to ask next.

Return your response in JSON format with the following structure:
{{
  "message": "Your detailed response to the user's question",
  "references": [{{
    "text": "Referenced information if applicable",
    "source": "Source of the reference"  
  }}],
  "suggestedQuestions": ["Question 1", "Question 2", "Question 3"]
}}
"""

def call_llm_for_chat(prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Call the LLM to generate a chat response."""
    try:
        # Create a system prompt to instruct Gemini
        system_prompt = "You are an educational AI assistant engaged in a conversation. You MUST return your response in valid JSON format with message, references, and suggestedQuestions fields."
        
        # Instantiate the Gemini model
        model = genai.GenerativeModel(MODEL_NAME)
        
        # Configure generation parameters
        generation_config = genai.GenerationConfig(
            temperature=0.7,  # Balanced temperature for conversational responses
            max_output_tokens=2000
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
        logger.info(f"Raw Gemini response for chat: {content[:100]}...")
        
        # Try to extract JSON from the response text
        # Look for JSON content between curly braces
        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(0)
            # Parse the JSON response
            chat_response = json.loads(json_str)
        else:
            # If no JSON found, create a structured response manually
            logger.warning("No JSON found in Gemini response for chat, creating structured response manually")
            chat_response = {
                "message": content,
                "references": [],
                "suggestedQuestions": [
                    "Can you explain more about this topic?",
                    "What are some practical applications of this?",
                    "How does this relate to other concepts?"
                ]
            }
        
        # Extract references from context if available and not already provided
        if not chat_response.get('references') and context.get('questionSets'):
            references = []
            for question_set in context.get('questionSets', []):
                for question in question_set.get('questions', []):
                    if question.get('text') and question.get('answer'):
                        references.append({
                            "text": question.get('answer'),
                            "source": f"{question_set.get('name', 'Unknown')} Question Set"
                        })
            chat_response['references'] = references[:2]  # Limit to 2 references
        
        # Ensure all required fields are present
        required_fields = ["message", "references", "suggestedQuestions"]
        for field in required_fields:
            if field not in chat_response:
                if field == "references":
                    chat_response[field] = []
                elif field == "suggestedQuestions":
                    chat_response[field] = [
                        "Can you explain more about this topic?",
                        "What are some practical applications of this?",
                        "How does this relate to other concepts?"
                    ]
                else:
                    chat_response[field] = ""
        
        return chat_response
        
    except Exception as e:
        logger.error(f"Error calling LLM for chat: {str(e)}")
        # Return a default response in case of error
        return {
            "message": "I apologize, but I encountered an error while processing your request. Please try again.",
            "references": [],
            "suggestedQuestions": [
                "Can you try asking a different question?",
                "Can you rephrase your question?",
                "Would you like to start a new conversation?"
            ]
        }

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port, debug=True)
