"""
Sample Implementation of the Elevate AI Service

This is a basic Flask application that implements the AI Service Contract.
It demonstrates how the Python service would handle requests from the Core API.

Note: This is a simplified implementation for demonstration purposes.
A production implementation would use a more robust architecture and include
proper error handling, authentication, logging, and integration with actual AI models.
"""

from flask import Flask, request, jsonify
import time
import random
import os
from typing import Dict, List, Any, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
API_KEY = os.environ.get("AI_SERVICE_API_KEY", "test-api-key")
MODEL_NAME = os.environ.get("AI_MODEL", "gpt-4")

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
            
            if token != API_KEY:
                return jsonify({"success": False, "error": {"code": "unauthorized", "message": "Invalid API key"}}), 401
        except ValueError:
            return jsonify({"success": False, "error": {"code": "unauthorized", "message": "Invalid authorization header format"}}), 401

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "version": "v1"}), 200

# Generate questions endpoint
@app.route('/v1/api/generate-questions', methods=['POST'])
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
        
        # Check if source text is too short
        if len(source_text) < 50:
            return jsonify({
                "success": False,
                "error": {
                    "code": "text_too_short",
                    "message": "Source text is too short to generate meaningful questions"
                }
            }), 400
        
        # In a real implementation, this would call an AI model to generate questions
        # For this sample, we'll generate some dummy questions
        questions = []
        for i in range(question_count):
            question_type = random.choice(question_types)
            
            if question_type == "multiple-choice":
                questions.append({
                    "text": f"Sample multiple-choice question {i+1} based on the provided text?",
                    "questionType": "multiple-choice",
                    "answer": "Option A",
                    "options": ["Option A", "Option B", "Option C", "Option D"],
                    "explanation": "This is an explanation for the correct answer."
                })
            elif question_type == "true-false":
                questions.append({
                    "text": f"Sample true-false statement {i+1} based on the provided text.",
                    "questionType": "true-false",
                    "answer": "true" if random.random() > 0.5 else "false",
                    "explanation": "This is an explanation for why the statement is true or false."
                })
            else:  # short-answer
                questions.append({
                    "text": f"Sample short-answer question {i+1} based on the provided text?",
                    "questionType": "short-answer",
                    "answer": "This is a sample answer to the short-answer question.",
                    "explanation": "This is additional context for the answer."
                })
        
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
@app.route('/v1/api/chat', methods=['POST'])
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
        
        # Check if context is too large
        if context and len(str(context)) > 10000:  # Arbitrary limit for demonstration
            return jsonify({
                "success": False,
                "error": {
                    "code": "context_too_large",
                    "message": "The provided context is too large. Please reduce the amount of context data."
                }
            }), 400
        
        # In a real implementation, this would call an AI model to generate a response
        # For this sample, we'll generate a dummy response
        
        # Check if we have question sets in the context to reference
        references = []
        if context.get('questionSets'):
            for question_set in context.get('questionSets'):
                for question in question_set.get('questions', []):
                    if any(keyword in message.lower() for keyword in question.get('text', '').lower().split()):
                        references.append({
                            "text": question.get('answer', ''),
                            "source": f"{question_set.get('name', 'Unknown')} Question Set"
                        })
                        break
        
        # Generate a response based on the message
        response_text = f"This is a simulated AI response to your question: '{message}'. "
        if context.get('userLevel') == 'beginner':
            response_text += "I've tailored this response for a beginner level of understanding. "
        
        # Add some suggested follow-up questions
        suggested_questions = [
            f"Can you explain more about {message.split()[0] if message.split() else 'this topic'}?",
            "How does this relate to other concepts?",
            "What are some practical applications of this?"
        ]
        
        processing_time = time.time() - start_time
        tokens_used = len(message.split()) + len(response_text.split()) + 20  # Simplified token calculation
        
        return jsonify({
            "success": True,
            "response": {
                "message": response_text,
                "references": references,
                "suggestedQuestions": suggested_questions
            },
            "metadata": {
                "processingTime": f"{processing_time:.2f}s",
                "model": MODEL_NAME,
                "tokensUsed": tokens_used
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

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
