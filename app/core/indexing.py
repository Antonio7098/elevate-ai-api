"""
Indexing and content generation functionality.

This module handles the blueprint-to-node pipeline for vector database
ingestion and content generation from LearningBlueprints.
"""

import json
import httpx
from typing import Dict, Any, List, Optional
from app.models.learning_blueprint import LearningBlueprint
from app.core.config import settings


async def index_blueprint(blueprint: LearningBlueprint, user_id: str) -> bool:
    """
    Index a LearningBlueprint into the vector database.
    
    This function implements the blueprint-to-node pipeline:
    1. Parse the blueprint structure
    2. Create TextNode objects with rich metadata
    3. Index them in the vector store
    """
    # TODO: Implement blueprint indexing
    # - Iterate through blueprint loci
    # - Create TextNode objects with rich metadata
    # - Index in vector database
    # - Handle relationships and pathways
    
    return True


async def generate_notes(blueprint_id: str, name: str, user_preferences: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate personalized notes from a LearningBlueprint.
    
    This function creates tailored notes based on the blueprint content
    and user preferences.
    """
    # TODO: Implement note generation
    # - Retrieve blueprint from database
    # - Apply user preferences (style, depth, focus areas)
    # - Generate structured notes
    # - Include relevant examples and explanations
    
    return {
        "note_id": "placeholder_id",
        "name": name,
        "content": "Placeholder note content",
        "blueprint_id": blueprint_id
    }


async def generate_questions(blueprint_id: str, name: str, options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate question sets from a LearningBlueprint.
    
    This function creates personalized questions for spaced repetition
    and assessment.
    """
    # TODO: Implement question generation
    # - Retrieve blueprint from database
    # - Generate questions based on knowledge primitives
    # - Apply difficulty and scope options
    # - Create spaced repetition schedule
    
    return {
        "question_set_id": "placeholder_id",
        "name": name,
        "questions": [],
        "blueprint_id": blueprint_id
    }


async def generate_questions_from_blueprint(
    blueprint_id: str, 
    name: str, 
    folder_id: Optional[int] = None,
    question_options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate question sets from a LearningBlueprint using the internal AI service.
    
    This function implements the actual question generation logic by:
    1. Retrieving the LearningBlueprint from the database
    2. Calling the internal AI service to generate questions
    3. Formatting and returning the response
    
    Args:
        blueprint_id: ID of the LearningBlueprint to use
        name: Name for the generated question set
        folder_id: Optional folder ID to store the question set
        question_options: Optional parameters to guide question generation
        
    Returns:
        Dict containing the generated question set data
        
    Raises:
        HTTPException: If the AI service fails or blueprint is not found
    """
    try:
        # Retrieve blueprint from database
        blueprint_data = await _get_blueprint_data(blueprint_id)
        
        # Check if we found actual blueprint data (not just placeholder)
        if not blueprint_data.get("source_text") or blueprint_data.get("source_text") == "Sample source text about mitochondria and cellular biology.":
            # This means we couldn't find the actual blueprint
            from fastapi import HTTPException, status
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"LearningBlueprint not found: {blueprint_id}"
            )
        
        # Prepare the request payload for the internal AI service
        # The AI service expects both the blueprint_json and source_text
        ai_service_payload = {
            "blueprint_json": blueprint_data,
            "source_text": blueprint_data.get("source_text", ""),
            "question_options": question_options or {}
        }
        
        # Call the internal AI service
        questions_data = await _call_ai_service_for_questions(ai_service_payload)
        
        # Format the response
        return {
            "id": 1,  # TODO: Generate from database
            "name": name,
            "blueprint_id": blueprint_id,
            "folder_id": folder_id,
            "questions": questions_data.get("questions", []),
            "created_at": "2025-01-27T10:00:00.000Z",  # TODO: Use actual timestamp
            "updated_at": "2025-01-27T10:00:00.000Z"   # TODO: Use actual timestamp
        }
        
    except HTTPException:
        # Re-raise HTTPExceptions as-is (e.g., 404 Not Found)
        raise
    except Exception as e:
        # Handle unexpected errors
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Question generation failed: {str(e)}"
        )


async def _get_blueprint_data(blueprint_id: str) -> Dict[str, Any]:
    """
    Retrieve LearningBlueprint data from the database.
    
    This function should retrieve the actual blueprint data from the database
    using the blueprint_id. For now, we'll try to load from local files
    and fall back to a placeholder if not found.
    """
    # TODO: Implement actual database retrieval
    # For now, try to load from local deconstruction files
    import os
    from pathlib import Path
    
    deconstructions_dir = Path("deconstructions")
    if deconstructions_dir.exists():
        for file_path in deconstructions_dir.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    if data.get("blueprint_id") == blueprint_id:
                        # Return the complete blueprint data including source_text
                        return {
                            "source_id": blueprint_id,
                            "source_text": data.get("source_text", ""),
                            "blueprint_json": data.get("blueprint_json", {}),
                            "source_title": data.get("blueprint_json", {}).get("source_title", ""),
                            "source_type": data.get("blueprint_json", {}).get("source_type", "text"),
                            "source_summary": data.get("blueprint_json", {}).get("source_summary", {}),
                            "sections": data.get("blueprint_json", {}).get("sections", []),
                            "knowledge_primitives": data.get("blueprint_json", {}).get("knowledge_primitives", {})
                        }
            except (json.JSONDecodeError, IOError):
                continue
    
    # Fallback to placeholder if not found
    return {
        "source_id": blueprint_id,
        "source_text": "Sample source text about mitochondria and cellular biology.",
        "source_title": "Sample Learning Blueprint",
        "source_type": "text",
        "source_summary": {
            "core_thesis_or_main_argument": "Mitochondria are the powerhouse of the cell",
            "inferred_purpose": "Educational content about cellular biology"
        },
        "sections": [
            {
                "section_id": "sec_1",
                "section_name": "Introduction to Mitochondria",
                "description": "Basic overview of mitochondrial function"
            }
        ],
        "knowledge_primitives": {
            "key_propositions_and_facts": [
                {
                    "id": "prop_1",
                    "statement": "Mitochondria are the powerhouse of the cell",
                    "supporting_evidence": ["ATP generation", "Cellular respiration"],
                    "sections": ["sec_1"]
                }
            ],
            "key_entities_and_definitions": [
                {
                    "id": "entity_1",
                    "entity": "Mitochondria",
                    "definition": "Organelles that generate most of the cell's supply of ATP",
                    "category": "Concept",
                    "sections": ["sec_1"]
                }
            ],
            "described_processes_and_steps": [],
            "identified_relationships": [],
            "implicit_and_open_questions": []
        }
    }


async def _call_ai_service_for_questions(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate questions using the LLM service.
    
    This function uses the LLM service to generate questions based on
    the LearningBlueprint and source text.
    
    Args:
        payload: The request payload containing blueprint_json, source_text, and question_options
        
    Returns:
        Dict containing the generated questions
        
    Raises:
        Exception: If the LLM service call fails
    """
    try:
        # Extract data from payload
        blueprint_json = payload.get("blueprint_json", {})
        source_text = payload.get("source_text", "")
        question_options = payload.get("question_options", {})
        
        # Import the LLM service and prompt function
        from app.core.llm_service import llm_service, create_question_generation_prompt
        
        # Create the prompt for question generation
        prompt = create_question_generation_prompt(blueprint_json, source_text, question_options)
        
        # Call the LLM service
        response = await llm_service.call_llm(
            prompt, 
            prefer_google=True, 
            operation="generate_questions"
        )
        
        # Parse the JSON response
        try:
            questions_data = json.loads(response.strip())
            
            # Validate the response structure
            if not isinstance(questions_data, list):
                raise ValueError("LLM response is not a list of questions")
            
            # Validate each question has required fields
            validated_questions = []
            for i, question in enumerate(questions_data):
                if not isinstance(question, dict):
                    print(f"Warning: Question {i} is not a dictionary, skipping")
                    continue
                
                # Ensure all required fields are present
                validated_question = {
                    "text": question.get("text", f"Question {i+1}"),
                    "answer": question.get("answer", "Answer not provided"),
                    "question_type": question.get("question_type", "understand"),
                    "total_marks_available": question.get("total_marks_available", 1),
                    "marking_criteria": question.get("marking_criteria", "Marking criteria not provided")
                }
                validated_questions.append(validated_question)
            
            return {"questions": validated_questions}
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse LLM response as JSON: {e}")
            print(f"Raw response: {response}")
            # Fallback to mock questions if LLM response is invalid
            return _generate_fallback_questions(blueprint_json, source_text, question_options)
            
    except Exception as e:
        print(f"LLM question generation failed: {e}")
        # Fallback to mock questions if LLM call fails
        return _generate_fallback_questions(blueprint_json, source_text, question_options)


def _generate_fallback_questions(blueprint_json: Dict[str, Any], source_text: str, question_options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate fallback questions when LLM service is unavailable.
    
    This provides basic question generation based on knowledge primitives
    when the LLM service fails or returns invalid responses.
    """
    # Extract knowledge primitives for fallback questions
    kp = blueprint_json.get("knowledge_primitives", {})
    propositions = kp.get("key_propositions_and_facts", [])
    entities = kp.get("key_entities_and_definitions", [])
    processes = kp.get("described_processes_and_steps", [])
    
    # Get question count
    count = question_options.get("count", 3)
    
    # Generate contextual fallback questions based on actual content
    questions = []
    
    # Question about entities
    if entities and len(questions) < count:
        entity = entities[0]
        questions.append({
            "text": f"What is {entity.get('entity', 'this concept')}?",
            "answer": entity.get('definition', 'Definition not available'),
            "question_type": "understand",
            "total_marks_available": 2,
            "marking_criteria": f"Award 2 marks for a complete definition of {entity.get('entity', 'the concept')}."
        })
    
    # Question about propositions
    if propositions and len(questions) < count:
        prop = propositions[0]
        questions.append({
            "text": f"Explain: {prop.get('statement', 'this statement')}",
            "answer": f"This statement is supported by: {', '.join(prop.get('supporting_evidence', ['various evidence']))}",
            "question_type": "explore",
            "total_marks_available": 3,
            "marking_criteria": "Award 1 mark for explanation, 1 mark for evidence, 1 mark for clarity."
        })
    
    # Question about processes
    if processes and len(questions) < count:
        process = processes[0]
        questions.append({
            "text": f"Describe the process of {process.get('process_name', 'this process')}",
            "answer": f"Steps: {'; '.join(process.get('steps', ['Step details not available']))}",
            "question_type": "use",
            "total_marks_available": 4,
            "marking_criteria": "Award 1 mark per step described correctly."
        })
    
    # Fallback questions if no specific content
    if not questions:
        questions = [
            {
                "text": "What is the main topic of this content?",
                "answer": f"The content covers: {blueprint_json.get('source_title', 'various topics')}",
                "question_type": "understand",
                "total_marks_available": 2,
                "marking_criteria": "Award 2 marks for identifying the main topic correctly."
            },
            {
                "text": "Summarize the key points from this content.",
                "answer": "Key points should include the main concepts and relationships discussed.",
                "question_type": "explore",
                "total_marks_available": 3,
                "marking_criteria": "Award 1 mark per key point identified."
            }
        ]
    
    return {"questions": questions}


async def create_text_nodes(blueprint: LearningBlueprint) -> List[Dict[str, Any]]:
    """
    Create TextNode objects from a LearningBlueprint.
    
    This function converts blueprint knowledge primitives into
    TextNode objects suitable for vector database indexing.
    """
    # TODO: Implement TextNode creation
    # - Extract propositions and facts
    # - Extract entities and definitions
    # - Extract processes and steps
    # - Add metadata (locusId, locusType, uueStage)
    
    return []


async def index_in_vector_store(text_nodes: List[Dict[str, Any]], user_id: str) -> bool:
    """
    Index TextNode objects in the vector database.
    
    This function handles the actual vector database operations
    for storing and indexing the knowledge nodes.
    """
    # TODO: Implement vector database indexing
    # - Initialize vector database connection
    # - Create embeddings for text content
    # - Store nodes with metadata
    # - Handle batch operations
    
    return True


async def evaluate_answer(question_id: int, user_answer: str) -> Dict[str, Any]:
    """
    Evaluate a user's answer to a question using the internal AI service.
    
    This function implements the actual answer evaluation logic by:
    1. Retrieving the question data from the database
    2. Calling the internal AI service to evaluate the answer
    3. Calculating marks achieved and returning the response
    
    Args:
        question_id: ID of the question to evaluate
        user_answer: The answer provided by the user
        
    Returns:
        Dict containing the evaluation results (corrected_answer, marks_available, marks_achieved)
        
    Raises:
        HTTPException: If the AI service fails or question is not found
    """
    try:
        # Retrieve question data from database
        question_data = await _get_question_data(question_id)
        
        # Check if we found actual question data
        if not question_data.get("text"):
            from fastapi import HTTPException, status
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Question not found: {question_id}"
            )
        
        # Prepare the request payload for the internal AI service
        ai_service_payload = {
            "question_text": question_data.get("text", ""),
            "expected_answer": question_data.get("answer", ""),
            "user_answer": user_answer,
            "question_type": question_data.get("question_type", "understand"),
            "total_marks_available": question_data.get("total_marks_available", 1),
            "marking_criteria": question_data.get("marking_criteria", ""),
            "context": {
                "question_set_name": question_data.get("question_set_name", ""),
                "folder_name": question_data.get("folder_name", ""),
                "blueprint_title": question_data.get("blueprint_title", "")
            }
        }
        
        # Call the internal AI service
        evaluation_data = await _call_ai_service_for_evaluation(ai_service_payload)
        
        # Calculate marks achieved (score × marks available, rounded)
        score = evaluation_data.get("score", 0.0)
        marks_available = question_data.get("total_marks_available", 1)
        marks_achieved = round(score * marks_available)
        
        # Format the response
        return {
            "corrected_answer": evaluation_data.get("corrected_answer", ""),
            "marks_available": marks_available,
            "marks_achieved": marks_achieved
        }
        
    except HTTPException:
        # Re-raise HTTPExceptions as-is (e.g., 404 Not Found)
        raise
    except Exception as e:
        # Handle unexpected errors
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Answer evaluation failed: {str(e)}"
        )


async def _get_question_data(question_id: int) -> Dict[str, Any]:
    """
    Retrieve question data from the database.
    
    This function should retrieve the actual question data from the database
    using the question_id. For now, we'll use mock data.
    """
    # TODO: Implement actual database retrieval
    # For now, return mock question data
    return {
        "id": question_id,
        "text": "What is the primary function of mitochondria?",
        "answer": "Mitochondria are the powerhouse of the cell, generating ATP through cellular respiration.",
        "question_type": "understand",
        "total_marks_available": 5,
        "marking_criteria": "Award 1 mark for mentioning 'powerhouse', 1 mark for 'ATP', 1 mark for 'cellular respiration', 1 mark for energy generation, and 1 mark for clarity.",
        "question_set_name": "Sample Question Set",
        "folder_name": "Biology",
        "blueprint_title": "Cellular Biology"
    }


async def _call_ai_service_for_evaluation(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate an answer using the LLM service.
    
    This function uses the LLM service to evaluate a user's answer against
    the expected answer and marking criteria.
    
    Args:
        payload: The request payload containing question and answer data
        
    Returns:
        Dict containing the evaluation results (score, corrected_answer)
        
    Raises:
        Exception: If the LLM service call fails
    """
    try:
        # Import the LLM service and prompt function
        from app.core.llm_service import llm_service, create_answer_evaluation_prompt
        
        # Create the prompt for answer evaluation
        prompt = create_answer_evaluation_prompt(payload)
        
        # Call the LLM service
        response = await llm_service.call_llm(
            prompt, 
            prefer_google=True, 
            operation="evaluate_answer"
        )
        
        # Parse the JSON response
        try:
            evaluation_data = json.loads(response.strip())
            
            # Validate the response structure
            if not isinstance(evaluation_data, dict):
                raise ValueError("LLM response is not a dictionary")
            
            # Ensure required fields are present
            validated_evaluation = {
                "score": evaluation_data.get("score", 0.0),
                "corrected_answer": evaluation_data.get("corrected_answer", ""),
                "feedback": evaluation_data.get("feedback", "")
            }
            
            return validated_evaluation
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse LLM response as JSON: {e}")
            print(f"Raw response: {response}")
            # Fallback to mock evaluation if LLM response is invalid
            return _generate_fallback_evaluation(payload)
            
    except Exception as e:
        print(f"LLM answer evaluation failed: {e}")
        # Fallback to mock evaluation if LLM call fails
        return _generate_fallback_evaluation(payload)


def _generate_fallback_evaluation(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate fallback evaluation when LLM service is unavailable.
    
    This provides basic answer evaluation based on simple text matching
    when the LLM service fails or returns invalid responses.
    """
    question_text = payload.get("question_text", "")
    expected_answer = payload.get("expected_answer", "")
    user_answer = payload.get("user_answer", "")
    marking_criteria = payload.get("marking_criteria", "")
    
    # Simple fallback evaluation based on keyword matching
    expected_keywords = expected_answer.lower().split()
    user_keywords = user_answer.lower().split()
    
    # Calculate a simple score based on keyword overlap
    matching_keywords = set(expected_keywords) & set(user_keywords)
    total_keywords = len(set(expected_keywords))
    
    if total_keywords > 0:
        score = len(matching_keywords) / total_keywords
    else:
        score = 0.0
    
    # Ensure score is between 0 and 1
    score = max(0.0, min(1.0, score))
    
    return {
        "score": score,
        "corrected_answer": expected_answer,
        "feedback": f"Fallback evaluation: {len(matching_keywords)}/{total_keywords} keywords matched."
    } 