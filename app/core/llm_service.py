"""
LLM service for specialist agent functions.

This module handles API calls to OpenAI and Google AI for the deconstruction pipeline.
"""

import json
import asyncio
import re
from typing import Dict, Any, List, Optional
from app.core.config import settings
from app.core.usage_tracker import log_llm_call
import google.generativeai as genai
from openai import AsyncOpenAI


def extract_json_from_response(response_text: str) -> str:
    """Extract JSON content from response, handling markdown code blocks."""
    # Handle None or empty responses
    if not response_text:
        raise ValueError("Empty or None response received from LLM")
    
    # Remove markdown code blocks if present
    json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    match = re.search(json_pattern, response_text)
    
    if match:
        # Extract content from code block
        extracted = match.group(1).strip()
        if not extracted:
            raise ValueError("Empty JSON content in code block")
        return extracted
    else:
        # No code block, return as-is
        stripped = response_text.strip()
        if not stripped:
            raise ValueError("Empty response after stripping")
        return stripped


class LLMService:
    """Service for handling LLM API calls."""
    
    def __init__(self):
        """Initialize LLM clients."""
        # OpenAI setup - disabled for now
        self.openai_client = None
            
        # Google AI setup
        if settings.google_api_key and settings.google_api_key != "your_google_api_key_here":
            genai.configure(api_key=settings.google_api_key)
            # Configure model with timeout settings
            self.google_model = genai.GenerativeModel(
                'gemini-1.5-flash',
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": 2000,
                },
                safety_settings=[
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"}
                ]
            )
        else:
            self.google_model = None
            
        # OpenRouter setup
        if settings.openrouter_api_key and settings.openrouter_api_key != "your_openrouter_api_key_here":
            self.openrouter_client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=settings.openrouter_api_key,
            )
        else:
            self.openrouter_client = None
    
    def _count_tokens(self, text: str, model: str = "gemini-1.5-flash") -> int:
        """Estimate token count for text.
        
        Note: This is an approximation since Google AI doesn't provide exact token counts.
        """
        # Rough approximation: 1 token ≈ 4 characters for English text
        return len(text) // 4
    
    async def call_openai(self, prompt: str, model: str = "gpt-4o-mini", operation: str = "unknown") -> str:
        """Call OpenAI API."""
        if not self.openai_client:
            raise ValueError("OpenAI API key not configured")
        
        input_tokens = self._count_tokens(prompt)
        
        try:
            response = await self.openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000
            )
            
            output_tokens = response.usage.completion_tokens if hasattr(response.usage, 'completion_tokens') else self._count_tokens(response.choices[0].message.content)
            
            # Log usage
            log_llm_call(
                model=model,
                provider="openai",
                operation=operation,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                success=True
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            # Log failed call
            log_llm_call(
                model=model,
                provider="openai",
                operation=operation,
                input_tokens=input_tokens,
                output_tokens=0,
                success=False,
                error_message=str(e)
            )
            raise Exception(f"OpenAI API call failed: {str(e)}")
    
    async def call_google_ai(self, prompt: str, operation: str = "unknown", max_retries: int = 2) -> str:
        """Call Google AI API with retry logic."""
        if not self.google_model:
            raise ValueError("Google AI API key not configured")
        
        input_tokens = self._count_tokens(prompt)
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Add timeout configuration for the API call
                loop = asyncio.get_event_loop()
                response = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: self.google_model.generate_content(
                            prompt,
                            generation_config={
                                "temperature": 0.1,
                                "max_output_tokens": 2000,
                            }
                        )
                    ),
                    timeout=30.0  # 30 second timeout
                )
                
                # Check if response has parts and text content
                if not hasattr(response, 'parts') or not response.parts:
                    raise Exception("Google AI returned response with no parts")
                
                # Check if response has text content
                try:
                    response_text_content = response.text
                    if not response_text_content:
                        raise Exception("Google AI returned empty response")
                except (IndexError, AttributeError) as e:
                    raise Exception(f"Google AI returned malformed response: {str(e)}")
                
                # Extract JSON from response (handles markdown code blocks)
                try:
                    response_text = extract_json_from_response(response_text_content)
                    
                    # Validate that we have non-empty content
                    if not response_text or response_text.strip() == "":
                        raise Exception("Extracted response text is empty")
                        
                    output_tokens = self._count_tokens(response_text)
                    
                    # Log usage
                    log_llm_call(
                        model="gemini-1.5-flash",
                        provider="google",
                        operation=operation,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        success=True
                    )
                    
                    return response_text
                    
                except Exception as extract_error:
                    # Store error for potential retry
                    last_error = extract_error
                    if attempt < max_retries - 1:
                        print(f"Attempt {attempt + 1} failed with extraction error: {extract_error}. Retrying...")
                        await asyncio.sleep(1)  # Brief delay before retry
                        continue
                    else:
                        # Log failed extraction on final attempt
                        log_llm_call(
                            model="gemini-1.5-flash",
                            provider="google",
                            operation=operation,
                            input_tokens=input_tokens,
                            output_tokens=0,
                            success=False,
                            error_message=f"JSON extraction failed after {max_retries} attempts: {str(extract_error)}"
                        )
                        raise Exception(f"Failed to extract valid response after {max_retries} attempts: {str(extract_error)}")
                        
            except asyncio.TimeoutError as e:
                # Store error for potential retry
                last_error = e
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} timed out. Retrying...")
                    await asyncio.sleep(2)  # Longer delay for timeout
                    continue
                else:
                    # Log failed call on final attempt
                    log_llm_call(
                        model="gemini-1.5-flash",
                        provider="google",
                        operation=operation,
                        input_tokens=input_tokens,
                        output_tokens=0,
                        success=False,
                        error_message=f"Request timeout after {max_retries} attempts"
                    )
                    raise Exception(f"Google AI API call failed: Request timeout after {max_retries} attempts")
                    
            except Exception as e:
                # Store error for potential retry
                last_error = e
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed with error: {e}. Retrying...")
                    await asyncio.sleep(1)  # Brief delay before retry
                    continue
                else:
                    # Log failed call on final attempt
                    log_llm_call(
                        model="gemini-1.5-flash",
                        provider="google",
                        operation=operation,
                        input_tokens=input_tokens,
                        output_tokens=0,
                        success=False,
                        error_message=f"Failed after {max_retries} attempts: {str(e)}"
                    )
                    raise Exception(f"Google AI API call failed after {max_retries} attempts: {str(e)}")
        
        # If all retries failed, fall back to OpenRouter
        print(f"Google AI call failed after {max_retries} attempts. Error: {last_error}. Falling back to OpenRouter.")
        
        if self.openrouter_client:
            try:
                return await self.call_openrouter_ai(prompt, operation=operation)
            except Exception as openrouter_error:
                raise Exception(f"Google AI and OpenRouter fallback both failed. Google error: {last_error}, OpenRouter error: {openrouter_error}")
        else:
            raise Exception(f"Google AI API call failed after {max_retries} attempts and OpenRouter is not configured: {str(last_error)}")
    
    async def call_openrouter_ai(self, prompt: str, model: str = "z-ai/glm-4.5-air:free", operation: str = "unknown", max_retries: int = 2) -> str:
        """Call OpenRouter API with retry logic."""
        if not self.openrouter_client:
            raise ValueError("OpenRouter API key not configured")

        input_tokens = self._count_tokens(prompt, model="openrouter/auto") # Use auto for estimation
        last_error = None

        for attempt in range(max_retries):
            try:
                response = await asyncio.wait_for(
                    self.openrouter_client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                        max_tokens=2000,
                    ),
                    timeout=60.0 # Longer timeout for potentially slower models
                )

                response_text = response.choices[0].message.content
                if not response_text or not response_text.strip():
                    raise ValueError("OpenRouter returned an empty response.")

                output_tokens = self._count_tokens(response_text, model="openrouter/auto")
                log_llm_call(
                    model=model,
                    provider="openrouter",
                    operation=operation,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    success=True
                )
                return response_text

            except Exception as e:
                last_error = e
                print(f"OpenRouter call attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                    continue
        
        log_llm_call(
            model=model,
            provider="openrouter",
            operation=operation,
            input_tokens=input_tokens,
            output_tokens=0,
            success=False,
            error_message=f"Failed after {max_retries} attempts: {str(last_error)}"
        )
        raise Exception(f"OpenRouter API call failed after {max_retries} attempts: {str(last_error)}")

    async def call_llm(self, prompt: str, prefer_google: bool = False, operation: str = "unknown") -> str:
        """Call the preferred LLM service."""
        if prefer_google and self.google_model:
            return await self.call_google_ai(prompt, operation)
        elif self.openai_client:
            return await self.call_openai(prompt, operation=operation)
        elif self.google_model:
            return await self.call_google_ai(prompt, operation)
        else:
            raise ValueError("No LLM API keys configured")


# Global LLM service instance
llm_service = LLMService()


def create_section_extraction_prompt(text: str) -> str:
    """Create prompt for section extraction."""
    return f"""
Extract ONLY the explicit section structure from the following text. Do not infer or create sections that are not clearly marked.

Text:
{text}

Return a JSON array of sections with the following structure:
[
  {{
    "section_id": "unique_id",
    "section_name": "Exact section title as written",
    "description": "Brief description of what is explicitly stated in this section",
    "parent_section_id": null
  }}
]

IMPORTANT: Only extract sections that are explicitly marked with headings (like #, ##, ###) or clearly defined boundaries. If no clear structure exists, create a single "Main" section.

Return only the JSON array, no additional text.
"""


def create_proposition_extraction_prompt(text: str, section_name: str) -> str:
    """Create prompt for proposition extraction."""
    return f"""
Extract ONLY explicitly stated propositions and facts from the following text section. Do not infer or add information not directly stated.

Section: {section_name}
Text: {text}

Return a JSON array of propositions with the following structure:
[
  {{
    "id": "unique_id",
    "statement": "Exact statement as written in the text",
    "supporting_evidence": ["explicit evidence mentioned in text"],
    "sections": ["section_id"]
  }}
]

IMPORTANT: 
- Only extract statements that are explicitly stated in the text
- Do not infer conclusions or add information not present
- Use the exact wording from the text when possible
- Only include evidence that is explicitly mentioned

Return only the JSON array, no additional text.
"""


def create_entity_extraction_prompt(text: str, section_name: str) -> str:
    """Create prompt for entity extraction."""
    return f"""
Extract ONLY explicitly defined terms, entities, and their definitions from the following text section. Do not infer definitions or add entities not mentioned.

Section: {section_name}
Text: {text}

Return a JSON array of entities with the following structure:
[
  {{
    "id": "unique_id",
    "entity": "Exact term or entity name as written",
    "definition": "Definition explicitly provided in the text",
    "category": "Person|Organization|Concept|Place|Object",
    "sections": ["section_id"]
  }}
]

CRITICAL REQUIREMENTS:
- Only extract terms that are explicitly defined or explained in the text
- Use the exact definition provided in the text
- Do not infer definitions or add external knowledge
- Only include entities that are actually mentioned
- NEVER return null, undefined, or empty values for any field
- If a term is mentioned but not defined, DO NOT include it
- All fields (entity, definition, category) must have valid string values
- If no entities are found, return an empty array []

Return only the JSON array, no additional text.
"""


def create_process_extraction_prompt(text: str, section_name: str) -> str:
    """Create prompt for process extraction."""
    return f"""
Extract ONLY explicitly described processes and steps from the following text section. Do not infer processes or add steps not mentioned.

Section: {section_name}
Text: {text}

Return a JSON array of processes with the following structure:
[
  {{
    "id": "unique_id",
    "process_name": "Exact name of the process as written",
    "steps": ["exact steps as described in text"],
    "sections": ["section_id"]
  }}
]

IMPORTANT:
- Only extract processes that are explicitly described in the text
- Use the exact steps as written, do not add missing steps
- Do not infer processes or procedures not mentioned
- Only include steps that are explicitly stated

Return only the JSON array, no additional text.
"""


def create_question_extraction_prompt(text: str, section_name: str) -> str:
    """Create prompt for question extraction."""
    return f"""
Extract ONLY explicitly stated questions from the following text section. Do not generate questions or infer what questions might be relevant.

Section: {section_name}
Text: {text}

Return a JSON array of questions with the following structure:
[
  {{
    "id": "unique_id",
    "question": "Exact question as written in the text",
    "sections": ["section_id"]
  }}
]

IMPORTANT:
- Only extract questions that are explicitly written in the text
- Do not generate questions or infer what questions might be relevant
- Do not create questions based on gaps in the content

Return only the JSON array, no additional text.
"""


def create_question_generation_prompt(blueprint_json: Dict[str, Any], source_text: str, question_options: Dict[str, Any]) -> str:
    """Create prompt for generating questions from a LearningBlueprint."""
    
    # Extract key information from blueprint
    knowledge_primitives = blueprint_json.get("knowledge_primitives", {})
    source_title = blueprint_json.get("source_title", "Unknown Topic")
    
    # Get question options with defaults
    count = question_options.get("count", 5)
    difficulty = question_options.get("difficulty", "Medium")
    scope = question_options.get("scope", "All")
    tone = question_options.get("tone", "Neutral")
    question_types = question_options.get("types", ["mixed"])
    
    # Format knowledge primitives for the prompt
    entities = knowledge_primitives.get("key_entities_and_definitions", [])
    propositions = knowledge_primitives.get("key_propositions_and_facts", [])
    processes = knowledge_primitives.get("described_processes_and_steps", [])
    relationships = knowledge_primitives.get("identified_relationships", [])
    
    # Create the prompt
    prompt = f"""
Generate {count} high-quality questions based on the following LearningBlueprint and source text.

TOPIC: {source_title}
DIFFICULTY: {difficulty}
SCOPE: {scope}
TONE: {tone}
QUESTION TYPES: {', '.join(question_types)}

LEARNING BLUEPRINT:
Source Title: {source_title}
Source Type: {blueprint_json.get("source_type", "text")}

Knowledge Primitives:
- Key Entities ({len(entities)}): {[e.get('entity', 'N/A') for e in entities[:5]]}
- Key Propositions ({len(propositions)}): {[p.get('statement', 'N/A')[:50] + '...' for p in propositions[:3]]}
- Processes ({len(processes)}): {[p.get('process_name', 'N/A') for p in processes[:3]]}
- Relationships ({len(relationships)}): {[r.get('relationship_type', 'N/A') for r in relationships[:3]]}

SOURCE TEXT:
{source_text[:1000]}{'...' if len(source_text) > 1000 else ''}

REQUIREMENTS:
1. Generate exactly {count} questions
2. Focus on {scope.lower()} content from the knowledge primitives
3. Use {tone.lower()} tone and {difficulty.lower()} difficulty
4. Include a mix of question types: {', '.join(question_types)}
5. Each question should test understanding of specific concepts from the blueprint
6. Provide detailed, accurate answers based on the source material
7. Include specific marking criteria for each question
8. Use question types: "understand" (basic comprehension), "use" (application), "explore" (analysis)

QUESTION FORMAT:
Return a JSON array with this exact structure:
[
  {{
    "text": "Clear, specific question text",
    "answer": "Detailed, accurate answer based on the source material",
    "question_type": "understand|use|explore",
    "total_marks_available": 2,
    "marking_criteria": "Specific criteria for awarding marks (e.g., 'Award 1 mark for mentioning X, 1 mark for explaining Y')"
  }}
]

IMPORTANT:
- Base all questions on the provided knowledge primitives and source text
- Ensure answers are factually accurate according to the source material
- Make marking criteria specific and objective
- Vary question types to test different levels of understanding
- Focus on the most important concepts from the blueprint

Return only the JSON array, no additional text or explanations.
"""
    
    return prompt


def create_relationship_extraction_prompt(
    propositions: List[Dict[str, Any]],
    entities: List[Dict[str, Any]],
    processes: List[Dict[str, Any]]
) -> str:
    """Create prompt for relationship extraction."""
    return f"""
Extract ONLY explicitly stated relationships between the following knowledge primitives. Do not infer relationships not directly stated.

Propositions: {json.dumps(propositions, indent=2)}
Entities: {json.dumps(entities, indent=2)}
Processes: {json.dumps(processes, indent=2)}

Return a JSON array of relationships with the following structure:
[
  {{
    "id": "unique_id",
    "relationship_type": "causal|part-of|component-of|similar-to|opposes",
    "source_primitive_id": "id_of_source_primitive",
    "target_primitive_id": "id_of_target_primitive",
    "description": "Exact description of the relationship as stated in text",
    "sections": ["section_id"]
  }}
]

IMPORTANT:
- Only extract relationships that are explicitly stated in the original text
- Do not infer relationships based on general knowledge
- Use the exact wording from the text when describing relationships
- Only include relationships that are directly mentioned

Return only the JSON array, no additional text.
"""


def create_answer_evaluation_prompt(payload: Dict[str, Any]) -> str:
    """Create prompt for answer evaluation (marks_achieved, corrected_answer, feedback only)."""
    question_text = payload.get("question_text", "")
    expected_answer = payload.get("expected_answer", "")
    user_answer = payload.get("user_answer", "")
    question_type = payload.get("question_type", "understand")
    total_marks_available = payload.get("total_marks_available", 1)
    marking_criteria = payload.get("marking_criteria", "")
    context = payload.get("context", {})
    
    # Build context information
    context_info = ""
    if context:
        context_info = f"""
CONTEXT:
- Question Set: {context.get('question_set_name', 'Unknown')}
- Folder: {context.get('folder_name', 'Unknown')}
- Blueprint: {context.get('blueprint_title', 'Unknown')}
"""
    
    # Create the prompt
    prompt = f"""
You are an expert educational assessor. Evaluate the user's answer against the expected answer and marking criteria.

QUESTION:
{question_text}

EXPECTED ANSWER:
{expected_answer}

USER'S ANSWER:
{user_answer}

QUESTION TYPE: {question_type}
TOTAL MARKS AVAILABLE: {total_marks_available}
MARKING CRITERIA:
{marking_criteria}{context_info}

TASK:
1. Compare the user's answer with the expected answer
2. Apply the marking criteria to determine the number of marks achieved (integer, 0 to {total_marks_available})
3. Provide a corrected/improved version of the answer
4. Give feedback that is specific, constructive, and encouraging
5. If the answer is incorrect or incomplete, explain what is missing and give a tip or suggestion for improvement
6. Always end your feedback with a positive or motivating remark

EXAMPLES OF ENCOURAGING FEEDBACK:
- "Your answer shows a good attempt! To improve, try to mention that energy cannot be created or destroyed, and that it can only change forms. Keep going—you're on the right track!"
- "Nice effort! You included some key points, but remember to explain that the total energy in a closed system remains constant. You're making progress—keep it up!"

FEEDBACK STYLE REQUIREMENTS:
- ALWAYS start with a positive acknowledgment (e.g., "Good try!", "Nice effort!", "You're on the right track!")
- Be specific about what they did well, even if the answer is mostly wrong
- Provide clear, actionable suggestions for improvement
- Use encouraging language throughout
- ALWAYS end with a motivating phrase (e.g., "Keep going!", "You're making progress!", "Great work on trying!")

Return a JSON object with the following structure:
{{
  "marks_achieved": <integer between 0 and {total_marks_available}>,
  "corrected_answer": "The ideal/correct answer",
  "feedback": "Encouraging, constructive feedback explaining the marks, suggestions for improvement, and ending with a positive remark."
}}

IMPORTANT:
- Only return marks_achieved (integer, 0 to {total_marks_available}), corrected_answer (string), and feedback (string)
- Do NOT return a score or percentage
- Be fair and consistent in your evaluation
- Provide specific, actionable, and encouraging feedback
- The corrected answer should be comprehensive and accurate
- Feedback should always include a positive or motivating remark at the end

Return only the JSON object, no additional text.
"""
    return prompt 


def create_encouraging_feedback_prompt(evaluation_result: Dict[str, Any]) -> str:
    """Create prompt for generating encouraging feedback based on evaluation results."""
    marks_achieved = evaluation_result.get("marks_achieved", 0)
    marks_available = evaluation_result.get("marks_available", 1)
    question_text = evaluation_result.get("question_text", "")
    user_answer = evaluation_result.get("user_answer", "")
    corrected_answer = evaluation_result.get("corrected_answer", "")
    
    # Determine the score percentage
    score_percentage = (marks_achieved / marks_available) * 100 if marks_available > 0 else 0
    
    prompt = f"""
You are a supportive and encouraging educational mentor. Generate a brief, encouraging feedback message for a student based on their performance.

QUESTION: {question_text}
STUDENT'S ANSWER: {user_answer}
CORRECT ANSWER: {corrected_answer}
MARKS ACHIEVED: {marks_achieved}/{marks_available} ({score_percentage:.0f}%)

TASK:
Generate a short, encouraging feedback message (1-2 sentences) that:
- Acknowledges their effort positively
- Provides specific, actionable improvement suggestions
- Ends with a motivating phrase
- Is warm, supportive, and educational

EXAMPLES OF ENCOURAGING FEEDBACK:
- "Great effort on this question! To improve, try to mention that energy cannot be created or destroyed, and that it can only change forms. Keep going—you're on the right track!"
- "Nice try! You included some key points, but remember to explain that the total energy in a closed system remains constant. You're making progress—keep it up!"
- "Good attempt! Focus on explaining both energy conservation and transformation. Every practice helps you learn—don't give up!"

Return only the encouraging feedback message as a string, no JSON formatting.
"""
    return prompt 