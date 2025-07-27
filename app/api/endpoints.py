from fastapi import APIRouter, HTTPException, status, Path
from app.api.schemas import (
    DeconstructRequest,
    DeconstructResponse,
    ChatMessageRequest,
    ChatMessageResponse,
    GenerateNotesRequest,
    GenerateQuestionsRequest,
    GenerateQuestionsFromBlueprintDto,
    QuestionSetResponseDto,
    EvaluateAnswerDto,
    EvaluateAnswerResponseDto,
    IndexBlueprintRequest,
    IndexBlueprintResponse,
    IndexingStatsResponse,
    SearchRequest,
    SearchResponse,
    RelatedLocusSearchRequest,
    RelatedLocusSearchResponse,
    ErrorResponse
)
from app.core.deconstruction import deconstruct_text
from app.core.chat import process_chat_message
from app.core.indexing import generate_notes, generate_questions, generate_questions_from_blueprint, evaluate_answer, _call_ai_service_for_evaluation
from app.core.usage_tracker import usage_tracker
from typing import Dict, Any, Optional
from app.core.indexing_pipeline import IndexingPipelineError
import uuid
from datetime import datetime, date

router = APIRouter()


@router.post("/deconstruct", response_model=DeconstructResponse)
async def deconstruct_endpoint(request: DeconstructRequest):
    """
    Deconstruct raw text into a structured LearningBlueprint.
    
    This is the core endpoint that transforms raw educational content into
    a structured JSON blueprint containing knowledge primitives and relationships.
    """
    try:
        # Use the actual deconstruction logic
        blueprint = await deconstruct_text(request.source_text, request.source_type_hint)
        
        return DeconstructResponse(
            blueprint_id=blueprint.source_id,
            source_text=request.source_text,
            blueprint_json=blueprint.model_dump(),
            created_at=datetime.utcnow().isoformat(),
            status="completed"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Deconstruction failed: {str(e)}"
        )


@router.post("/chat/message", response_model=ChatMessageResponse)
async def chat_endpoint(request: ChatMessageRequest):
    """
    Process a chat message with the AI assistant.
    
    Provides intelligent responses using RAG (Retrieval-Augmented Generation)
    based on the user's knowledge base and conversation context.
    """
    try:
        from app.core.query_transformer import QueryTransformer
        from app.core.rag_search import RAGSearchService
        from app.core.context_assembly import ContextAssembler
        from app.core.response_generation import ResponseGenerator, ResponseGenerationRequest
        from app.core.vector_store import create_vector_store
        from app.services.gemini_service import GeminiService
        from app.core.embeddings import GoogleEmbeddingService
        import os
        
        # Initialize services
        vector_store = create_vector_store(
            store_type=os.getenv("VECTOR_STORE_TYPE", "pinecone"),
            api_key=os.getenv("PINECONE_API_KEY", ""),
            environment=os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp"),
            index_name="elevate-ai-main"
        )
        
        embedding_service = GoogleEmbeddingService(
            api_key=os.getenv("GOOGLE_API_KEY", "")
        )
        
        gemini_service = GeminiService()
        
        await vector_store.initialize()
        await embedding_service.initialize()
        
        # Initialize RAG components
        query_transformer = QueryTransformer(embedding_service)
        rag_search_service = RAGSearchService(vector_store, embedding_service)
        context_assembler = ContextAssembler(rag_search_service)
        response_generator = ResponseGenerator(gemini_service)
        
        # Step 1: Transform the user query
        query_transformation = await query_transformer.transform_query(
            query=request.message_content,
            user_context=request.context or {}
        )
        
        # Step 2: Assemble context from all memory tiers
        assembled_context = await context_assembler.assemble_context(
            user_id=request.user_id,
            session_id=request.session_id,
            current_query=request.message_content,
            query_transformation=query_transformation
        )
        
        # Step 3: Add current message to conversation buffer
        context_assembler.add_message_to_buffer(
            session_id=request.session_id,
            role="user",
            content=request.message_content,
            metadata=request.metadata or {}
        )
        
        # Step 4: Generate response using LLM
        response_request = ResponseGenerationRequest(
            user_query=request.message_content,
            query_transformation=query_transformation,
            assembled_context=assembled_context,
            max_tokens=request.max_tokens or 1000,
            temperature=request.temperature or 0.7,
            include_sources=True,
            metadata=request.metadata or {}
        )
        
        generated_response = await response_generator.generate_response(response_request)
        
        # Step 5: Add assistant response to conversation buffer
        context_assembler.add_message_to_buffer(
            session_id=request.session_id,
            role="assistant",
            content=generated_response.content,
            metadata={
                "response_type": generated_response.response_type.value,
                "confidence_score": generated_response.confidence_score,
                "factual_accuracy_score": generated_response.factual_accuracy_score,
                "generation_time_ms": generated_response.generation_time_ms
            }
        )
        
        # Step 6: Update session state with extracted information
        session_updates = context_assembler.extract_session_updates(
            request.message_content,
            generated_response.content
        )
        
        if session_updates:
            context_assembler.update_session_state(request.session_id, session_updates)
        
        # Step 7: Format retrieved context for response
        retrieved_context = []
        for result in assembled_context.retrieved_knowledge[:5]:  # Top 5 results
            retrieved_context.append({
                "source_id": result.blueprint_id,
                "content": result.content,
                "locus_type": result.locus_type,
                "relevance_score": result.final_score,
                "metadata": result.metadata
            })
        
        # Track usage
        usage_tracker.track_request(
            endpoint="chat_message",
            user_id=request.user_id,
            tokens_used=generated_response.token_count,
            model_used="gemini",
            cost_estimate=generated_response.token_count * 0.000001  # Rough estimate
        )
        
        return ChatMessageResponse(
            role="assistant",
            content=generated_response.content,
            retrieved_context=retrieved_context,
            metadata={
                "response_type": generated_response.response_type.value,
                "tone_style": generated_response.tone_style.value,
                "confidence_score": generated_response.confidence_score,
                "factual_accuracy_score": generated_response.factual_accuracy_score,
                "context_quality_score": assembled_context.context_quality_score,
                "assembly_time_ms": assembled_context.assembly_time_ms,
                "generation_time_ms": generated_response.generation_time_ms,
                "total_context_tokens": assembled_context.total_tokens,
                "response_tokens": generated_response.token_count,
                "sources_count": len(generated_response.sources),
                "query_intent": query_transformation.intent.value,
                "query_expanded": query_transformation.expanded_query
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat processing failed: {str(e)}"
        )


@router.post("/generate/notes")
async def generate_notes_endpoint(request: GenerateNotesRequest):
    """
    Generate personalized notes from a LearningBlueprint.
    
    Creates tailored notes based on the blueprint content and user preferences.
    """
    try:
        # TODO: Implement actual note generation logic
        note_id = str(uuid.uuid4())
        
        return {
            "note_id": note_id,
            "name": request.name,
            "content": "Placeholder note content",
            "blueprint_id": request.blueprint_id,
            "folder_id": request.folder_id,
            "created_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Note generation failed: {str(e)}"
        )


@router.post("/generate/questions")
async def generate_questions_endpoint(request: GenerateQuestionsRequest):
    """
    Generate question sets from a LearningBlueprint.
    
    Creates personalized questions for spaced repetition and assessment.
    """
    try:
        # TODO: Implement actual question generation logic
        question_set_id = str(uuid.uuid4())
        
        return {
            "question_set_id": question_set_id,
            "name": request.name,
            "questions": [],
            "blueprint_id": request.blueprint_id,
            "folder_id": request.folder_id,
            "created_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Question generation failed: {str(e)}"
        )


@router.post("/ai-rag/learning-blueprints/{blueprint_id}/question-sets", response_model=QuestionSetResponseDto)
async def generate_questions_from_blueprint_endpoint(
    request: GenerateQuestionsFromBlueprintDto,
    blueprint_id: str = Path(..., description="ID of the LearningBlueprint to use for question generation")
):
    """
    Generate question sets from a LearningBlueprint.
    
    Creates personalized questions for spaced repetition and assessment based on
    the content of a specific LearningBlueprint.
    """
    try:
        # Call the actual question generation logic
        result = await generate_questions_from_blueprint(
            blueprint_id=blueprint_id,
            name=request.name,
            folder_id=request.folder_id,
            question_options=request.question_options
        )
        
        # Convert the result to QuestionDto objects
        from app.api.schemas import QuestionDto
        
        questions = []
        for q in result.get("questions", []):
            questions.append(QuestionDto(
                text=q.get("text", ""),
                answer=q.get("answer", ""),
                question_type=q.get("question_type", "understand"),
                total_marks_available=q.get("total_marks_available", 1),
                marking_criteria=q.get("marking_criteria", "")
            ))
        
        return QuestionSetResponseDto(
            id=result.get("id", 1),
            name=result.get("name", request.name),
            blueprint_id=result.get("blueprint_id", blueprint_id),
            folder_id=result.get("folder_id"),
            questions=questions,
            created_at=result.get("created_at", datetime.utcnow().isoformat()),
            updated_at=result.get("updated_at", datetime.utcnow().isoformat())
        )
        
    except ValueError as e:
        # Handle validation errors from the DTO
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation error: {str(e)}"
        )
    except HTTPException:
        # Re-raise HTTPExceptions as-is (e.g., from the core function)
        raise
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Question generation failed: {str(e)}"
        )


@router.post("/suggest/inline")
async def inline_suggestions_endpoint(request: Dict[str, Any]):
    """
    Provide real-time suggestions during note-taking.
    
    This is a highly optimized, low-latency endpoint for providing
    suggestions as users type in the note editor.
    """
    try:
        # TODO: Implement actual inline suggestion logic
        return {
            "suggestions": [],
            "links": [],
            "completions": []
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inline suggestions failed: {str(e)}"
        )


@router.post("/ai/evaluate-answer", response_model=EvaluateAnswerResponseDto)
async def evaluate_answer_endpoint(request: EvaluateAnswerDto):
    """
    Evaluate a user's answer to a question using AI.
    
    This endpoint receives complete question context from the Core API and
    evaluates the user's answer against the expected answer and marking criteria.
    """
    try:
        # Extract data from the Core API's payload format
        question_context = request.questionContext
        user_answer = request.userAnswer
        
        # Prepare the payload for the AI evaluation logic
        ai_service_payload = {
            "question_text": question_context.questionText,
            "expected_answer": question_context.expectedAnswer,
            "user_answer": user_answer,
            "question_type": question_context.questionType.lower(),
            "total_marks_available": question_context.marksAvailable,
            "marking_criteria": question_context.markingCriteria or "",
            "context": {
                "question_set_name": request.context.questionSetName if request.context else "",
                "folder_name": request.context.folderName if request.context else "",
                "blueprint_title": request.context.questionSetName if request.context else ""
            }
        }
        
        # Call the AI evaluation logic directly (bypassing the mock data lookup)
        evaluation_data = await _call_ai_service_for_evaluation(ai_service_payload)
        
        # Format the response
        return EvaluateAnswerResponseDto(
            corrected_answer=evaluation_data.get("corrected_answer", ""),
            marks_available=question_context.marksAvailable,
            marks_achieved=evaluation_data.get("marks_achieved", 0),
            feedback=evaluation_data.get("feedback", "")
        )
        
    except ValueError as e:
        # Handle validation errors from the DTO
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation error: {str(e)}"
        )
    except HTTPException:
        # Re-raise HTTPExceptions as-is (e.g., from the core function)
        raise
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Answer evaluation failed: {str(e)}"
        )


@router.get("/usage")
async def get_usage_stats(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    Get LLM API usage statistics.
    
    Args:
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)
    """
    try:
        # Parse dates if provided
        start = None
        end = None
        if start_date:
            start = date.fromisoformat(start_date)
        if end_date:
            end = date.fromisoformat(end_date)
        
        # Get usage summary
        summary = usage_tracker.get_usage_summary(start, end)
        
        return {
            "summary": summary,
            "period": {
                "start_date": start_date,
                "end_date": end_date
            }
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid date format: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get usage stats: {str(e)}"
        )


@router.get("/usage/recent")
async def get_recent_usage(limit: int = 50):
    """
    Get recent LLM API usage records.
    
    Args:
        limit: Maximum number of records to return (default: 50)
    """
    try:
        recent_usage = usage_tracker.get_recent_usage(limit)
        return {
            "recent_usage": recent_usage,
            "count": len(recent_usage)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get recent usage: {str(e)}"
        )


@router.post("/index-blueprint", response_model=IndexBlueprintResponse)
async def index_blueprint_endpoint(request: IndexBlueprintRequest):
    """
    Index a LearningBlueprint into the vector database.
    
    Transforms a LearningBlueprint into searchable TextNodes with vector embeddings
    for use in RAG (Retrieval-Augmented Generation) operations.
    
    Uses a translation layer to convert arbitrary blueprint JSON formats into 
    the strict LearningBlueprint Pydantic model for consistent processing.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        from app.models.learning_blueprint import LearningBlueprint
        from app.utils.blueprint_translator import translate_blueprint, BlueprintTranslationError
        
        # ENHANCED DEBUG LOGGING
        blueprint_id = getattr(request, 'blueprint_id', 'unknown')
        logger.info(f"🔍 [DEBUG] Processing index request for blueprint: {blueprint_id}")
        logger.info(f"🔍 [DEBUG] Request blueprint_json keys: {list(request.blueprint_json.keys()) if request.blueprint_json else 'None'}")
        
        # Check for sections and knowledge_primitives
        sections = request.blueprint_json.get('sections', []) if request.blueprint_json else []
        knowledge_primitives = request.blueprint_json.get('knowledge_primitives', {}) if request.blueprint_json else {}
        logger.info(f"🔍 [DEBUG] Sections count: {len(sections)}")
        logger.info(f"🔍 [DEBUG] Knowledge primitives keys: {list(knowledge_primitives.keys()) if knowledge_primitives else 'None'}")
        
        # Translate arbitrary blueprint JSON to LearningBlueprint model
        try:
            logger.info(f"🔍 [DEBUG] Starting blueprint translation...")
            # Ensure the blueprint_id from the request is used as the source_id
            blueprint_json_with_id = request.blueprint_json.copy()
            blueprint_json_with_id['source_id'] = request.blueprint_id
            learning_blueprint = translate_blueprint(blueprint_json_with_id)
            logger.info(f"🔍 [DEBUG] Translation successful - source_id: {learning_blueprint.source_id}")
        except BlueprintTranslationError as e:
            logger.error(f"🔍 [DEBUG] Translation failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Blueprint format error: {str(e)}"
            )
        
        # Extract user_id from blueprint_json or use default
        user_id = request.blueprint_json.get('user_id') or request.blueprint_json.get('userId', 'default')
        logger.info(f"🔍 [DEBUG] User ID: {user_id}")
        
        # Use the indexing pipeline to process the blueprint
        from app.core.indexing_pipeline import IndexingPipeline, IndexingPipelineError
        pipeline = IndexingPipeline()
        
        # Index the blueprint
        logger.info(f"🔍 [DEBUG] Starting indexing pipeline...")
        result = await pipeline.index_blueprint(learning_blueprint)
        logger.info(f"🔍 [DEBUG] Indexing pipeline completed with result: {result}")
        
        # Convert result to response format
        node_count = result.get('processed_nodes', 0)
        from datetime import datetime
        processing_time = result.get('elapsed_seconds', 0.0)
        
        return IndexBlueprintResponse(
            blueprint_id=str(learning_blueprint.source_id),
            blueprint_title=learning_blueprint.source_title,
            indexing_completed=True,
            nodes_processed=node_count,
            embeddings_generated=node_count,  # Assuming 1 embedding per node
            vectors_stored=node_count,        # Assuming all nodes were stored
            success_rate=1.0 if node_count > 0 else 0.0,
            elapsed_seconds=processing_time,
            created_at=datetime.utcnow().isoformat(),
            errors=[]
        )
        
    except HTTPException:
        # Re-raise HTTPExceptions as-is
        raise
    except IndexingPipelineError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing pipeline error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Blueprint indexing failed: {str(e)}"
        )


@router.get("/indexing/stats", response_model=IndexingStatsResponse)
async def get_indexing_stats(blueprint_id: Optional[str] = None):
    """
    Get statistics about indexed content in the vector database.
    
    Returns overall statistics or blueprint-specific statistics if blueprint_id is provided.
    """
    try:
        from app.core.indexing_pipeline import IndexingPipeline, IndexingPipelineError
        
        pipeline = IndexingPipeline()
        stats = await pipeline.get_indexing_stats(blueprint_id)
        
        return IndexingStatsResponse(
            total_nodes=stats.get("total_nodes", 0),
            total_blueprints=stats.get("total_blueprints", 0),
            blueprint_specific=stats.get("blueprint_specific"),
            created_at=datetime.utcnow().isoformat()
        )
        
    except IndexingPipelineError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get indexing stats: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get indexing stats: {str(e)}"
        )


@router.delete("/index-blueprint/{blueprint_id}")
async def delete_blueprint_index(blueprint_id: str):
    """
    Delete all indexed nodes for a specific blueprint.
    
    Removes all TextNodes and vectors associated with the specified blueprint
    from the vector database.
    """
    try:
        from app.core.indexing_pipeline import IndexingPipeline, IndexingPipelineError
        
        pipeline = IndexingPipeline()
        result = await pipeline.delete_blueprint_index(blueprint_id)
        
        return {
            "blueprint_id": blueprint_id,
            "nodes_deleted": result["nodes_deleted"],
            "deletion_completed": result["deletion_completed"],
            "message": f"Successfully deleted {result['nodes_deleted']} nodes for blueprint {blueprint_id}"
        }
        
    except IndexingPipelineError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete blueprint index: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete blueprint index: {str(e)}"
        )


@router.post("/search", response_model=SearchResponse)
async def search_nodes_endpoint(request: SearchRequest):
    """
    Search for TextNodes with metadata filtering.
    
    Performs vector similarity search with advanced metadata filtering including
    locus type, UUE stage, relationships, and content-based filtering.
    """
    try:
        from app.core.search_service import SearchService, SearchServiceError
        from app.core.vector_store import create_vector_store
        from app.core.embeddings import create_embedding_service
        import os
        
        # Initialize services
        vector_store = create_vector_store(
            store_type=os.getenv("VECTOR_STORE_TYPE", "chromadb"),
            api_key=os.getenv("PINECONE_API_KEY", ""),
            environment=os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
        )
        
        embedding_service = create_embedding_service(
            service_type=os.getenv("EMBEDDING_SERVICE_TYPE", "openai"),
            api_key=os.getenv("OPENAI_API_KEY", "")
        )
        
        await vector_store.initialize()
        await embedding_service.initialize()
        
        search_service = SearchService(vector_store, embedding_service)
        result = await search_service.search_nodes(request)
        
        return result
        
    except SearchServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search service error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.get("/search/locus-type/{locus_type}")
async def search_by_locus_type_endpoint(
    locus_type: str,
    blueprint_id: Optional[str] = None,
    limit: int = 50
):
    """
    Search for nodes by specific locus type.
    
    Args:
        locus_type: Type of locus (foundational_concept, use_case, exploration, key_term, common_misconception)
        blueprint_id: Optional blueprint ID to filter by
        limit: Maximum number of results (default: 50)
    """
    try:
        from app.core.search_service import SearchService, SearchServiceError
        from app.core.vector_store import create_vector_store
        from app.core.embeddings import create_embedding_service
        from app.models.text_node import LocusType
        import os
        
        # Validate locus type
        try:
            locus_type_enum = LocusType(locus_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid locus type: {locus_type}. Must be one of: {[e.value for e in LocusType]}"
            )
        
        # Initialize services
        vector_store = create_vector_store(
            store_type=os.getenv("VECTOR_STORE_TYPE", "chromadb"),
            api_key=os.getenv("PINECONE_API_KEY", ""),
            environment=os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
        )
        
        embedding_service = create_embedding_service(
            service_type=os.getenv("EMBEDDING_SERVICE_TYPE", "openai"),
            api_key=os.getenv("OPENAI_API_KEY", "")
        )
        
        await vector_store.initialize()
        await embedding_service.initialize()
        
        search_service = SearchService(vector_store, embedding_service)
        results = await search_service.search_by_locus_type(locus_type_enum, blueprint_id, limit)
        
        return {
            "locus_type": locus_type,
            "blueprint_id": blueprint_id,
            "results": results,
            "total_results": len(results),
            "created_at": datetime.utcnow().isoformat()
        }
        
    except SearchServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search service error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Locus type search failed: {str(e)}"
        )


@router.get("/search/uue-stage/{uue_stage}")
async def search_by_uue_stage_endpoint(
    uue_stage: str,
    blueprint_id: Optional[str] = None,
    limit: int = 50
):
    """
    Search for nodes by UUE stage.
    
    Args:
        uue_stage: UUE stage (understand, use, evaluate)
        blueprint_id: Optional blueprint ID to filter by
        limit: Maximum number of results (default: 50)
    """
    try:
        from app.core.search_service import SearchService, SearchServiceError
        from app.core.vector_store import create_vector_store
        from app.core.embeddings import create_embedding_service
        from app.models.text_node import UUEStage
        import os
        
        # Validate UUE stage
        try:
            uue_stage_enum = UUEStage(uue_stage)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid UUE stage: {uue_stage}. Must be one of: {[e.value for e in UUEStage]}"
            )
        
        # Initialize services
        vector_store = create_vector_store(
            store_type=os.getenv("VECTOR_STORE_TYPE", "chromadb"),
            api_key=os.getenv("PINECONE_API_KEY", ""),
            environment=os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
        )
        
        embedding_service = create_embedding_service(
            service_type=os.getenv("EMBEDDING_SERVICE_TYPE", "openai"),
            api_key=os.getenv("OPENAI_API_KEY", "")
        )
        
        await vector_store.initialize()
        await embedding_service.initialize()
        
        search_service = SearchService(vector_store, embedding_service)
        results = await search_service.search_by_uue_stage(uue_stage_enum, blueprint_id, limit)
        
        return {
            "uue_stage": uue_stage,
            "blueprint_id": blueprint_id,
            "results": results,
            "total_results": len(results),
            "created_at": datetime.utcnow().isoformat()
        }
        
    except SearchServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search service error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"UUE stage search failed: {str(e)}"
        )


@router.post("/search/related-loci", response_model=RelatedLocusSearchResponse)
async def search_related_loci_endpoint(request: RelatedLocusSearchRequest):
    """
    Find loci related to a specific locus through relationships.
    
    Performs graph traversal to find related loci based on relationships,
    with configurable depth and relationship type filtering.
    """
    try:
        from app.core.search_service import SearchService, SearchServiceError
        from app.core.vector_store import create_vector_store
        from app.core.embeddings import create_embedding_service
        import os
        
        # Initialize services
        vector_store = create_vector_store(
            store_type=os.getenv("VECTOR_STORE_TYPE", "chromadb"),
            api_key=os.getenv("PINECONE_API_KEY", ""),
            environment=os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
        )
        
        embedding_service = create_embedding_service(
            service_type=os.getenv("EMBEDDING_SERVICE_TYPE", "openai"),
            api_key=os.getenv("OPENAI_API_KEY", "")
        )
        
        await vector_store.initialize()
        await embedding_service.initialize()
        
        search_service = SearchService(vector_store, embedding_service)
        result = await search_service.find_related_loci(request)
        
        return result
        
    except SearchServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search service error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Related loci search failed: {str(e)}"
        )


@router.get("/search/suggestions")
async def get_search_suggestions_endpoint(
    q: str,
    limit: int = 5
):
    """
    Get search suggestions based on partial query.
    
    Args:
        q: Partial search query
        limit: Maximum number of suggestions (default: 5)
    """
    try:
        from app.core.search_service import SearchService, SearchServiceError
        from app.core.vector_store import create_vector_store
        from app.core.embeddings import create_embedding_service
        import os
        
        if len(q) < 3:
            return {
                "query": q,
                "suggestions": [],
                "message": "Query must be at least 3 characters long"
            }
        
        # Initialize services
        vector_store = create_vector_store(
            store_type=os.getenv("VECTOR_STORE_TYPE", "chromadb"),
            api_key=os.getenv("PINECONE_API_KEY", ""),
            environment=os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
        )
        
        embedding_service = create_embedding_service(
            service_type=os.getenv("EMBEDDING_SERVICE_TYPE", "openai"),
            api_key=os.getenv("OPENAI_API_KEY", "")
        )
        
        await vector_store.initialize()
        await embedding_service.initialize()
        
        search_service = SearchService(vector_store, embedding_service)
        suggestions = await search_service.get_search_suggestions(q, limit)
        
        return {
            "query": q,
            "suggestions": suggestions,
            "created_at": datetime.utcnow().isoformat()
        }
        
    except SearchServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search service error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search suggestions failed: {str(e)}"
        ) 