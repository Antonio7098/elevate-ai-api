# LangGraph Sequential Generation Workflow

This document describes the new LangGraph-based implementation of the sequential generation workflow, which replaces the custom orchestrator with a modern, robust workflow framework.

## Overview

The LangGraph-based workflow implements the sequential generation pattern:
```
source → blueprint → sections → primitives → mastery criteria → questions
                                    ↓              ↓
                                  notes ←─────── notes
```

## Key Benefits of LangGraph

### 1. **Modern Functional API**
- Uses `@task` decorators for individual workflow steps
- Uses `@entrypoint` for main workflow orchestration
- Cleaner, more maintainable code structure

### 2. **Built-in State Management**
- Automatic state persistence with `InMemorySaver`
- Workflow resumption capabilities
- State checkpointing for long-running workflows

### 3. **Human-in-the-Loop Support**
- Built-in `interrupt()` mechanism for user input
- Automatic workflow pausing and resumption
- User edit integration

### 4. **Production Ready**
- Error handling and recovery
- Workflow monitoring and debugging
- Scalable architecture

## Architecture

### Core Components

#### `SequentialGenerationState`
```python
class SequentialGenerationState(TypedDict):
    workflow_id: str                    # Unique workflow identifier
    current_step: str                   # Current step in workflow
    status: str                         # Workflow status
    source_content: str                 # Original source material
    blueprint: Dict[str, Any]          # Generated learning blueprint
    sections: List[Dict[str, Any]]     # Generated sections
    primitives: List[Dict[str, Any]]   # Knowledge primitives
    mastery_criteria: List[Dict[str, Any]]  # Mastery criteria
    questions: List[Dict[str, Any]]    # Assessment questions
    notes: List[Dict[str, Any]]        # Study notes
    user_edits: List[Dict[str, Any]]   # User modifications
    errors: List[str]                   # Any errors encountered
```

#### Workflow Steps
1. **`initialize_workflow`** - Set up initial state
2. **`generate_blueprint`** - Create learning blueprint from source
3. **`generate_sections`** - Extract and process sections
4. **`extract_primitives`** - Identify knowledge primitives
5. **`generate_mastery_criteria`** - Create learning objectives
6. **`generate_questions`** - Generate assessment questions
7. **`generate_notes`** - Create comprehensive study materials
8. **`complete_workflow`** - Finalize and summarize

### User Interaction Points

#### Human-in-the-Loop with `interrupt()`
```python
@task
async def wait_for_user_review(state: SequentialGenerationState):
    # Pause workflow and wait for user input
    user_feedback = interrupt({
        "action": "Please review the generated content",
        "current_step": state["current_step"],
        "content_summary": {...}
    })
    
    # Process user feedback and continue
    if user_feedback and "edits" in user_feedback:
        state = await apply_user_edits(state, user_feedback["edits"])
    
    return state
```

#### Workflow Resumption
```python
async def resume_workflow(self, workflow_id: str, user_input: Dict[str, Any] = None):
    """Resume a paused workflow with user input"""
    config = {"configurable": {"thread_id": workflow_id}}
    
    if user_input:
        result = await self.workflow.ainvoke(user_input, config=config)
    else:
        result = await self.workflow.ainvoke({}, config=config)
    
    return result
```

## Usage Examples

### Basic Workflow Execution

```python
from app.core.premium.workflows.sequential_generation_workflow import SequentialGenerationWorkflow

# Initialize workflow
workflow = SequentialGenerationWorkflow()

# Start workflow
workflow_id = await workflow.start_workflow(
    source_content="Your source material here",
    source_type="educational_text",
    user_preferences={"difficulty_level": "intermediate"}
)

# Get status
status = await workflow.get_workflow_status(workflow_id)
print(f"Workflow status: {status['status']}")
```

### User Edit Integration

```python
# Apply user edits
edits = {
    "sections": [{"title": "Updated Section", "content": "New content"}],
    "primitives": [{"title": "New Primitive", "description": "Updated description"}]
}

success = await workflow.apply_user_edit(workflow_id, edits)
if success:
    print("Edits applied successfully")
```

### Workflow Resumption

```python
# Resume workflow after user input
resumed_status = await workflow.resume_workflow(
    workflow_id, 
    {"user_feedback": "Continue with current content"}
)
```

## Testing

### Run the Test Script

```bash
cd elevate-ai-api
python test_langgraph_workflow.py
```

### Test Features

1. **Basic Workflow Execution** - Tests the complete sequential generation
2. **Workflow Resumption** - Tests pause/resume capabilities
3. **Error Handling** - Tests error scenarios and recovery
4. **State Persistence** - Tests checkpointing functionality

## Migration from Custom Orchestrator

### Key Differences

| Feature | Custom Orchestrator | LangGraph Workflow |
|---------|-------------------|-------------------|
| State Management | Manual session tracking | Automatic with checkpointer |
| User Interaction | Custom edit system | Built-in `interrupt()` |
| Error Handling | Basic try/catch | Robust error recovery |
| Workflow Resumption | Manual state restoration | Automatic resumption |
| Code Structure | Class-based methods | Functional API with decorators |
| Testing | Complex setup required | Simple task-based testing |

### Migration Steps

1. **Replace Orchestrator Calls**
   ```python
   # Old
   orchestrator = GenerationOrchestrator()
   session_id = await orchestrator.start_generation_session(...)
   
   # New
   workflow = SequentialGenerationWorkflow()
   workflow_id = await workflow.start_workflow(...)
   ```

2. **Update State Access**
   ```python
   # Old
   progress = await orchestrator.get_generation_progress(session_id)
   
   # New
   status = await workflow.get_workflow_status(workflow_id)
   ```

3. **Replace User Edit System**
   ```python
   # Old
   await orchestrator.user_edit_content(session_id, step, edits)
   
   # New
   await workflow.apply_user_edit(workflow_id, edits)
   ```

## Configuration

### Environment Variables

The workflow uses the same environment variables as the existing system:
- `GOOGLE_API_KEY` - For Gemini LLM service
- `GEMINI_MODEL` - Model selection (defaults to `gemini-2.5-flash`)

### Checkpointer Configuration

```python
# In-memory checkpointer (default)
checkpointer = InMemorySaver()

# For production, consider:
# - Redis checkpointer for distributed systems
# - Database checkpointer for persistence
# - Custom checkpointer for specific requirements
```

## Performance Considerations

### Parallel Execution
While the current implementation is sequential, LangGraph supports parallel execution:

```python
# Example of parallel task execution
@entrypoint(checkpointer=checkpointer)
async def parallel_workflow():
    # Execute tasks in parallel
    blueprint_future = generate_blueprint()
    sections_future = generate_sections()
    
    # Wait for both to complete
    blueprint, sections = await asyncio.gather(
        blueprint_future, sections_future
    )
    
    return {"blueprint": blueprint, "sections": sections}
```

### Streaming Support
LangGraph supports streaming for real-time updates:

```python
# Stream workflow execution
async for chunk in workflow.astream(inputs, config=config):
    print(f"Step completed: {chunk}")
```

## Monitoring and Debugging

### Workflow Visualization
LangGraph provides built-in workflow visualization:

```python
# Get workflow graph
graph = workflow.get_graph()
# Visualize with Mermaid or other tools
```

### State Inspection
```python
# Get current workflow state
state = await workflow.get_state(config)

# Inspect specific components
print(f"Current step: {state['current_step']}")
print(f"Generated content: {len(state['sections'])} sections")
```

## Future Enhancements

### 1. **Advanced Routing**
- Conditional workflow paths based on content complexity
- Dynamic step selection based on user preferences

### 2. **Multi-Agent Coordination**
- Specialized agents for different content types
- Agent communication and handoff protocols

### 3. **Content Quality Gates**
- Automatic quality assessment at each step
- Content validation and improvement loops

### 4. **Integration with LangSmith**
- Workflow monitoring and analytics
- Performance optimization and A/B testing

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure LangGraph is installed: `pip install langgraph`
   - Check Python path configuration

2. **Workflow Hanging**
   - Check for missing `await` keywords
   - Verify LLM service connectivity

3. **State Persistence Issues**
   - Verify checkpointer configuration
   - Check thread_id uniqueness

4. **User Input Not Received**
   - Ensure `interrupt()` is properly called
   - Check workflow resumption logic

### Debug Mode

Enable debug logging for detailed workflow execution:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Conclusion

The LangGraph-based workflow provides a robust, scalable, and maintainable solution for sequential content generation. It leverages modern workflow orchestration patterns while maintaining the same user experience and functionality as the custom orchestrator.

The key advantages are:
- **Maintainability**: Clean, functional API design
- **Reliability**: Built-in error handling and recovery
- **Scalability**: Production-ready architecture
- **Flexibility**: Easy to extend and modify
- **Integration**: Seamless LangGraph ecosystem support

This implementation represents a significant improvement over the custom orchestrator and provides a solid foundation for future enhancements and scaling.

