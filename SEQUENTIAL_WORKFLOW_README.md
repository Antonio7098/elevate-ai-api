# Sequential Generation Workflow with User Editing

## Overview

The Elevate AI API now implements a **sequential generation workflow** that creates cohesive learning paths by building each component on the previous one's output. This ensures better content coherence and allows users to edit content between each step.

## Workflow Architecture

```
source → blueprint → sections → primitives → mastery criteria → questions
                           |                       |                  |
                                          Notes
```

### Sequential Flow Benefits

1. **Content Coherence**: Each step builds on the previous step's understanding
2. **Tailored Relationships**: Mastery criteria are specific to primitives, questions are specific to criteria
3. **Quality Control**: Validate each step before proceeding
4. **User Control**: Edit content between steps for customization

## API Endpoints

### 1. Start Generation Session
```http
POST /api/v1/orchestrator/start
```

**Request:**
```json
{
  "source_content": "Raw source text to process...",
  "source_type": "textbook_chapter",
  "user_preferences": {
    "learning_style": "visual",
    "difficulty": "intermediate"
  },
  "session_title": "Optional session title"
}
```

**Response:**
```json
{
  "session_id": "uuid",
  "current_step": "source_analysis",
  "status": "in_progress",
  "message": "Generation session started successfully...",
  "next_actions": ["Proceed to next step", "Edit current content"]
}
```

### 2. Proceed to Next Step
```http
POST /api/v1/orchestrator/proceed
```

**Request:**
```json
{
  "session_id": "uuid"
}
```

**Response:**
```json
{
  "session_id": "uuid",
  "current_step": "blueprint_creation",
  "status": "ready_for_next",
  "completed_steps": ["source_analysis"],
  "current_content": { /* generated content */ },
  "user_edits": [],
  "errors": [],
  "next_actions": ["Proceed to next step", "Edit current content"]
}
```

### 3. Edit Generated Content
```http
POST /api/v1/orchestrator/edit?session_id=uuid
```

**Request:**
```json
{
  "step": "section_generation",
  "content_id": "sec_001",
  "edited_content": {
    "title": "Updated Section Title"
  },
  "user_notes": "Made title more descriptive"
}
```

### 4. Get Generation Progress
```http
GET /api/v1/orchestrator/progress/{session_id}
```

### 5. Get Complete Learning Path
```http
GET /api/v1/orchestrator/complete/{session_id}
```

### 6. Delete Session
```http
DELETE /api/v1/orchestrator/session/{session_id}
```

## Generation Steps

### 1. Source Analysis
- **Purpose**: Analyze and prepare source content
- **Input**: Raw source text
- **Output**: Structured source data with metadata
- **Status**: `in_progress` → `ready_for_next`

### 2. Blueprint Creation
- **Purpose**: Create learning blueprint from source
- **Input**: Analyzed source content
- **Output**: Structured blueprint with sections
- **Status**: `in_progress` → `ready_for_next`

### 3. Section Generation
- **Purpose**: Extract and organize content sections
- **Input**: Blueprint structure
- **Output**: Hierarchical section organization
- **Status**: `in_progress` → `ready_for_next`
- **User Editing**: ✅ Available

### 4. Primitive Extraction
- **Purpose**: Extract knowledge primitives from sections
- **Input**: Organized sections
- **Output**: Knowledge primitives with metadata
- **Status**: `in_progress` → `ready_for_next`
- **User Editing**: ✅ Available

### 5. Mastery Criteria Generation
- **Purpose**: Create mastery criteria for primitives
- **Input**: Knowledge primitives
- **Output**: UUE-progressive mastery criteria
- **Status**: `in_progress` → `ready_for_next`
- **User Editing**: ✅ Available

### 6. Question Generation
- **Purpose**: Generate questions from mastery criteria
- **Input**: Mastery criteria
- **Output**: Tailored assessment questions
- **Status**: `in_progress` → `ready_for_next`
- **User Editing**: ✅ Available

### 7. Note Generation
- **Purpose**: Create comprehensive study notes
- **Input**: Complete learning path
- **Output**: Integrated study materials
- **Status**: `in_progress` → `completed`

## Session States

- **`pending`**: Session created but not started
- **`in_progress`**: Currently executing a step
- **`ready_for_next`**: Step complete, ready for next step or user editing
- **`user_editing`**: User is editing content
- **`completed`**: All steps finished
- **`failed`**: Error occurred during execution

## User Editing Workflow

1. **Review Generated Content**: After each step, review the generated content
2. **Make Edits**: Use the edit endpoint to modify content
3. **Apply Changes**: Edits are applied immediately to the session
4. **Proceed**: Continue to the next step with updated content
5. **Track Changes**: All edits are logged with timestamps and user notes

## Example Usage

### Python Client Example
```python
import asyncio
import aiohttp

async def create_learning_path():
    async with aiohttp.ClientSession() as session:
        # 1. Start generation session
        start_response = await session.post(
            "http://localhost:8000/api/v1/orchestrator/start",
            json={
                "source_content": "Your source text here...",
                "source_type": "textbook_chapter",
                "user_preferences": {"learning_style": "visual"}
            }
        )
        start_data = await start_response.json()
        session_id = start_data["session_id"]
        
        # 2. Proceed through steps
        while True:
            # Get current progress
            progress_response = await session.get(
                f"http://localhost:8000/api/v1/orchestrator/progress/{session_id}"
            )
            progress = await progress_response.json()
            
            if progress["status"] == "completed":
                break
            
            # Edit content if needed (example: edit sections)
            if progress["current_step"] == "section_generation":
                await session.post(
                    f"http://localhost:8000/api/v1/orchestrator/edit?session_id={session_id}",
                    json={
                        "step": "section_generation",
                        "content_id": "sec_001",
                        "edited_content": {"title": "Updated Title"},
                        "user_notes": "Made title more descriptive"
                    }
                )
            
            # Proceed to next step
            await session.post(
                "http://localhost:8000/api/v1/orchestrator/proceed",
                json={"session_id": session_id}
            )
        
        # 3. Get complete learning path
        complete_response = await session.get(
            f"http://localhost:8000/api/v1/orchestrator/complete/{session_id}"
        )
        complete_path = await complete_response.json()
        return complete_path

# Run the workflow
complete_path = asyncio.run(create_learning_path())
```

### JavaScript/Node.js Example
```javascript
const axios = require('axios');

async function createLearningPath() {
    try {
        // 1. Start generation session
        const startResponse = await axios.post('http://localhost:8000/api/v1/orchestrator/start', {
            source_content: 'Your source text here...',
            source_type: 'textbook_chapter',
            user_preferences: { learning_style: 'visual' }
        });
        
        const sessionId = startResponse.data.session_id;
        
        // 2. Proceed through steps
        let isComplete = false;
        while (!isComplete) {
            const progressResponse = await axios.get(
                `http://localhost:8000/api/v1/orchestrator/progress/${sessionId}`
            );
            const progress = progressResponse.data;
            
            if (progress.status === 'completed') {
                isComplete = true;
                break;
            }
            
            // Proceed to next step
            await axios.post('http://localhost:8000/api/v1/orchestrator/proceed', {
                session_id: sessionId
            });
        }
        
        // 3. Get complete learning path
        const completeResponse = await axios.get(
            `http://localhost:8000/api/v1/orchestrator/complete/${sessionId}`
        );
        
        return completeResponse.data;
    } catch (error) {
        console.error('Error creating learning path:', error);
        throw error;
    }
}

// Run the workflow
createLearningPath().then(completePath => {
    console.log('Learning path created:', completePath);
});
```

## Demo Script

Run the demo to see the workflow in action:

```bash
python demo_sequential_workflow.py
```

The demo shows:
- Complete workflow execution
- User editing between steps
- Content generation at each stage
- Final cohesive learning path

## Benefits of Sequential Generation

### 1. **Content Quality**
- Each component is tailored to previous outputs
- Better alignment between sections, primitives, and criteria
- Cohesive learning experience

### 2. **User Control**
- Edit content at any step before proceeding
- Customize learning path to preferences
- Maintain quality standards

### 3. **Debugging & Validation**
- Easy to identify where issues occur
- Validate each step independently
- Clear error handling and recovery

### 4. **Scalability**
- Can parallelize within steps where appropriate
- Modular architecture for easy extension
- Session-based state management

## Integration with Existing Services

The orchestrator integrates with existing services:
- **Blueprint Generation**: Uses existing deconstruction logic
- **Mastery Criteria**: Integrates with mastery criteria service
- **Question Generation**: Uses question generation service
- **Note Generation**: Integrates with note services

## Future Enhancements

1. **Parallel Processing**: Parallelize independent operations within steps
2. **Caching**: Cache intermediate results for faster regeneration
3. **Templates**: Pre-defined generation templates for common use cases
4. **Collaboration**: Multi-user editing and review workflows
5. **Versioning**: Track changes and maintain version history

## Troubleshooting

### Common Issues

1. **Session Not Found**: Ensure session ID is correct and session exists
2. **Step Not Ready**: Check if previous step completed successfully
3. **Content Generation Failed**: Review error logs and retry
4. **User Edit Conflicts**: Resolve conflicts before proceeding

### Error Handling

- All endpoints return appropriate HTTP status codes
- Error messages include actionable guidance
- Failed sessions can be restarted or deleted
- User edits are preserved even if generation fails

## Support

For questions or issues with the sequential workflow:
- Check the API documentation
- Review error logs
- Run the demo script to verify functionality
- Contact the development team



