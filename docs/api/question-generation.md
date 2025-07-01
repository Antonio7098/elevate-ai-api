# Question Generation API Documentation

This document provides detailed information about the question generation endpoint that creates QuestionSet objects from LearningBlueprints.

## Endpoint Overview

**URL:** `POST /api/v1/ai-rag/learning-blueprints/{blueprint_id}/question-sets`

**Purpose:** Generates a new QuestionSet based on the content of a specific LearningBlueprint using AI-powered question generation.

**Authentication:** Required (Bearer token)

## Request Format

### Path Parameters

- `blueprint_id` (string, required): The ID of the LearningBlueprint to use for question generation

### Request Body

```json
{
  "name": "Mitochondria Quiz",
  "folder_id": 123,
  "question_options": {
    "scope": "KeyConcepts",
    "tone": "Formal",
    "difficulty": "Medium",
    "count": 10,
    "types": ["multiple_choice", "short_answer"]
  }
}
```

#### Field Descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | The title for the new QuestionSet |
| `folder_id` | integer | No | ID of the folder to store the new question set in |
| `question_options` | object | No | Additional parameters to guide the AI's question generation process |

#### Question Options Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `scope` | string | "All" | What content to focus on: "KeyConcepts", "All", "Processes", "Relationships" |
| `tone` | string | "Neutral" | Question style: "Formal", "Casual", "Academic", "Conversational" |
| `difficulty` | string | "Medium" | Question difficulty: "Easy", "Medium", "Hard" |
| `count` | integer | 5 | Number of questions to generate (1-20) |
| `types` | array | ["mixed"] | Question types: "multiple_choice", "short_answer", "essay", "true_false" |

## Response Format

### Success Response (200 OK)

```json
{
  "id": 1,
  "name": "Mitochondria Quiz",
  "blueprint_id": "blueprint_123",
  "folder_id": 123,
  "questions": [
    {
      "text": "What is the primary function of the mitochondria?",
      "answer": "It is the powerhouse of the cell, generating ATP.",
      "question_type": "understand",
      "total_marks_available": 2,
      "marking_criteria": "Award 1 mark for 'powerhouse' and 1 mark for mentioning 'ATP'."
    },
    {
      "text": "Explain the relationship between mitochondria and cellular energy.",
      "answer": "Mitochondria convert nutrients into ATP, which serves as the cell's energy currency.",
      "question_type": "explore",
      "total_marks_available": 3,
      "marking_criteria": "Award 1 mark for mentioning conversion, 1 mark for ATP, 1 mark for energy currency concept."
    }
  ],
  "created_at": "2025-01-27T10:00:00.000Z",
  "updated_at": "2025-01-27T10:00:00.000Z"
}
```

### Response Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `id` | integer | Unique identifier for the QuestionSet |
| `name` | string | Name of the question set |
| `blueprint_id` | string | ID of the source LearningBlueprint |
| `folder_id` | integer | ID of the folder containing the question set (null if not specified) |
| `questions` | array | List of generated questions |
| `created_at` | string | Timestamp when the question set was created (ISO 8601) |
| `updated_at` | string | Timestamp when the question set was last updated (ISO 8601) |

#### Question Object Structure

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | The question text |
| `answer` | string | The correct answer or explanation |
| `question_type` | string | Type of question: "understand", "use", "explore" |
| `total_marks_available` | integer | Total marks available for this question |
| `marking_criteria` | string | Detailed marking criteria for scoring |

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Validation error: Name cannot be empty"
}
```

**Causes:**
- Empty or missing `name` field
- Invalid `folder_id` (negative number)
- Invalid question options format

### 401 Unauthorized
```json
{
  "detail": "Invalid API key"
}
```

**Causes:**
- Missing or invalid authentication token

### 404 Not Found
```json
{
  "detail": "LearningBlueprint not found or access denied"
}
```

**Causes:**
- The specified `blueprint_id` does not exist
- User does not have access to the specified blueprint

### 502 Bad Gateway
```json
{
  "detail": "Question generation failed: AI service unavailable"
}
```

**Causes:**
- Internal AI service is down
- AI service returned an error
- Network connectivity issues

## Usage Examples

### Basic Question Generation

```bash
curl -X POST "http://localhost:8000/api/v1/ai-rag/learning-blueprints/blueprint_123/question-sets" \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Basic Quiz"
  }'
```

### Advanced Question Generation with Options

```bash
curl -X POST "http://localhost:8000/api/v1/ai-rag/learning-blueprints/blueprint_123/question-sets" \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Advanced Assessment",
    "folder_id": 456,
    "question_options": {
      "scope": "KeyConcepts",
      "tone": "Academic",
      "difficulty": "Hard",
      "count": 15,
      "types": ["essay", "short_answer"]
    }
  }'
```

### Python Example

```python
import requests

url = "http://localhost:8000/api/v1/ai-rag/learning-blueprints/blueprint_123/question-sets"
headers = {
    "Authorization": "Bearer your_api_key",
    "Content-Type": "application/json"
}
data = {
    "name": "Python Quiz",
    "folder_id": 789,
    "question_options": {
        "scope": "All",
        "tone": "Formal",
        "difficulty": "Medium",
        "count": 10
    }
}

response = requests.post(url, json=data, headers=headers)
if response.status_code == 200:
    question_set = response.json()
    print(f"Generated {len(question_set['questions'])} questions")
    for question in question_set['questions']:
        print(f"Q: {question['text']}")
        print(f"A: {question['answer']}")
        print(f"Marks: {question['total_marks_available']}")
        print("---")
else:
    print(f"Error: {response.status_code} - {response.text}")
```

## Best Practices

1. **Question Options**: Use specific question options to get more targeted results
2. **Question Count**: Keep question count reasonable (5-15) for better quality
3. **Error Handling**: Always handle potential error responses in your client code
4. **Rate Limiting**: Be mindful of API rate limits when generating multiple question sets
5. **Validation**: Validate the response structure before processing questions

## Integration Notes

- The endpoint integrates with an internal AI service for question generation
- Question generation may take several seconds depending on complexity
- Generated questions are based on the knowledge primitives in the LearningBlueprint
- Questions include detailed marking criteria for automated or manual scoring
- The endpoint supports various question types and difficulty levels 