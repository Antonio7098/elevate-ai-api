# Answer Evaluation API

## Overview

The Answer Evaluation API allows you to evaluate user answers to questions using AI-powered assessment. This endpoint provides intelligent feedback, corrected answers, and marks achieved based on the expected answer and marking criteria.

## Endpoint

### POST /api/v1/ai/evaluate-answer

Evaluates a user's answer to a specific question.

**Authentication:** Required (JWT Bearer token)

## Request

### Headers
```
Authorization: Bearer <your_jwt_token>
Content-Type: application/json
```

### Request Body

```json
{
  "question_id": 123,
  "user_answer": "Mitochondria are the powerhouse of the cell and generate ATP."
}
```

#### Field Descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `question_id` | integer | Yes | The ID of the question to evaluate. Must be a positive integer. |
| `user_answer` | string | Yes | The answer provided by the user. Cannot be empty. |

## Response

### Success Response (200 OK)

```json
{
  "corrected_answer": "Mitochondria are the powerhouse of the cell, generating ATP through cellular respiration.",
  "marks_available": 5,
  "marks_achieved": 4
}
```

#### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `corrected_answer` | string | The corrected or improved version of the user's answer |
| `marks_available` | integer | Total marks available for this question |
| `marks_achieved` | integer | Marks awarded to the user (0 to marks_available) |

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Validation error: Question ID must be a positive integer"
}
```

### 401 Unauthorized
```json
{
  "detail": "Not authenticated"
}
```

### 404 Not Found
```json
{
  "detail": "Question not found: 999"
}
```

### 422 Validation Error
```json
{
  "detail": [
    {
      "loc": ["body", "question_id"],
      "msg": "Question ID must be a positive integer",
      "type": "value_error"
    }
  ]
}
```

### 500 Internal Server Error
```json
{
  "detail": "Answer evaluation failed: <error_message>"
}
```

## Examples

### Example 1: Correct Answer

**Request:**
```bash
curl -X POST "https://api.elevate.ai/v1/ai/evaluate-answer" \
  -H "Authorization: Bearer your_jwt_token" \
  -H "Content-Type: application/json" \
  -d '{
    "question_id": 1,
    "user_answer": "Mitochondria are the powerhouse of the cell, generating ATP through cellular respiration."
  }'
```

**Response:**
```json
{
  "corrected_answer": "Mitochondria are the powerhouse of the cell, generating ATP through cellular respiration.",
  "marks_available": 5,
  "marks_achieved": 5
}
```

### Example 2: Partially Correct Answer

**Request:**
```bash
curl -X POST "https://api.elevate.ai/v1/ai/evaluate-answer" \
  -H "Authorization: Bearer your_jwt_token" \
  -H "Content-Type: application/json" \
  -d '{
    "question_id": 1,
    "user_answer": "Mitochondria are the powerhouse and make ATP."
  }'
```

**Response:**
```json
{
  "corrected_answer": "Mitochondria are the powerhouse of the cell, generating ATP through cellular respiration.",
  "marks_available": 5,
  "marks_achieved": 3
}
```

### Example 3: Incorrect Answer

**Request:**
```bash
curl -X POST "https://api.elevate.ai/v1/ai/evaluate-answer" \
  -H "Authorization: Bearer your_jwt_token" \
  -H "Content-Type: application/json" \
  -d '{
    "question_id": 1,
    "user_answer": "Mitochondria are small organelles."
  }'
```

**Response:**
```json
{
  "corrected_answer": "Mitochondria are the powerhouse of the cell, generating ATP through cellular respiration.",
  "marks_available": 5,
  "marks_achieved": 1
}
```

## Error Examples

### Invalid Question ID
```bash
curl -X POST "https://api.elevate.ai/v1/ai/evaluate-answer" \
  -H "Authorization: Bearer your_jwt_token" \
  -H "Content-Type: application/json" \
  -d '{
    "question_id": 0,
    "user_answer": "Test answer"
  }'
```

**Response:**
```json
{
  "detail": [
    {
      "loc": ["body", "question_id"],
      "msg": "Question ID must be a positive integer",
      "type": "value_error"
    }
  ]
}
```

### Empty Answer
```bash
curl -X POST "https://api.elevate.ai/v1/ai/evaluate-answer" \
  -H "Authorization: Bearer your_jwt_token" \
  -H "Content-Type: application/json" \
  -d '{
    "question_id": 1,
    "user_answer": ""
  }'
```

**Response:**
```json
{
  "detail": [
    {
      "loc": ["body", "user_answer"],
      "msg": "User answer cannot be empty",
      "type": "value_error"
    }
  ]
}
```

## Implementation Notes

### AI Evaluation Process

1. **Question Retrieval**: The system retrieves the question data including the expected answer, marking criteria, and total marks available.

2. **AI Assessment**: The answer is evaluated using an AI service that considers:
   - Semantic similarity to the expected answer
   - Coverage of key concepts and terminology
   - Adherence to marking criteria
   - Clarity and completeness of explanation

3. **Fallback Evaluation**: If the AI service is unavailable, the system falls back to keyword-based evaluation.

### Marking Criteria

The AI evaluates answers based on the question's marking criteria, which typically includes:
- Key concepts and terminology
- Required level of detail
- Specific points to be awarded
- Clarity and coherence of explanation

### Performance Considerations

- **Response Time**: Typical response time is 2-5 seconds depending on answer complexity
- **Rate Limiting**: Standard API rate limits apply
- **Fallback Mode**: System gracefully degrades to keyword matching if AI service is unavailable

## Integration Examples

### JavaScript/Node.js
```javascript
const evaluateAnswer = async (questionId, userAnswer, token) => {
  const response = await fetch('https://api.elevate.ai/v1/ai/evaluate-answer', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      question_id: questionId,
      user_answer: userAnswer
    })
  });
  
  return await response.json();
};

// Usage
const result = await evaluateAnswer(1, "Mitochondria are the powerhouse of the cell", "your_jwt_token");
console.log(`Marks achieved: ${result.marks_achieved}/${result.marks_available}`);
```

### Python
```python
import requests

def evaluate_answer(question_id, user_answer, token):
    url = "https://api.elevate.ai/v1/ai/evaluate-answer"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    data = {
        "question_id": question_id,
        "user_answer": user_answer
    }
    
    response = requests.post(url, headers=headers, json=data)
    return response.json()

# Usage
result = evaluate_answer(1, "Mitochondria are the powerhouse of the cell", "your_jwt_token")
print(f"Marks achieved: {result['marks_achieved']}/{result['marks_available']}")
```

## Related Endpoints

- [Question Generation API](./question-generation.md) - Generate questions from LearningBlueprints
- [Deconstruction API](./deconstruction.md) - Create LearningBlueprints from source text
- [Chat API](./chat.md) - Interactive learning conversations 