# AI Service Contract

This document defines the contract between the Elevate Core API (Node.js) and the AI Service (Python). It specifies the request and response formats for all endpoints that the Core API will use to communicate with the AI Service.

## Base URL

The AI Service will be accessible at a configurable base URL, which will be stored in the Core API's environment variables:

```
AI_SERVICE_URL=http://localhost:5000
```

## Authentication

All requests from the Core API to the AI Service will include an API key for authentication:

```
Authorization: Bearer <API_KEY>
```

The API key will be stored in the Core API's environment variables:

```
AI_SERVICE_API_KEY=your-api-key-here
```

## Endpoints

### 1. Generate Questions from Source Text

#### Endpoint

```
POST /api/generate-questions
```

#### Request

```json
{
  "sourceText": "The text from which to generate questions",
  "questionCount": 5,
  "questionTypes": ["multiple-choice", "true-false", "short-answer"],
  "difficulty": "medium",
  "topics": ["optional", "list", "of", "topics"],
  "language": "en"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| sourceText | string | Yes | The source text from which to generate questions. |
| questionCount | integer | Yes | The number of questions to generate. |
| questionTypes | array of strings | No | The types of questions to generate. Possible values: "multiple-choice", "true-false", "short-answer". If not provided, all types will be considered. |
| difficulty | string | No | The difficulty level of the questions. Possible values: "easy", "medium", "hard". Default: "medium". |
| topics | array of strings | No | Specific topics to focus on when generating questions. |
| language | string | No | The language of the source text and generated questions. ISO 639-1 code. Default: "en" (English). |

#### Response

```json
{
  "success": true,
  "questions": [
    {
      "text": "What is the capital of France?",
      "questionType": "multiple-choice",
      "answer": "Paris",
      "options": ["Paris", "London", "Berlin", "Madrid"],
      "explanation": "Paris is the capital and most populous city of France."
    },
    {
      "text": "The Earth revolves around the Sun.",
      "questionType": "true-false",
      "answer": "true",
      "explanation": "The Earth orbits the Sun at an average distance of about 93 million miles."
    },
    {
      "text": "What is the main function of the mitochondria in a cell?",
      "questionType": "short-answer",
      "answer": "Powerhouse of the cell that generates energy in the form of ATP",
      "explanation": "Mitochondria are organelles that generate most of the cell's supply of adenosine triphosphate (ATP), used as a source of chemical energy."
    }
  ],
  "metadata": {
    "processingTime": "2.5s",
    "model": "gpt-4",
    "sourceTextLength": 1500
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| success | boolean | Indicates whether the request was successful. |
| questions | array of objects | The generated questions. |
| questions[].text | string | The question text. |
| questions[].questionType | string | The type of question. Possible values: "multiple-choice", "true-false", "short-answer". |
| questions[].answer | string | The correct answer to the question. |
| questions[].options | array of strings | For multiple-choice questions, the possible options. The correct answer is included in this array. |
| questions[].explanation | string | An explanation of the answer, useful for learning. |
| metadata | object | Additional information about the processing. |
| metadata.processingTime | string | The time taken to process the request. |
| metadata.model | string | The AI model used to generate the questions. |
| metadata.sourceTextLength | integer | The length of the source text in characters. |

#### Error Response

```json
{
  "success": false,
  "error": {
    "code": "invalid_request",
    "message": "Source text is too short to generate meaningful questions."
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| success | boolean | Always false for error responses. |
| error | object | Error details. |
| error.code | string | A machine-readable error code. |
| error.message | string | A human-readable error message. |

### 2. Chat with AI

#### Endpoint

```
POST /api/chat
```

#### Request

```json
{
  "message": "Can you explain the concept of photosynthesis?",
  "conversation": [
    {
      "role": "user",
      "content": "What is biology?"
    },
    {
      "role": "assistant",
      "content": "Biology is the scientific study of life and living organisms, including their physical structure, chemical processes, molecular interactions, physiological mechanisms, development, and evolution."
    }
  ],
  "context": {
    "questionSets": [
      {
        "id": 1,
        "name": "Biology 101",
        "questions": [
          {
            "text": "What is photosynthesis?",
            "answer": "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll."
          }
        ]
      }
    ],
    "userLevel": "beginner",
    "preferredLearningStyle": "visual"
  },
  "language": "en"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| message | string | Yes | The user's current message to the AI. |
| conversation | array of objects | No | The conversation history. Each object has a "role" (either "user" or "assistant") and "content" (the message text). |
| context | object | No | Additional context to help the AI generate more relevant responses. |
| context.questionSets | array of objects | No | Question sets that provide context for the conversation. |
| context.userLevel | string | No | The user's knowledge level. Possible values: "beginner", "intermediate", "advanced". |
| context.preferredLearningStyle | string | No | The user's preferred learning style. Possible values: "visual", "auditory", "reading", "kinesthetic". |
| language | string | No | The language of the conversation. ISO 639-1 code. Default: "en" (English). |

#### Response

```json
{
  "success": true,
  "response": {
    "message": "Photosynthesis is the process by which plants, algae, and some bacteria convert sunlight, carbon dioxide, and water into glucose (a sugar) and oxygen. This process is crucial for life on Earth as it produces oxygen and serves as the primary source of energy for most ecosystems.\n\nThe process occurs in the chloroplasts of plant cells, specifically in the chlorophyll-containing structures called thylakoids. Chlorophyll is the green pigment that absorbs sunlight.\n\nThe basic equation for photosynthesis is:\n6CO₂ + 6H₂O + light energy → C₆H₁₂O₆ + 6O₂\n\nThis means that six molecules of carbon dioxide plus six molecules of water, with the addition of light energy, produce one molecule of glucose and six molecules of oxygen.",
    "references": [
      {
        "text": "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll.",
        "source": "Biology 101 Question Set"
      }
    ],
    "suggestedQuestions": [
      "What are the stages of photosynthesis?",
      "How does photosynthesis differ from cellular respiration?",
      "What factors affect the rate of photosynthesis?"
    ]
  },
  "metadata": {
    "processingTime": "1.2s",
    "model": "gpt-4",
    "tokensUsed": 350
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| success | boolean | Indicates whether the request was successful. |
| response | object | The AI's response. |
| response.message | string | The AI's message in response to the user's query. |
| response.references | array of objects | References to information used in the response. |
| response.references[].text | string | The referenced text. |
| response.references[].source | string | The source of the referenced text. |
| response.suggestedQuestions | array of strings | Suggested follow-up questions the user might want to ask. |
| metadata | object | Additional information about the processing. |
| metadata.processingTime | string | The time taken to process the request. |
| metadata.model | string | The AI model used to generate the response. |
| metadata.tokensUsed | integer | The number of tokens used in processing the request. |

#### Error Response

```json
{
  "success": false,
  "error": {
    "code": "context_too_large",
    "message": "The provided context is too large. Please reduce the amount of context data."
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| success | boolean | Always false for error responses. |
| error | object | Error details. |
| error.code | string | A machine-readable error code. |
| error.message | string | A human-readable error message. |

## Error Codes

The AI Service may return the following error codes:

| Code | Description |
|------|-------------|
| invalid_request | The request was invalid or improperly formatted. |
| context_too_large | The provided context is too large. |
| text_too_short | The provided text is too short to process. |
| text_too_long | The provided text is too long to process. |
| unsupported_language | The requested language is not supported. |
| rate_limit_exceeded | The rate limit for API calls has been exceeded. |
| internal_error | An internal error occurred in the AI Service. |
| service_unavailable | The AI Service is temporarily unavailable. |

## Rate Limiting

The AI Service implements rate limiting to prevent abuse. The Core API should handle rate limiting errors gracefully and implement appropriate retry mechanisms.

## Versioning

The AI Service API is versioned. The current version is v1. The version is specified in the URL:

```
http://localhost:5000/v1/api/generate-questions
```

## Implementation Notes

1. The Core API should implement error handling for all possible error responses from the AI Service.
2. Timeouts should be implemented for all requests to the AI Service.
3. The Core API should implement retry logic for transient errors.
4. All sensitive data should be properly sanitized before being sent to the AI Service.
5. The AI Service should be deployed in a secure environment with appropriate access controls.
