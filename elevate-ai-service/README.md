# Elevate AI Service

This is the AI service component of the Elevate platform, powered by Google's Gemini API. It provides intelligent educational services including answer evaluation, question generation, and conversational AI.

## Features

### Answer Evaluation
- `/evaluate-answer` endpoint to assess user answers against expected answers
- Support for different question types (multiple-choice, true-false, short-answer)
- Detailed feedback and scoring

### Question Generation
- `/generate-questions` endpoint to create educational questions from source text
- Support for multiple question types and difficulty levels
- Topic-focused question generation with explanations

### Conversational AI
- `/chat` endpoint for educational conversations
- Context-aware responses with reference citations
- Suggested follow-up questions

## Setup

1. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   - Copy `.env.example` to `.env` (or create a new `.env` file)
   - Add your Google Gemini API key and other configuration

4. Run the service:
   ```
   flask run --port=8000
   ```
   
   Or with gunicorn (production):
   ```
   gunicorn -w 4 -b 0.0.0.0:8000 app:app
   ```

## API Endpoints

### Health Check

```
GET /health
```

Returns the status and version of the service.

### Evaluate Answer

```
POST /evaluate-answer
```

Evaluates a user's answer against an expected answer.

#### Request Body

```json
{
  "questionContext": {
    "questionId": "q123",
    "questionText": "What is the capital of France?",
    "expectedAnswer": "Paris",
    "questionType": "short-answer"
  },
  "userAnswer": "Paris is the capital of France"
}
```

#### Response

```json
{
  "success": true,
  "evaluation": {
    "isCorrect": true,
    "isPartiallyCorrect": false,
    "score": 1.0,
    "feedbackText": "Your answer is correct! Paris is indeed the capital of France.",
    "suggestedCorrectAnswer": "Paris"
  },
  "metadata": {
    "processingTime": "0.85s",
    "model": "gemini-1.5-flash-latest",
    "questionId": "q123"
  }
}
```

### Generate Questions

```
POST /generate-questions
```

Generates educational questions from source text.

#### Request Body

```json
{
  "sourceText": "The water cycle, also known as the hydrologic cycle, describes the continuous movement of water on, above, and below the surface of the Earth.",
  "questionCount": 3,
  "questionTypes": ["multiple-choice", "true-false", "short-answer"],
  "difficulty": "medium",
  "topics": ["water cycle", "earth science"],
  "language": "en"
}
```

#### Response

```json
{
  "success": true,
  "questions": [
    {
      "text": "Which of the following processes is NOT a part of the water cycle?",
      "questionType": "multiple-choice",
      "answer": "Sublimation",
      "options": ["Evaporation", "Condensation", "Precipitation", "Sublimation"],
      "explanation": "While sublimation (the transition from solid to gas) involves water, it's not explicitly mentioned as a main process in the water cycle description provided."
    },
    {
      "text": "True or false: The amount of water on Earth changes significantly over time.",
      "questionType": "true-false",
      "answer": "false",
      "explanation": "The water cycle maintains a relatively constant amount of water on Earth, though individual water molecules move between different reservoirs."
    },
    {
      "text": "Explain how water moves in the water cycle.",
      "questionType": "short-answer",
      "answer": "Water moves through the water cycle via processes like evaporation, condensation, precipitation, infiltration, surface runoff, and subsurface flow.",
      "explanation": "These processes allow water to circulate between the atmosphere, land, and oceans in different states (liquid, solid, vapor)."
    }
  ],
  "metadata": {
    "processingTime": "2.67s",
    "model": "gemini-1.5-flash-latest",
    "sourceTextLength": 142
  }
}
```

### Chat with AI

```
POST /chat
```

Provides conversational AI responses with educational content.

#### Request Body

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
      "content": "Biology is the scientific study of life and living organisms."
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
            "answer": "Photosynthesis is the process by which green plants use sunlight to synthesize foods with the help of chlorophyll."
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

#### Response

```json
{
  "success": true,
  "response": {
    "message": "Photosynthesis is the amazing process by which plants make their own food using sunlight! Imagine plants as tiny solar-powered factories. They take in sunlight, water, and carbon dioxide, and convert these ingredients into glucose (sugar) and oxygen. The green pigment called chlorophyll captures the sun's energy to power this process.",
    "references": [
      {
        "text": "Photosynthesis is the process by which green plants use sunlight to synthesize foods with the help of chlorophyll.",
        "source": "Biology 101 Question Set"
      }
    ],
    "suggestedQuestions": [
      "What is chlorophyll and how does it work?",
      "What are the different stages of photosynthesis?",
      "How does photosynthesis affect the Earth's atmosphere?"
    ]
  },
  "metadata": {
    "processingTime": "3.37s",
    "model": "gemini-1.5-flash-latest",
    "tokensUsed": 419
  }
}
```

## Authentication

All API requests (except `/health`) require an API key to be provided in the Authorization header:

```
Authorization: Bearer your-api-key-here
```

The API key should match the `CORE_API_ACCESS_KEY` environment variable.
