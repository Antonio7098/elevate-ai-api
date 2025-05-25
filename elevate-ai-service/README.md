# Elevate AI Service

This is the AI service component of the Elevate platform, responsible for evaluating user answers to questions using Google's Gemini API.

## Features

- `/evaluate-answer` endpoint to assess user answers against expected answers
- Support for different question types (multiple-choice, true-false, short-answer)
- Detailed feedback and scoring

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

## Authentication

All API requests (except `/health`) require an API key to be provided in the Authorization header:

```
Authorization: Bearer your-api-key-here
```

The API key should match the `CORE_API_ACCESS_KEY` environment variable.
