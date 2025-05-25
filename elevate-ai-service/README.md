# Elevate AI Service

This is the AI service component of the Elevate platform, powered by Google's Gemini API. It provides intelligent educational services including answer evaluation, question generation, and conversational AI capabilities.

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Configuration](#configuration)
- [API Documentation](#api-documentation)
- [Testing](#testing)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Features

### 📝 Answer Evaluation
- Intelligent assessment of student answers
- Support for multiple question types:
  - Multiple-choice
  - True/False
  - Short answer
- Detailed feedback and scoring
- Partial credit for partially correct answers

### ❓ Question Generation
- Generate educational questions from any text
- Customizable question types and difficulty levels
- Topic-focused question generation
- Includes explanations and answer keys

### 💬 Conversational AI
- Natural language understanding
- Context-aware responses
- Educational content generation
- Follow-up question suggestions

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- Google Gemini API key
- Virtual environment (recommended)

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/elevate-ai-api.git
   cd elevate-ai-api/elevate-ai-service
   ```

2. **Create and activate a virtual environment**
   ```bash
   # On Unix/macOS
   python -m venv venv
   source venv/bin/activate
   
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. **Environment Variables**
   Copy the example environment file and update it with your configuration:
   ```bash
   cp .env.example .env
   ```

2. **Required Configuration**
   Edit the `.env` file with your settings:
   ```env
   # Required
   CORE_API_ACCESS_KEY=your_secure_api_key
   GEMINI_API_KEY=your_gemini_api_key
   
   # Optional (with defaults)
   FLASK_APP=app.py
   FLASK_ENV=development
   PORT=8000
   AI_MODEL=gemini-1.5-flash-latest
   ```

## Running the Service

### Development Mode
```bash
flask run --port=8000
```

### Production Mode (using Gunicorn)
```bash
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

### Using Docker
```bash
# Build the image
docker build -t elevate-ai-service .

# Run the container
docker run -p 8000:8000 --env-file .env elevate-ai-service
```

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

## API Documentation

### Base URL
All endpoints are relative to: `http://localhost:8000`

### Authentication
Include your API key in the `Authorization` header:
```
Authorization: Bearer your_api_key_here
```

### Health Check
```http
GET /health
```

**Response**
```json
{
  "status": "ok",
  "version": "v1"
}
```

### Evaluate Answer

```http
POST /evaluate-answer
```

Evaluates a user's answer against an expected answer.

**Request Body**

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

**Response**

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

**Error Responses**

| Status Code | Error Code | Description |
|-------------|------------|-------------|
| 400 | invalid_request | Missing or invalid parameters |
| 401 | unauthorized | Invalid or missing API key |
| 500 | internal_error | Server error |

### Generate Questions

```http
POST /generate-questions
```

Generates educational questions from source text.

**Request Body**

```json
{
  "sourceText": "The water cycle describes the continuous movement of water on Earth.",
  "questionCount": 2,
  "questionTypes": ["multiple-choice", "true-false"],
  "difficulty": "medium",
  "topics": ["water cycle", "earth science"],
  "language": "en"
}
```

**Response**

```json
{
  "success": true,
  "questions": [
    {
      "text": "What is the water cycle?",
      "questionType": "multiple-choice",
      "answer": "The continuous movement of water on Earth",
      "options": [
        "The cycle of seasons",
        "The continuous movement of water on Earth",
        "The process of water freezing",
        "The way rivers flow"
      ],
      "explanation": "The water cycle describes how water moves through different states and locations on Earth."
    },
    {
      "text": "True or False: The water cycle includes the process of evaporation.",
      "questionType": "true-false",
      "answer": "true",
      "explanation": "Evaporation is a key part of the water cycle where water changes from liquid to vapor."
    }
  ],
  "metadata": {
    "processingTime": "1.23s",
    "model": "gemini-1.5-flash-latest",
    "sourceTextLength": 65
  }
}
```

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

```http
POST /chat
```

Provides conversational AI responses with educational content.

**Request Body**

```json
{
  "message": "Explain the water cycle",
  "conversation": [
    {
      "role": "user",
      "content": "Hello"
    },
    {
      "role": "assistant",
      "content": "Hi there! How can I help you with your learning today?"
    }
  ],
  "context": {
    "subject": "Science",
    "gradeLevel": "middle school"
  },
  "language": "en"
}
```

**Response**

```json
{
  "success": true,
  "message": "The water cycle is the continuous movement of water on, above, and below the Earth's surface. It includes processes like evaporation, condensation, precipitation, and runoff. Water changes between liquid, solid, and gas states as it moves through these processes.",
  "suggestedQuestions": [
    "What is evaporation?",
    "How does condensation form clouds?",
    "What are the different forms of precipitation?"
  ],
  "references": [
    {
      "text": "The water cycle describes how water moves through different states and locations on Earth.",
      "source": "National Geographic"
    }
  ],
  "metadata": {
    "processingTime": "1.45s",
    "model": "gemini-1.5-flash-latest",
    "tokensUsed": 245
  }
}
```

## Testing

### Unit Tests

Run the test suite:

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
pytest
```

### Integration Tests

Test the API endpoints:

```bash
# Test evaluate-answer endpoint
python test_evaluate_answer_endpoint.py

# Test generate-questions endpoint
python test_generate_questions_endpoint.py

# Test chat endpoint
python test_chat_endpoint.py

# Run all integration tests
python -m unittest discover -s tests
```

## Deployment

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CORE_API_ACCESS_KEY` | Yes | - | API key for authentication |
| `GEMINI_API_KEY` | Yes | - | Google Gemini API key |
| `FLASK_APP` | No | `app.py` | Flask application entry point |
| `FLASK_ENV` | No | `development` | Environment (development/production) |
| `PORT` | No | `8000` | Port to run the server on |
| `AI_MODEL` | No | `gemini-1.5-flash-latest` | AI model to use |

### Production Deployment

1. **Using Gunicorn**
   ```bash
   gunicorn -w 4 -b 0.0.0.0:8000 app:app --timeout 120
   ```

2. **Using Docker**
   ```bash
   # Build the image
   docker build -t elevate-ai-service .
   
   # Run the container
   docker run -d \
     --name elevate-ai \
     -p 8000:8000 \
     --env-file .env \
     elevate-ai-service
   ```

3. **Using Docker Compose**
   ```yaml
   version: '3.8'
   
   services:
     ai-service:
       build: .
       ports:
         - "8000:8000"
       env_file:
         - .env
       restart: unless-stopped
   ```

## Troubleshooting

### Common Issues

1. **API Key Issues**
   - Ensure the `CORE_API_ACCESS_KEY` is set correctly
   - Verify the `GEMINI_API_KEY` is valid and has the necessary permissions

2. **Connection Refused**
   - Make sure the server is running
   - Check if the port is not blocked by a firewall

3. **Module Not Found**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check your Python version (requires 3.8+)

### Logs

View logs with:

```bash
# For Docker
# docker logs -f elevate-ai

# For direct execution
# tail -f nohup.out  # If using nohup
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Support

For support, please email support@elevate-ai.com or open an issue in the GitHub repository.
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
