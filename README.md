# Elevate AI API

An AI-powered learning co-pilot designed to facilitate deep mastery of complex subjects through a structured and personalized learning environment.

## Overview

Elevate transforms raw educational content into intelligent, personalized learning experiences through its "Deconstruct & Synthesize" loop:

1. **Deconstruction**: AI analyzes raw text and transforms it into structured LearningBlueprints
2. **Synthesis**: Generates personalized notes, questions, and learning materials
3. **Memory**: Maintains comprehensive user cognitive profiles and knowledge bases

## Architecture

- **Backend**: Python 3.12+ with FastAPI
- **AI Framework**: LlamaIndex for RAG implementation
- **Vector Database**: Pinecone for semantic search
- **Authentication**: JWT Bearer token
- **Dependency Management**: Poetry

## Project Structure

```
elevate-ai-api/
├── app/
│   ├── api/
│   │   ├── endpoints.py      # FastAPI endpoints
│   │   └── schemas.py        # Pydantic request/response models
│   ├── core/
│   │   ├── config.py         # Application configuration
│   │   ├── deconstruction.py # Core deconstruction logic
│   │   ├── chat.py           # RAG-powered chat functionality
│   │   └── indexing.py       # Blueprint-to-node pipeline
│   ├── models/
│   │   └── learning_blueprint.py # LearningBlueprint Pydantic models
│   └── main.py               # FastAPI application
├── tests/                    # Unit tests
├── docs/                     # Documentation
├── pyproject.toml            # Poetry configuration
└── env.example               # Environment variables template
```

## Setup

1. **Install Poetry** (if not already installed):
   ```bash
   sudo apt install python3-poetry
   ```

2. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd elevate-ai-api
   poetry install
   ```

3. **Configure environment**:
   ```bash
   cp env.example .env
   # Edit .env with your actual API keys and configuration
   ```

4. **Run the application**:
   ```bash
   poetry run python -m app.main
   ```

## API Endpoints

### Core Endpoints

- `POST /api/v1/deconstruct` - Transform raw text into LearningBlueprint
- `POST /api/v1/chat/message` - RAG-powered conversational interface
- `POST /api/v1/generate/notes` - Generate personalized notes
- `POST /api/v1/generate/questions` - Create question sets
- `POST /api/v1/suggest/inline` - Real-time note-taking suggestions

### Health Check

- `GET /` - Root endpoint with API info
- `GET /health` - Health check endpoint

## Authentication

All API endpoints require authentication using a Bearer token:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" http://localhost:8000/api/v1/health
```

## Development

### Running Tests

```bash
poetry run pytest
```

### Code Formatting

```bash
poetry run black .
poetry run isort .
```

### Type Checking

```bash
poetry run mypy .
```

## Current Sprint Status

**Sprint 01: Initial Project Setup** ✅
- Project structure established
- FastAPI application configured
- Basic endpoints and schemas created
- LearningBlueprint Pydantic models defined
- Security and configuration implemented

**Next: Sprint 02: Core Deconstruction Engine**
- Implement specialist agent functions
- Build the `/deconstruct` endpoint logic
- Add testing and validation

## Contributing

1. Follow the sprint-based development approach
2. Ensure all code is properly tested
3. Use type hints and follow PEP 8
4. Update documentation as needed

## License

[Add your license information here] 