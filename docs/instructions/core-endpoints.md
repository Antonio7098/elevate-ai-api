# RAG API Integration Guide

This guide provides detailed instructions for interacting with the new RAG-powered AI endpoints in the Elevate Core API. These endpoints enable the creation of structured learning materials and facilitate contextual conversations.

**Authentication**: All endpoints described below require authentication. Requests must include an `Authorization` header with a valid JWT Bearer token:
`Authorization: Bearer <YOUR_JWT_TOKEN>`

---

## 1. Learning Blueprint Creation

This is the first step in any RAG-based content generation workflow. A client sends raw text, and the API deconstructs it into a structured `LearningBlueprint`, which serves as the foundation for generating other materials.

### `POST /api/ai-rag/learning-blueprints`

-   **Purpose**: Creates a new `LearningBlueprint` by deconstructing source text via the internal AI service.
-   **Request Body**: `CreateLearningBlueprintDto`

    ```json
    {
      "sourceText": "The mitochondria is the powerhouse of the cell. It generates most of the cell's supply of adenosine triphosphate (ATP), used as a source of chemical energy.",
      "folderId": 123
    }
    ```
    -   `sourceText` (string, required): The raw text content to be deconstructed.
    -   `folderId` (number, optional): The ID of the folder where the resulting blueprint should be stored.

-   **Success Response**: `201 Created`
    -   Returns a `LearningBlueprintResponseDto` object.

    ```json
    {
      "id": 1,
      "userId": 1,
      "sourceText": "The mitochondria is the powerhouse of the cell...",
      "blueprintJson": {
        "topic": "Cellular Biology",
        "key_concepts": [
          { "concept": "Mitochondria", "definition": "The powerhouse of the cell." },
          { "concept": "ATP", "definition": "A source of chemical energy." }
        ]
      },
      "folderId": 123,
      "createdAt": "2025-06-24T10:00:00.000Z",
      "updatedAt": "2025-06-24T10:00:00.000Z"
    }
    ```

-   **Error Responses**:
    -   `400 Bad Request`: Validation error (e.g., `sourceText` is missing or empty).
    -   `401 Unauthorized`: Invalid or missing authentication token.
    -   `502 Bad Gateway`: The internal AI service failed to process the request.

---

## 2. Generating Materials from a Blueprint

Once a `LearningBlueprint` is created, you can use its ID to generate specific learning materials like question sets or notes.

### `POST /api/ai-rag/learning-blueprints/:blueprintId/question-sets`

-   **Purpose**: Generates a new `QuestionSet` based on the content of a specific `LearningBlueprint`.
-   **Request Body**: `GenerateQuestionsFromBlueprintDto`

    ```json
    {
      "name": "Mitochondria Quiz",
      "folderId": 123,
      "questionOptions": {
        "scope": "KeyConcepts",
        "tone": "Formal"
      }
    }
    ```
    -   `name` (string, required): The title for the new `QuestionSet`.
    -   `folderId` (number, optional): The ID of the folder to store the new question set in. If omitted, it may be placed in a default location.
    -   `questionOptions` (object, optional): Additional parameters to guide the AI's question generation process.

-   **Success Response**: `200 OK`
    -   Returns the newly created `QuestionSet` object with associated questions.

-   **Error Responses**:
    -   `400 Bad Request`: Validation error (e.g., `name` is missing).
    -   `401 Unauthorized`: Invalid or missing authentication token.
    -   `404 Not Found`: The specified `blueprintId` does not exist or the user does not have access to it.
    -   `502 Bad Gateway`: The internal AI service failed to generate questions.

### `POST /api/ai-rag/learning-blueprints/:blueprintId/notes`

-   **Purpose**: Generates a new `Note` based on the content of a specific `LearningBlueprint`.
-   **Request Body**: `GenerateNoteFromBlueprintDto`

    ```json
    {
      "name": "Summary of Mitochondria",
      "folderId": 123
    }
    ```
    -   `name` (string, required): The title for the new `Note`.
    -   `folderId` (number, optional): The ID of the folder to store the new note in.

-   **Success Response**: `200 OK`
    -   Returns the newly created `Note` object.

-   **Error Responses**:
    -   `400 Bad Request`: Validation error (e.g., `name` is missing).
    -   `401 Unauthorized`: Invalid or missing authentication token.
    -   `404 Not Found`: The specified `blueprintId` does not exist or the user does not have access to it.
    -   `502 Bad Gateway`: The internal AI service failed to generate the note.

---

--- 

## 3. Internal AI Service Contract

This section outlines the expected request and response formats for the internal AI service endpoints that the Core API calls. Developers building the AI service should adhere to this contract.

### `POST /api/v1/deconstruct`

-   **Purpose**: Deconstructs raw text into a structured `LearningBlueprint` JSON.
-   **Request Body**:

    ```json
    {
      "source_text": "The mitochondria is the powerhouse of the cell."
    }
    ```

-   **Success Response Body**:

    ```json
...
    ```

### `POST /api/v1/generate/questions`

-   **Purpose**: Generates a set of questions from a `LearningBlueprint`.
-   **Request Body**:

    ```json
    {
      "blueprint_json": { /* ... LearningBlueprint JSON ... */ },
      "question_options": {
        "scope": "KeyConcepts",
        "tone": "Formal"
      }
    }
    ```

-   **Success Response Body**:

    ```json
    {
      "questions": [
        {
          "text": "What is the primary function of the mitochondria?",
          "answer": "It is the powerhouse of the cell, generating ATP.",
          "question_type": "understand/use/explore",
          "total_marks_available": 2,
          "marking_criteria": "Award 1 mark for 'powerhouse' and 1 mark for mentioning 'ATP'."
        }
      ]
    }
    ```

### `POST /api/v1/generate/notes`

-   **Purpose**: Generates a structured note from a `LearningBlueprint`.
-   **Request Body**:

    ```json
    {
      "source_text": "The full original text...",
      "blueprint_json": { /* ... LearningBlueprint JSON ... */ },
      "note_options": { /* ... any specific options ... */ }
    }
    ```

-   **Success Response Body**:

    ```json
    {
      "content": { /* ... JSON content for the note (e.g., TipTap format) ... */ },
      "plain_text": "A summary of the key points about mitochondria..."
    }
    ```

### `POST /api/v1/chat`

-   **Purpose**: Handles a conversational turn, potentially with context.
-   **Request Body**:

    ```json
    {
      "user_id": 123,
      "message_content": "Explain this in simpler terms.",
      "chat_history": [
        { "role": "user", "content": "What is ATP?" },
        { "role": "assistant", "content": "ATP is adenosine triphosphate..." }
      ],
      "context": {
        "blueprintId": 1
      }
    }
    ```

-   **Success Response Body**:

    ```json
    {
      "role": "assistant",
      "content": "Think of mitochondria as tiny power plants inside a cell..."
    }
    ```

---

## 4. Conversational Chat

The chat endpoint provides a unified interface for conversational interaction, with the ability to reference specific learning materials for context.

### `POST /api/ai-rag/chat/message`

-   **Purpose**: Sends a message to the AI assistant and receives a response. The conversation can be contextualized by referencing a `LearningBlueprint`, `QuestionSet`, `Folder`, or other items.
-   **Request Body**: `ChatMessageDto`

    *Example 1: General question*
    ```json
    {
      "messageContent": "What is the capital of Spain?"
    }
    ```

    *Example 2: Question contextualized to a Learning Blueprint*
    ```json
    {
      "messageContent": "Explain this in simpler terms.",
      "context": {
        "blueprintId": 1
      }
    }
    ```
    -   `messageContent` (string, required): The user's message.
    -   `context` (object, optional): An object specifying the context for the conversation.
        -   `blueprintId` (number): ID of a `LearningBlueprint`.
        -   `questionSetId` (number): ID of a `QuestionSet`.
        -   `folderId` (number): ID of a `Folder`.

-   **Success Response**: `200 OK`
    -   Returns a chat response object.

    ```json
    {
      "role": "assistant",
      "content": "The capital of Spain is Madrid."
    }
    ```

-   **Error Responses**:
    -   `400 Bad Request`: Validation error (e.g., `messageContent` is missing).
    -   `401 Unauthorized`: Invalid or missing authentication token.
    -   `404 Not Found`: A resource specified in the `context` (e.g., `blueprintId`) does not exist or the user does not have access to it.
    -   `502 Bad Gateway`: The internal AI service failed to process the chat message.