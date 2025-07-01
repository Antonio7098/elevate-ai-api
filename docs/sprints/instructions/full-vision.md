Elevate is an AI-powered learning co-pilot designed to facilitate deep mastery of complex subjects through a structured and personalized learning environment.


Core Mechanic: The "Deconstruct & Synthesize" Loop
Deconstruction (Analysis & Blueprint Generation): The AI analyzes raw text (e.g., lecture notes) and transforms it into a structured JSON object called the LearningBlueprint. This blueprint contains foundational concepts, use cases, explorations, key terms, common misconceptions, and relationship maps. It is saved in a PostgreSQL database. Key agents involved include a Dispatcher, Specialist Deconstructors, a UUE_Auditor, and a JSON_Finalizer.
Synthesis (Personalized Content Generation): Using the LearningBlueprint and user preferences, the system synthesizes tailored notes and questions.
User's "Brain": The Long-Term Memory System


A comprehensive Cognitive Profile for each user is stored in PostgreSQL to personalize the experience. This includes:
Core Profile (`UserMemory`): Stores explicit user preferences (learning style, goals, AI tone).
Inferred Knowledge Base: AI-populated tables tracking user knowledge (`ConceptMastery`), recurring mistakes (`UserErrorPattern`), and general context learned from chats (`InferredGeneralContext`).
Personal Content Library & Progress: User-owned data like `QuestionSet` (for Spaced Repetition System) and editable `Note`s.
Intelligent Chatbot: A RAG-Powered Agent


The chat feature uses a multi-tier memory system:
Tier 1 (Short-Term): Conversational Buffer (last 5-10 messages).
Tier 2 (Mid-Term): Session State JSON (structured "scratchpad" for the current session), updated by a "Helper Agent."
Tier 3 (Long-Term): Knowledge Base (vector database of Learning Blueprint-indexed notes) and Cognitive Profile (PostgreSQL user data) for deep understanding.
"Coach" Role: Using Historical Data


The system leverages historical data for insights, including `UserQuestionAnswer` for feedback on mistakes, `UserStudySession` for effort tracking, and `masteryHistory` for progress visualization.












The Inline Co-pilot: Transforming the Note-Taking Experience
Beyond the main chat window, Elevate's agent extends directly into the note-taking process itself through a feature called the Inline Co-pilot. This transforms the editor from a static page into a dynamic, collaborative space.
The system operates on a highly efficient "Trigger, Search, Suggest" loop, which activates when the user pauses typing. It is not constantly streaming keystrokes.
Automatic Linking: As a user writes, the co-pilot can identify concepts and suggest creating a direct link to an existing Locus in their knowledge base (e.g., 🔗 Link to Locus: "The Marshall Plan"?).
Factual Augmentation: It can provide auto-completions for common facts or phrases (e.g., suggesting "the powerhouse of the cell" after a user types "The mitochondria is...").
Idea Expansion: If a user creates a new heading, the co-pilot can suggest a starting structure or a list of bullet points based on the context from the LearningBlueprint.




Technical Architecture


Elevate has a two-part backend system:
Core API (Node.js/NestJS): The "Conductor" handling users, security, PostgreSQL, and workflow orchestration.
Endpoints: /deconstruct, /generate/notes, /generate/questions, /chat (main conversational turn), and /suggest/inline (a new, highly-optimized, low-latency endpoint for real-time suggestions during note-taking).
Core Principle: "Intelligent Chunking" with Blueprints (RAG Implementation)


The AI-generated LearningBlueprint creates semantically meaningful "nodes" with rich metadata, stored in a vector database. This process involves:
Blueprint-to-Node Pipeline (Data Ingestion): Iterating through the blueprint to create `TextNode` objects enriched with metadata and indexing them in a vector store.
Advanced Retrieval with Metadata Filtering: Allowing specific retrievals based on metadata.
Prompt Assembly and Generation: Combining user profile, retrieved context, and chat history for the LLM.
Strategic Roadmap: Phased Implementation


The roadmap outlines three phases:
Phase 1: Foundational, High-Quality RAG Core: Focuses on accurate and relevant answers through intelligent ingestion, metadata filtering, hybrid search, and re-ranking.
Phase 2: Building the Intelligent & Trustworthy Agent: Aims for intelligent, proactive, and reliable AI with query transformations, self-correction, and basic tool use.
Phase 3: Advanced Synthesis & Long-Term Autonomy: Targets complex synthesis and graceful scaling with multi-hop reasoning, stateful agents, and context pruning.
Optimizing for Cost: Running Elevate Efficiently


Managing LLM token consumption is crucial for financial viability.


High-Cost Center 1: The "Deconstruction" Pipeline
Why expensive: High input/output tokens, multiple LLM calls.
Mitigation: Aggressive caching of source text hashes, model tiering (cheapest for simple tasks, powerful for complex), and efficient prompt engineering.
High-Cost Center 2: The "Synthesis" Phase (Note & Question Generation)
Why expensive: High output tokens, large context input.
Mitigation: Smart context provisioning (sending only relevant blueprint sections), rate limiting, and on-demand generation.
High-Cost Center 3: The Intelligent Chatbot
Why expensive: Volume, frequency, RAG overhead.
Mitigation: Optimizing the "Helper Agent" with cheapest models, efficient retrieval, and context pruning.
High-Cost Center 4: Data Ingestion & Vector Management
Why expensive: Embedding API calls, vector database hosting costs.
Mitigation: Choosing cost-effective embedding models, using quantization to compress vectors, and right-sizing the vector database instance.
High-Cost Center 5: The Inline Co-pilot
Why it's expensive: While each call is very small, the potential frequency is extremely high, as it can be triggered every few seconds for an active user. The cumulative cost could be significant.
Mitigation Strategies:
Smart Triggers: Use intelligent client-side logic to trigger the suggestion engine only on meaningful pauses (debouncing), not on every keystroke.
Aggressive Model Tiering: This endpoint must use the absolute cheapest and fastest model available (e.g., Gemini 1.5 Flash). The task is simple enough that a powerful model is unnecessary and would introduce too much latency and cost.
Efficient Retrieval: The RAG search for this feature must be optimized for speed, retrieving only the top 1-2 most relevant results to keep the context passed to the LLM minimal.

General Architectural Recommendations for Cost Control
Build a cost-monitoring dashboard to track LLM API calls.
Explore fine-tuning smaller open-source models for frequent, predictable tasks.
Set budgets and alerts using cloud provider billing tools.
Blueprint as a Knowledge Graph


The Blueprint is proposed to be treated as a graph data structure with Nodes (Loci) representing individual knowledge pieces and Edges (Pathways) defining relationships between them.


Proposed JSON Structure for a Pathway:
{
  "pathwayId": "P12",
  "sourceLocusId": "F1",
  "targetLocusId": "U1",
  "label": "Is demonstrated by",
  "type": "AI_GENERATED", // or 'USER_CREATED'
  "strength": 0.95
}
This pathways array is a new top-level array in the Blueprint JSON.


How This Fits Into the Overall Blueprint (Example Snippet):


The Blueprint will contain a `"loci"` array for knowledge pieces and a `"pathways"` array for the relationships between them.


Why This Approach is Powerful:
Creates a true Knowledge Graph, enabling rich, queryable data.
Powers the AI's reasoning by providing semantic context for "how" and "why" questions, allowing the AI to "walk" the pathways.
Drives visualization for mapping knowledge.
Enables user collaboration by distinguishing AI-generated from user-created connections.
Building the Knowledge Base: From Blueprint to Brain


Once the LearningBlueprint is generated, it constructs two complementary pillars of long-term memory:
Constructing the Vector Database (For Semantic Search): Iterates over `loci` in the Blueprint, creates `TextNode` objects with content and metadata (e.g., `locusId`, `locusType`, `uueStage`), and indexes them in a vector store.
Constructing the Knowledge Graph (For Relational Reasoning): Iterates through `loci` to create nodes and through `pathways` to create directed edges (`sourceLocusId`, `targetLocusId`, `label`) in a graph structure.
Synergy in Action:


These two systems work together. A user's query can find a relevant starting Locus in the Vector Database, and then the AI can traverse the Knowledge Graph's Pathways to gather deeper, connected context for more insightful answers.


Plan for Building the "Elevate" AI API (Python Service)


Part 1: Core AI API Architecture & Setup
Technology Stack: FastAPI, LangChain, LlamaIndex, vector database client (e.g., pinecone-client), httpx.
Project Structure: Standardized directory structure.
Security & Configuration: Environment variables, private network deployment, API key/bearer token.
Part 2: Phased Endpoint Implementation


Phase 1: Foundational, High-Quality RAG Core
Endpoint: `/deconstruct` (POST):
Request: `source_text`, `source_type_hint`.
Response: Validated JSON LearningBlueprint.
Logic: Dispatcher, Specialist Deconstructors, UUE_Auditor, JSON_Finalizer.
Background Process: Blueprint-to-Node Ingestion (POST `/index-blueprint`):
Request: `question_set_id`, `learning_blueprint`.
Logic: Parses blueprint, creates LlamaIndex TextNodes, indexes in vector store.
Endpoint: `/chat` (Phase 1 version) (POST):
Request: `question_set_id`, `user_id`, `chat_history` (Tier 1), `session_state` (Tier 2), `cognitive_profile` (Tier 3), `query`.
Response: `answer`, `retrieved_context`, `profile_updates`.
Logic: Retrieval

Sources (1)
Features: Query Transformations, Self-Correction (Chain-of-Verification), Prompt-based Style Control, Basic Tool Use, and Inline Co-pilot (implementing the real-time suggestion engine for the note editor).