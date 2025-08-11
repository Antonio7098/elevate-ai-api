The Intelligent Learning Companion: An Agentic RAG Mixture of Experts System
Our application is designed as an advanced learning and memorization companion, leveraging a sophisticated Agentic Retrieval-Augmented Generation (RAG) Mixture of Experts (MoE) system. Its core purpose is to provide highly personalized, adaptive, and effective learning experiences by intelligently interacting with users, discussing learning materials, facilitating comprehension through questions and deep dives, and enabling dynamic editing of notes and question sets.
1. Foundational Principles
Personalization at Scale: The system adapts its approach based on individual user progress, struggles, learning styles, and memory data.
Contextual Intelligence: Every interaction is informed by a rich, multi-layered memory system, now explicitly managed by a dedicated Context Assembly Agent, ensuring relevance and coherence.
Agentic Decision-Making: AI agents not only respond but also reason, plan, and execute multi-step tasks, proactively guiding the learning journey.
Modularity and Scalability: The Mixture of Experts architecture, enhanced by a specialized Context Assembly Agent, allows for highly performant and independently evolvable components.
2. Core Architectural Components
The system is comprised of the following key interconnected components:
2.1. User Interface (UI)
Chat-Centric Interaction: The primary mode of interaction is a dynamic chat interface.
Explicit Mode Selection: A crucial "Chat Options" dropdown menu provides users with explicit control over the agent's operating mode (e.g., "Walk-through," "Test," "Deep Dive," "Note Editing," "Smart Mode"). This dropdown serves as a direct input to the Routing Agent.
Active Mode Indicator: The UI prominently displays the currently active mode, ensuring the user is always aware of the agent's focus.
Agentic Cursor Style Editing: For modes like "Note Editing," the UI supports an "agentic cursor" experience, where agent-generated suggestions or modifications can be directly applied to the user's notes or question sets, potentially with visual highlights or diff views.
2.2. Input Pre-processing and Intent Detection
Before reaching the core intelligence, user inputs undergo initial parsing and contextual framing using the Short-Term Memory (last 10 messages). This stage extracts basic commands, keywords, and initial intent signals for the Routing Agent.
2.3. Routing Agent (Mixture of Experts Orchestrator)
This is the central intelligence hub, a specialized LLM responsible for directing the flow of information and control within the system.
Inputs:
User Query.
Short-Term Memory (last 10 messages).
Medium-Term Memory (the structured JSON object containing global state and current_mode_state).
Explicit Mode Selection (from UI).
Metadata from RAG (initial, light signals if pre-computed).
Decision-Making Logic:
Interprets explicit user-selected modes or infers the most appropriate mode.
Identifies the optimal Expert Agent(s) to handle the request.
Crucially, it also identifies the specific context requirements for the chosen Expert based on the detected mode and user query. This information is then passed to the newly introduced Context Assembly Agent.
Manages confidence scoring, fallbacks, and orchestration of multi-step processes.
2.4. Context Assembly Agent
This is a dedicated, specialized agent (or pipeline) responsible for gathering, cleaning, and formatting all necessary contextual information for the target Expert Agent. It operates as an intermediary between the Router and the Expert, performing advanced "Context Engineering."
Inputs:
The raw user query.
The medium_term_memory object (including current_mode_state).
The Router's decision, which includes:
The ID of the target Expert Agent.
Specific context requirements (e.g., "retrieve relevant blueprints for 'backpropagation'", "fetch user's last 3 quiz scores on 'calculus'", "get specific personal memories related to 'difficult concepts'").
Responsibilities:
Orchestrated Hybrid RAG Calls: This agent is central to implementing Hybrid RAG (GraphRAG). It executes targeted queries against both the Vector Database (Pinecone) for semantic similarity over unstructured text and the Knowledge Graph for explicit relational context. This involves:
Formulating queries for both data sources.
Retrieving semantically similar text chunks (from Pinecone).
Retrieving structured facts and relationships (from the Knowledge Graph).
Combining and prioritizing results from both sources.
Memory Retrieval: Fetches specific fields or nested objects from the Medium-Term Memory JSON.
Tool/API Calls: Invokes specialized tools (e.g., Quiz Progress Tracking API, SRS API) to retrieve specific data points required by the Expert.
Data Cleaning & Normalization: Ensures all retrieved data is consistent, free of irrelevant noise, and formatted correctly.
Context Compression/Summarization: Applies summarization techniques (potentially using a smaller, dedicated LLM for this task) to condense large volumes of retrieved text or chat history, ensuring it fits within the target Expert's context window and is token-efficient.
Structured Context Delivery: Packages all assembled, cleaned, and optimized context into a single, structured object (e.g., a dictionary) that the target Expert can easily consume.
Benefits: Reduces the burden on the Router and Experts, centralizes context management logic, improves efficiency, and enhances system modularity and robustness.
2.5. Expert Agents (Specialized LLMs/Pipelines)
Each expert is a highly specialized AI module, often an LLM fine-tuned or specifically prompted for its task.
Inputs: Now, Experts primarily receive their context directly from the Context Assembly Agent. This pre-processed context is highly relevant, clean, and optimized, allowing the Expert to focus purely on its core task. They also receive the current user query.
Responsibilities:
Q&A/Test Expert: Generates consolidation questions, evaluates answers, provides hints.
Deep Dive/Explanation Expert: Provides in-depth explanations, analogies, and detailed breakdowns of complex concepts.
Walk-through Expert: Guides the user sequentially through learning blueprints or specific notes.
Note/Question Set Editing Expert: Modifies user-created notes or question sets based on natural language commands ("agentic cursor" style).
Progress & Strategy Expert: Analyzes user performance and memory data to suggest personalized learning paths and review strategies.
Memory Integration Expert: Processes new user insights to integrate them into Long-Term User Memories.
Updates Medium-Term Memory: Experts continue to be responsible for updating the current_mode_state within the Medium-Term Memory as their tasks progress (e.g., Test Expert updates current_question_index).
2.6. Memory Systems
The backbone of personalization and intelligence, now orchestrated largely by the Context Assembly Agent for retrieval and by Experts for updates.
Short-Term Memory (Last 10 Messages): A simple conversational buffer for immediate context.
Medium-Term Memory (Structured JSON Object): Persistent, dynamic, and comprehensive state including global user learning state and the current_mode_state (mode-specific JSON schemas).
Long-Term Memory:
User Memories (Vector Database - Pinecone): Your existing Pinecone instance will continue to store vector embeddings of unstructured user notes, highlights, and episodic learning history for semantic similarity retrieval.
Learning Blueprints (Vector Database - Pinecone): Your existing Pinecone instance will store vector embeddings of chunks from your deconstructed source materials for semantic RAG.
Knowledge Graph (NEW COMPONENT - Neo4j): This is a new, crucial part of your Long-Term Memory.
Purpose: To explicitly represent entities (concepts, users, questions, skills) and their relationships, hierarchies, and dependencies. It enables multi-hop reasoning, provides precise factual grounding, enhances explainability, and facilitates intelligent content organization.
Structure:
Learning Blueprints Knowledge Graph (Curriculum Graph): Nodes like Concept, Blueprint, Section, QuestionSet. Relationships like IS_PREREQUISITE_FOR, COVERS_CONCEPT, ASSESSES_CONCEPT. This models the interconnections within your learning material.
User Memory Knowledge Graph (Learner Profile Graph): Nodes like User, LearnedConcept, StruggledConcept, QuizAttempt, Insight, LearningSession, PreferredLearningStyle. Relationships like MASTERED, STRUGGLES_WITH, TOOK_QUIZ, HAS_INSIGHT_ON, PREFERS. This models the user's personal learning journey and state.
Interconnection: The User Memory KG will link directly to the Learning Blueprints KG, connecting a user's struggles or mastery to specific concepts defined in your curriculum.
2.7. Tools and APIs
Specialized external functionalities invoked primarily by the Context Assembly Agent (for data retrieval) and sometimes directly by Expert Agents (for specific actions like generating a new question).
Question Generation API.
Note Editing API.
Quiz/Progress Tracking API.
Spaced Repetition System (SRS) API.
UI Interaction API.
3. Updated Workflow Example: User Asks a Question
User Input: "Can you explain backpropagation to me? I keep getting stuck on it."
Input Pre-processing: Extracts "explain," "backpropagation," "stuck."
Routing Agent:
Analyzes User Input, Short-Term Memory, and Medium-Term Memory (which shows struggles includes backpropagation, progress_understanding for it is low).
Decision: Route to Deep Dive/Explanation Expert.
Context Request: Instructs the Context Assembly Agent to prepare context including: "Relevant sections from 'Neural Networks' blueprint about 'backpropagation'", "User's progress_understanding for 'backpropagation'", "Any personal memories related to 'backpropagation' difficulties," and specifically requests relational context from the Knowledge Graph about prerequisites or common challenges for 'backpropagation'.
Context Assembly Agent:
Executes Hybrid RAG:
Performs semantic RAG queries on Pinecone (Learning Blueprints & User Memories) for general text relevant to "backpropagation" and user struggles.
Queries the Knowledge Graph (Neo4j) for explicit relationships (e.g., IS_PREREQUISITE_FOR, HAS_CHALLENGE) connected to the "Backpropagation" concept, and user-specific relational data.
Fetches progress_understanding.backpropagation from Medium-Term Memory.
Summarizes retrieved content (both textual and graph-derived) if extensive.
Formats all this into a structured context object for the Deep Dive Expert.
Deep Dive Expert Actions:
Receives the user query and the pre-assembled, optimized context from the Context Assembly Agent, now enriched with both semantic and relational knowledge.
Generates a tailored explanation, comprehensively addressing the user's struggle and current understanding, leveraging both textual and graphical insights.
Updates medium_term_memory.deep_dive_state (e.g., current_topic, depth_level).
(Optional) May recommend a mode switch to the Router (e.g., "suggest Test Mode after explanation").
Response Synthesis: The Deep Dive Expert's output is formatted.
User Output: The detailed explanation of backpropagation.