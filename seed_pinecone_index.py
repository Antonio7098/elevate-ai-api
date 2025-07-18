#!/usr/bin/env python3
"""
Script to seed Pinecone index with sample blueprint content for RAG retrieval validation.
"""

import asyncio
import os
from dotenv import load_dotenv
from app.core.blueprint_parser import BlueprintParser
from app.core.indexing_pipeline import IndexingPipeline
from app.core.vector_store import create_vector_store
from app.core.embeddings import GoogleEmbeddingService
from app.models.learning_blueprint import (
    LearningBlueprint, 
    Section, 
    KnowledgePrimitives,
    Proposition, 
    Entity, 
    Process, 
    Relationship, 
    Question
)
from app.models.text_node import LocusType, UUEStage

# Load environment variables
load_dotenv(override=True)

class PineconeSeeder:
    """Seed Pinecone index with sample blueprint content."""
    
    def __init__(self):
        self.vector_store = None
        self.embedding_service = None
        self.indexing_pipeline = None
        
    async def initialize_services(self):
        """Initialize vector store and embedding services."""
        print("🚀 Initializing services for Pinecone seeding...")
        
        # Initialize global embedding service (required for IndexingPipeline)
        from app.core.embeddings import initialize_embedding_service
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        await initialize_embedding_service(service_type="google", api_key=google_api_key)
        
        # Initialize Pinecone vector store
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_env = os.getenv("PINECONE_ENVIRONMENT", "us-east1-gcp")
        
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
            
        self.vector_store = create_vector_store(
            store_type="pinecone",
            api_key=pinecone_api_key,
            environment=pinecone_env
        )
        
        await self.vector_store.initialize()
        
        # Initialize indexing pipeline
        self.indexing_pipeline = IndexingPipeline()
        
        print("✅ Services initialized successfully!")
        
    def create_sample_blueprint(self) -> LearningBlueprint:
        """Create a sample Python learning blueprint."""
        return LearningBlueprint(
            source_id="python-fundamentals-001",
            source_title="Python Programming Fundamentals",
            source_type="tutorial",
            source_summary={
                "core_thesis_or_main_argument": "Python is a beginner-friendly programming language that emphasizes readability and simplicity.",
                "inferred_purpose": "To teach fundamental Python concepts including variables, data types, operators, and control flow."
            },
            sections=[
                Section(
                    section_id="sec_intro",
                    section_name="Introduction to Python",
                    description="Overview of Python programming language, its history, and key characteristics.",
                    parent_section_id=None
                ),
                Section(
                    section_id="sec_basics",
                    section_name="Python Basics",
                    description="Core concepts including variables, data types, and operators.",
                    parent_section_id=None
                ),
                Section(
                    section_id="sec_control",
                    section_name="Control Flow",
                    description="Conditional statements and loops for program control.",
                    parent_section_id=None
                ),
                Section(
                    section_id="sec_functions",
                    section_name="Functions and Modules",
                    description="Creating reusable code with functions and organizing code with modules.",
                    parent_section_id=None
                )
            ],
            knowledge_primitives=KnowledgePrimitives(
                key_propositions_and_facts=[
                    Proposition(
                        id="prop_python_readable",
                        statement="Python emphasizes code readability and allows programmers to express concepts in fewer lines of code than languages like C++ or Java.",
                        supporting_evidence=["Python's design philosophy", "Comparative code examples"],
                        sections=["sec_intro"]
                    ),
                    Proposition(
                        id="prop_dynamic_typing",
                        statement="Python variables are dynamically typed, meaning you don't need to declare the type explicitly.",
                        supporting_evidence=["Python language specification", "Runtime behavior examples"],
                        sections=["sec_basics"]
                    ),
                    Proposition(
                        id="prop_indentation",
                        statement="Python uses indentation (whitespace) to define code blocks instead of curly braces.",
                        supporting_evidence=["Python syntax rules", "PEP 8 style guide"],
                        sections=["sec_control"]
                    )
                ],
                key_entities_and_definitions=[
                    Entity(
                        id="entity_variable",
                        entity="Variable",
                        definition="A container for storing data values in Python. Created when you assign a value to it.",
                        category="Concept",
                        sections=["sec_basics"]
                    ),
                    Entity(
                        id="entity_function",
                        entity="Function",
                        definition="A reusable block of code that performs a specific task, defined using the 'def' keyword.",
                        category="Concept",
                        sections=["sec_functions"]
                    ),
                    Entity(
                        id="entity_module",
                        entity="Module",
                        definition="A file containing Python code that can be imported and used in other Python programs.",
                        category="Concept",
                        sections=["sec_functions"]
                    )
                ],
                described_processes_and_steps=[
                    Process(
                        id="process_create_function",
                        process_name="Creating a Python Function",
                        steps=[
                            "Use the 'def' keyword followed by function name",
                            "Define parameters in parentheses",
                            "Add a colon and indent the function body",
                            "Optionally use 'return' to return a value"
                        ],
                        sections=["sec_functions"]
                    )
                ],
                identified_relationships=[
                    Relationship(
                        id="rel_variables_types",
                        relationship_type="uses",
                        source_primitive_id="entity_variable",
                        target_primitive_id="prop_dynamic_typing",
                        description="Python variables utilize dynamic typing system",
                        sections=["sec_basics"]
                    )
                ],
                implicit_and_open_questions=[
                    Question(
                        id="q_when_use_functions",
                        question="When should I create a function instead of writing code directly?",
                        sections=["sec_functions"]
                    )
                ]
            )
        )
    
    def create_ml_blueprint(self) -> LearningBlueprint:
        """Create a sample Machine Learning blueprint."""
        return LearningBlueprint(
            source_id="ml-fundamentals-001",
            source_title="Machine Learning Fundamentals",
            source_type="course",
            source_summary={
                "core_thesis_or_main_argument": "Machine Learning enables computers to learn patterns from data without explicit programming.",
                "inferred_purpose": "To provide a comprehensive introduction to machine learning concepts, algorithms, and evaluation methods."
            },
            sections=[
                Section(
                    section_id="sec_ml_intro",
                    section_name="Introduction to Machine Learning",
                    description="Overview of machine learning, its relationship to AI, and basic concepts.",
                    parent_section_id=None
                ),
                Section(
                    section_id="sec_supervised",
                    section_name="Supervised Learning",
                    description="Learning from labeled data including classification and regression.",
                    parent_section_id=None
                ),
                Section(
                    section_id="sec_unsupervised",
                    section_name="Unsupervised Learning",
                    description="Finding patterns in unlabeled data through clustering and dimensionality reduction.",
                    parent_section_id=None
                ),
                Section(
                    section_id="sec_evaluation",
                    section_name="Model Evaluation",
                    description="Methods and metrics for assessing machine learning model performance.",
                    parent_section_id=None
                )
            ],
            knowledge_primitives=KnowledgePrimitives(
                key_propositions_and_facts=[
                    Proposition(
                        id="prop_ml_definition",
                        statement="Machine Learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
                        supporting_evidence=["Academic definitions", "Industry applications"],
                        sections=["sec_ml_intro"]
                    ),
                    Proposition(
                        id="prop_supervised_uses_labels",
                        statement="Supervised learning uses labeled training data to learn a mapping from input features to output labels.",
                        supporting_evidence=["Training methodology", "Algorithm examples"],
                        sections=["sec_supervised"]
                    ),
                    Proposition(
                        id="prop_unsupervised_no_labels",
                        statement="Unsupervised learning finds patterns in data without labeled examples.",
                        supporting_evidence=["Clustering examples", "Dimensionality reduction techniques"],
                        sections=["sec_unsupervised"]
                    )
                ],
                key_entities_and_definitions=[
                    Entity(
                        id="entity_overfitting",
                        entity="Overfitting",
                        definition="When a model learns the training data too well, including noise and random fluctuations, resulting in poor performance on new, unseen data.",
                        category="Concept",
                        sections=["sec_evaluation"]
                    ),
                    Entity(
                        id="entity_classification",
                        entity="Classification",
                        definition="A type of supervised learning that predicts categorical outcomes or class labels.",
                        category="Concept",
                        sections=["sec_supervised"]
                    ),
                    Entity(
                        id="entity_clustering",
                        entity="Clustering",
                        definition="An unsupervised learning technique that groups similar data points together.",
                        category="Concept",
                        sections=["sec_unsupervised"]
                    )
                ],
                described_processes_and_steps=[
                    Process(
                        id="process_ml_workflow",
                        process_name="Machine Learning Model Development",
                        steps=[
                            "Define the problem and collect data",
                            "Preprocess and clean the data",
                            "Choose appropriate algorithm",
                            "Train the model on training data",
                            "Evaluate model performance",
                            "Deploy and monitor the model"
                        ],
                        sections=["sec_ml_intro", "sec_evaluation"]
                    )
                ],
                identified_relationships=[
                    Relationship(
                        id="rel_ml_ai",
                        relationship_type="subset_of",
                        source_primitive_id="prop_ml_definition",
                        target_primitive_id="entity_classification",
                        description="Machine Learning is a subset of Artificial Intelligence",
                        sections=["sec_ml_intro"]
                    ),
                    Relationship(
                        id="rel_supervised_classification",
                        relationship_type="includes",
                        source_primitive_id="prop_supervised_uses_labels",
                        target_primitive_id="entity_classification",
                        description="Supervised learning includes classification tasks",
                        sections=["sec_supervised"]
                    )
                ],
                implicit_and_open_questions=[
                    Question(
                        id="q_algorithm_selection",
                        question="How do I choose the right machine learning algorithm for my specific problem?",
                        sections=["sec_supervised", "sec_unsupervised"]
                    ),
                    Question(
                        id="q_prevent_overfitting",
                        question="What techniques can I use to prevent overfitting in my models?",
                        sections=["sec_evaluation"]
                    )
                ]
            )
        )
    
    async def seed_index(self):
        """Seed the Pinecone index with sample blueprint content."""
        print("🌱 Starting Pinecone index seeding...")
        
        try:
            # Create sample blueprints
            python_blueprint = self.create_sample_blueprint()
            ml_blueprint = self.create_ml_blueprint()
            
            blueprints = [python_blueprint, ml_blueprint]
            
            # Process each blueprint
            for i, blueprint in enumerate(blueprints, 1):
                print(f"\n📚 Processing blueprint {i}/{len(blueprints)}: {blueprint.source_title}")
                
                # Index the blueprint
                result = await self.indexing_pipeline.index_blueprint(blueprint)
                
                if result.get("indexing_completed"):
                    print(f"   ✅ Successfully indexed {result.get('processed_nodes', 0)} nodes")
                    print(f"   ✅ Blueprint ID: {result.get('blueprint_id')}")
                else:
                    print(f"   ❌ Failed to index blueprint: {result.get('error', 'Unknown error')}")
            
            # Get index statistics
            print("\n📊 Index Statistics:")
            try:
                stats = await self.vector_store.get_stats("blueprint-nodes")
                print(f"   Total vectors: {stats.get('total_vector_count', 'N/A')}")
                print(f"   Dimension: {stats.get('dimension', 'N/A')}")
                print(f"   Index fullness: {stats.get('index_fullness', 'N/A')}")
            except Exception as e:
                print(f"   ⚠️  Could not retrieve stats: {e}")
                
            print("\n🎉 Pinecone index seeding completed!")
            
        except Exception as e:
            print(f"\n❌ Seeding failed: {e}")
            raise

async def main():
    """Main seeding function."""
    seeder = PineconeSeeder()
    
    try:
        await seeder.initialize_services()
        await seeder.seed_index()
        
        print("\n" + "="*60)
        print("✅ Pinecone index successfully seeded with sample content!")
        print("🚀 You can now run the RAG demo to test retrieval functionality.")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Seeding process failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
