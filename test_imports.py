#!/usr/bin/env python3
"""
Test Import Verification Script
Verifies that all test modules can be imported correctly.
"""

import sys
import os

def test_imports():
    """Test importing all test modules."""
    print("🔍 Testing Import Verification")
    print("=" * 50)
    
    # Add current directory to Python path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    test_modules = [
        ("tests.services.test_llm_services", "LLM Services"),
        ("tests.services.test_vector_stores", "Vector Stores"),
        ("tests.services.test_rag_search", "RAG & Search"),
        ("tests.services.test_note_services", "Note Services"),
        ("tests.services.test_premium_agents", "Premium Agents"),
        ("tests.api.test_api_endpoints", "API Endpoints")
    ]
    
    all_imports_successful = True
    
    for module_name, display_name in test_modules:
        try:
            print(f"   📦 Importing {display_name}...")
            __import__(module_name)
            print(f"   ✅ {display_name} imported successfully")
        except ImportError as e:
            print(f"   ❌ {display_name} import failed: {e}")
            all_imports_successful = False
        except Exception as e:
            print(f"   ❌ {display_name} unexpected error: {e}")
            all_imports_successful = False
    
    print("\n" + "=" * 50)
    if all_imports_successful:
        print("🎉 All test modules imported successfully!")
        print("✅ The test suite is ready to run.")
    else:
        print("⚠️  Some test modules failed to import.")
        print("❌ Please check the errors above before running tests.")
    
    return all_imports_successful


if __name__ == "__main__":
    test_imports()




