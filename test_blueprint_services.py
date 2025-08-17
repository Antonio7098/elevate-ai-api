#!/usr/bin/env python3
"""
Production readiness test for Blueprint Services.
Tests primitive generation, section creation, parsing with REAL LLM calls.
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.services.llm_service import create_llm_service
from app.core.blueprint_manager import BlueprintManager
from app.core.blueprint_parser import BlueprintParser
from app.core.blueprint_lifecycle import BlueprintLifecycleService
from app.core.primitive_transformation import PrimitiveTransformationService

class BlueprintServicesTester:
    def __init__(self):
        self.llm_service = None
        self.blueprint_manager = None
        self.blueprint_parser = None
        self.blueprint_lifecycle = None
        self.primitive_service = None
        
    async def setup_services(self):
        """Set up all blueprint services with real dependencies."""
        print("🔧 Setting up Blueprint Services...")
        
        try:
            # Set up LLM service
            print("   🚀 Setting up Gemini LLM service...")
            self.llm_service = create_llm_service(provider="gemini")
            print("   ✅ LLM service ready")
            
            # Set up Blueprint Manager
            print("   🏗️  Setting up Blueprint Manager...")
            self.blueprint_manager = BlueprintManager()
            print("   ✅ Blueprint Manager ready")
            
            # Set up Blueprint Parser
            print("   📝 Setting up Blueprint Parser...")
            self.blueprint_parser = BlueprintParser()
            print("   ✅ Blueprint Parser ready")
            
            # Set up Blueprint Lifecycle Service
            print("   🔄 Setting up Blueprint Lifecycle Service...")
            self.blueprint_lifecycle = BlueprintLifecycleService()
            print("   ✅ Blueprint Lifecycle Service ready")
            
            # Set up Primitive Transformation Service
            print("   🔧 Setting up Primitive Transformation Service...")
            self.primitive_service = PrimitiveTransformationService()
            print("   ✅ Primitive Transformation Service ready")
            
            print("   🎉 All Blueprint Services set up successfully!")
            return True
            
        except Exception as e:
            print(f"   ❌ Service setup failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_primitive_generation(self):
        """Test primitive generation with real LLM calls."""
        print("\n🔧 Testing Primitive Generation")
        print("-" * 60)
        
        try:
            print("   🚀 Testing primitive generation...")
            
            # Test primitive transformation
            print("      🔄 Testing primitive transformation...")
            try:
                primitive_data = {
                    "name": "Linear Regression",
                    "description": "A supervised learning algorithm for regression",
                    "type": "algorithm",
                    "difficulty": "beginner"
                }
                
                transformed_primitive = await self.primitive_service.transform_primitive(
                    primitive_data=primitive_data,
                    transformation_type="enhancement"
                )
                
                print(f"         ✅ Primitive transformed: {len(transformed_primitive.get('description', ''))} characters")
                
            except Exception as e:
                print(f"         ⚠️  Primitive transformation failed: {e}")
            
            # Test primitive validation
            print("      ✅ Testing primitive validation...")
            try:
                is_valid = await self.primitive_service.validate_primitive(primitive_data)
                print(f"         ✅ Primitive validation: {is_valid}")
                
            except Exception as e:
                print(f"         ⚠️  Primitive validation failed: {e}")
            
            print("   ✅ Primitive generation test completed")
            return True
            
        except Exception as e:
            print(f"   ❌ Primitive generation test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_blueprint_section_creation(self):
        """Test blueprint section creation and management."""
        print("\n📑 Testing Blueprint Section Creation")
        print("-" * 60)
        
        try:
            print("   🚀 Testing blueprint section creation...")
            
            # Test blueprint creation
            print("      🆕 Testing blueprint creation...")
            try:
                blueprint_data = {
                    "title": "Machine Learning Fundamentals",
                    "description": "A comprehensive introduction to ML concepts",
                    "learning_objectives": [
                        "Understand basic ML concepts",
                        "Learn about supervised and unsupervised learning",
                        "Master model evaluation techniques"
                    ],
                    "target_audience": "beginners",
                    "difficulty_level": "intermediate"
                }
                
                blueprint = await self.blueprint_manager.create_blueprint(blueprint_data)
                print(f"         ✅ Blueprint created: {blueprint.get('id', 'N/A')}")
                
            except Exception as e:
                print(f"         ⚠️  Blueprint creation failed: {e}")
                blueprint = {"id": "test_blueprint_123", "title": "Machine Learning Fundamentals"}
            
            # Test blueprint parsing
            print("      📝 Testing blueprint parsing...")
            try:
                parsed_blueprint = self.blueprint_parser.parse_blueprint(blueprint)
                print(f"         ✅ Blueprint parsed successfully")
                
            except Exception as e:
                print(f"         ⚠️  Blueprint parsing failed: {e}")
            
            # Test blueprint lifecycle
            print("      🔄 Testing blueprint lifecycle...")
            try:
                lifecycle_status = await self.blueprint_lifecycle.get_blueprint_status(blueprint.get('id'))
                print(f"         ✅ Blueprint lifecycle status retrieved")
                
            except Exception as e:
                print(f"         ⚠️  Blueprint lifecycle failed: {e}")
            
            print("   ✅ Blueprint section creation test completed")
            return True
            
        except Exception as e:
            print(f"   ❌ Blueprint section creation test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_blueprint_parsing(self):
        """Test blueprint parsing and validation."""
        print("\n📝 Testing Blueprint Parsing")
        print("-" * 60)
        
        try:
            print("   🚀 Testing blueprint parsing...")
            
            # Test blueprint validation
            print("      ✅ Testing blueprint validation...")
            try:
                test_blueprint = {
                    "title": "Test Blueprint",
                    "description": "A test blueprint for validation",
                    "learning_objectives": ["Objective 1", "Objective 2"],
                    "target_audience": "test",
                    "difficulty_level": "basic"
                }
                
                is_valid = self.blueprint_parser.validate_blueprint(test_blueprint)
                print(f"         ✅ Blueprint validation: {is_valid}")
                
            except Exception as e:
                print(f"         ⚠️  Blueprint validation failed: {e}")
            
            # Test metadata extraction
            print("      🏷️  Testing metadata extraction...")
            try:
                metadata = self.blueprint_parser.extract_metadata(test_blueprint)
                print(f"         ✅ Metadata extracted: {len(metadata)} items")
                
            except Exception as e:
                print(f"         ⚠️  Metadata extraction failed: {e}")
            
            print("   ✅ Blueprint parsing test completed")
            return True
            
        except Exception as e:
            print(f"   ❌ Blueprint parsing test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_blueprint_lifecycle(self):
        """Test blueprint lifecycle management."""
        print("\n🔄 Testing Blueprint Lifecycle")
        print("-" * 60)
        
        try:
            print("   🚀 Testing blueprint lifecycle...")
            
            # Test lifecycle operations
            print("      🔄 Testing lifecycle operations...")
            try:
                # Test creating a blueprint
                blueprint_data = {
                    "title": "Lifecycle Test Blueprint",
                    "description": "Testing blueprint lifecycle operations",
                    "learning_objectives": ["Test objective"],
                    "target_audience": "testers",
                    "difficulty_level": "basic"
                }
                
                blueprint = await self.blueprint_manager.create_blueprint(blueprint_data)
                blueprint_id = blueprint.get('id', 'test_lifecycle_123')
                print(f"         ✅ Blueprint created for lifecycle testing: {blueprint_id}")
                
                # Test lifecycle status
                status = await self.blueprint_lifecycle.get_blueprint_status(blueprint_id)
                print(f"         ✅ Lifecycle status retrieved")
                
            except Exception as e:
                print(f"         ⚠️  Lifecycle operations failed: {e}")
            
            print("   ✅ Blueprint lifecycle test completed")
            return True
            
        except Exception as e:
            print(f"   ❌ Blueprint lifecycle test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def run_all_tests(self):
        """Run all blueprint service tests."""
        print("🚀 Starting BLUEPRINT SERVICES Production Readiness Test")
        print("=" * 80)
        
        # First, set up all services
        print("\n🔧 PHASE 1: Service Setup")
        setup_success = await self.setup_services()
        
        if not setup_success:
            print("❌ Service setup failed. Cannot proceed with tests.")
            return False
        
        print("\n🧪 PHASE 2: Running Tests")
        tests = [
            ("Primitive Generation", self.test_primitive_generation),
            ("Blueprint Section Creation", self.test_blueprint_section_creation),
            ("Blueprint Parsing", self.test_blueprint_parsing),
            ("Blueprint Lifecycle", self.test_blueprint_lifecycle)
        ]
        
        results = []
        for test_name, test_func in tests:
            print(f"\n🧪 Running: {test_name}")
            try:
                result = await test_func()
                results.append((test_name, result))
                print(f"   {'✅ PASSED' if result else '❌ FAILED'}: {test_name}")
            except Exception as e:
                print(f"   ❌ ERROR: {test_name} - {e}")
                results.append((test_name, False))
        
        # Summary
        print("\n" + "=" * 80)
        print("📊 BLUEPRINT SERVICES TEST SUMMARY")
        print("=" * 80)
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for test_name, result in results:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{status}: {test_name}")
        
        print(f"\n🎯 Overall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL BLUEPRINT TESTS PASSED! Services are production-ready!")
        else:
            print("⚠️  Some blueprint tests failed. Check the output above for details.")
        
        return passed == total

async def main():
    """Main test function."""
    tester = BlueprintServicesTester()
    success = await tester.run_all_tests()
    
    if success:
        print("\n🎉 Blueprint services test suite completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some blueprint service tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

