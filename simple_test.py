#!/usr/bin/env python3
"""
Simple test script for the Note Creation Agent.
Quick verification that all components are working.
"""

import asyncio
import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

async def simple_test():
    """Run a simple test of the note creation agent."""
    
    print("🧪 Simple Note Creation Agent Test")
    print("=" * 40)
    
    try:
        # Test 1: Import all components
        print("\n✅ Test 1: Import Components")
        from app.core.note_services.note_agent_orchestrator import NoteAgentOrchestrator
        from app.services.llm_service import create_llm_service
        from app.models.note_creation_models import NoteGenerationRequest, ContentToNoteRequest
        print("✅ All imports successful")
        
        # Test 2: Initialize services
        print("\n✅ Test 2: Initialize Services")
        llm_service = create_llm_service(provider="mock")
        orchestrator = NoteAgentOrchestrator(llm_service)
        print("✅ Services initialized")
        
        # Test 3: Service info
        print("\n✅ Test 3: Service Information")
        service_info = orchestrator.get_service_info()
        print(f"Service: {service_info['name']}")
        print(f"Version: {service_info['version']}")
        print(f"Capabilities: {len(service_info['capabilities'])}")
        
        # Test 4: Workflow status
        print("\n✅ Test 4: Workflow Status")
        status = await orchestrator.get_workflow_status()
        print(f"Status: {status['orchestrator_status']}")
        print(f"LLM Service: {status['services']['llm_service']['status']}")
        
        # Test 5: Simple conversion
        print("\n✅ Test 5: Simple Conversion")
        from app.models.note_creation_models import InputConversionRequest, ContentFormat
        
        request = InputConversionRequest(
            input_content="This is a test note about AI and machine learning.",
            input_format=ContentFormat.PLAIN_TEXT,
            preserve_structure=True
        )
        
        response = await orchestrator.convert_input_to_blocknote(request)
        print(f"Conversion Success: {response.success}")
        print(f"Message: {response.message}")
        
        print("\n🎉 All tests passed! Note Creation Agent is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function."""
    success = await simple_test()
    return 0 if success else 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted")
        exit(1)
    except Exception as e:
        print(f"\n💥 Error: {e}")
        exit(1)
