#!/usr/bin/env python3
"""
Simple test script for the LangChain-based Blueprint Editing Agent.
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import the agent
from app.core.premium.agents.blueprint_editing_agent import BlueprintEditingAgent

async def test_langchain_agent():
    """Test the LangChain agent with a simple operation."""
    print("🚀 Testing LangChain Blueprint Editing Agent...")
    
    try:
        # Initialize the agent
        agent = BlueprintEditingAgent()
        print("✅ Agent initialized successfully")
        
        # Test a simple granular edit operation
        content = "Sample blueprint content with sections"
        parameters = {"section_name": "Advanced Topics", "content": "Advanced concepts and applications"}
        
        print("🔧 Testing granular edit operation...")
        result = await agent.execute_granular_edit(
            edit_type="add_section",
            content=content,
            parameters=parameters
        )
        
        print(f"📝 Result: {result}")
        
        if "successfully" in result.lower():
            print("✅ Granular edit test passed!")
            return True
        else:
            print("❌ Granular edit test failed!")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {str(e)}")
        return False

async def main():
    """Main test function."""
    print("🧪 LangChain Agent Simple Test")
    print("=" * 40)
    
    success = await test_langchain_agent()
    
    if success:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n❌ Test failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
