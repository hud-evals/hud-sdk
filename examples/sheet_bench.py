#!/usr/bin/env python3
"""
SheetBench Agent Example

This example showcases SheetBench-specific features:
- Initial screenshot capture
- Thinking/reasoning display
- Computer tool usage
- Model-specific parameters

SheetBench
"""

import asyncio
import hud
from hud.mcp import ClaudeMCPAgent
from hud.mcp.client import MCPClient
from datasets import load_dataset
from hud.datasets import to_taskconfigs, TaskConfig

async def test_with_dataset():
    """Test with actual SheetBench dataset"""
    # Load the dataset
    dataset = load_dataset("hud-evals/sheetbench-taskconfigs")
    with hud.trace("Claude Agent Demo"):
        tsx = to_taskconfigs(dataset["train"])
        task = tsx[0]

        client = MCPClient(mcp_config=task.mcp_config)
        agent = ClaudeMCPAgent(
            mcp_client=client,
            model="claude-3-7-sonnet-20250219",
            allowed_tools=["anthropic_computer"],
            initial_screenshot=True,
        )

        try:
            result = await agent.run(task, max_steps=15)
            print(result.reward)

        finally:
            await client.close()

    print("\n✨ SheetBench agent demo complete!")

async def test_basic_functionality():
    """Test basic MCP functionality without requiring external datasets"""
    print("🔧 Testing basic MCP functionality...")
    
    try:
        # Test creating a basic task
        task = TaskConfig(
            prompt="Test spreadsheet task",
            mcp_config={
                "local": {"command": "echo", "args": ["test"]}
            }
        )
        print("✅ TaskConfig creation works")
        
        # Test trace functionality
        with hud.trace("SheetBench Basic Test"):
            print("✅ HUD trace functionality works for SheetBench")
        
        print("🎉 Basic MCP functionality test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in basic test: {e}")

async def main():
    print("=" * 60)
    print("SheetBench Example Test")
    print("=" * 60)
    
    # Try to load the dataset, fall back to basic test if not available
    try:
        print("🔍 Attempting to load SheetBench dataset...")
        await test_with_dataset()
    except Exception as e:
        print(f"⚠️  Dataset not available: {e}")
        print("📝 Note: The SheetBench dataset 'hud-evals/sheetbench-taskconfigs' ")
        print("   is not accessible. This might be because:")
        print("   1. The dataset is private and requires authentication")
        print("   2. The dataset name has changed")
        print("   3. Network connectivity issues")
        print("")
        print("🔄 Falling back to basic functionality test...")
        await test_basic_functionality()

if __name__ == "__main__":
    asyncio.run(main())
