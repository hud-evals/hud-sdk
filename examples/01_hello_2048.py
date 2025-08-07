#!/usr/bin/env python3
"""
Hello 2048 - Simplest Game Agent Example

This example shows the most basic usage of the HUD SDK with a local game:
- Running a Docker-based MCP environment (2048 game)
- Creating an MCP client for local execution
- Playing a simple game with an agent
- Using hud.trace() for telemetry

Prerequisites:
- Docker installed and running
- pip install hud-python
- Build the 2048 image: docker build -t hud-text-2048 environments/text_2048
"""

# Alternative test without Docker - comment out the Docker version
# Original Docker version requires: docker build -t hud-text-2048 environments/text_2048
# 
# import asyncio
# import hud
# from hud.datasets import TaskConfig
# from hud.mcp import ClaudeMCPAgent, MCPClient

# Simple test to verify the HUD SDK works
import asyncio
import hud
print("✅ Testing HUD SDK import - SUCCESS")

# Original example that requires Docker:
async def docker_version():
    """This requires Docker to be installed and the 2048 image to be built"""
    from hud.datasets import TaskConfig
    from hud.mcp import ClaudeMCPAgent, MCPClient
    task = TaskConfig(
        prompt="Play 2048 and try to get as high as possible. Do not stop even after 2048 is reached.",
        mcp_config={
            "local": {"command": "docker", "args": ["run", "--rm", "-i", "hud-text-2048"]}
        },
        setup_tool={
            "name": "setup",
            "arguments": {"function": "board", "args": {"board_size": 4}},
        },
        evaluate_tool={
            "name": "evaluate",
            "arguments": {"function": "max_number"},
        },
    )

    # Create client and agent
    client = MCPClient(mcp_config=task.mcp_config)
    agent = ClaudeMCPAgent(
        mcp_client=client,
        model="claude-3-7-sonnet-20250219",
        allowed_tools=["move"],  # let the agent only use the move tool
    )

    # Simple trace for telemetry
    with hud.trace("Hello 2048 Game"):
        try:
            print("\n🤖 Agent playing 2048...")
            result = await agent.run(task, max_steps=-1)

            print(f"\n✅ Game session completed!")
            print(f"   Reward: {result.reward}")

        finally:
            await client.close()

# Test the basic SDK functionality
async def test_basic_hud():
    """Test basic HUD functionality without requiring Docker"""
    print("🔧 Testing basic HUD functionality...")
    
    try:
        # Test trace functionality
        with hud.trace("Basic HUD Test"):
            print("✅ HUD trace functionality works")
        
        # Test TaskConfig creation
        from hud.datasets import TaskConfig
        task = TaskConfig(
            prompt="Test task",
            mcp_config={"test": "config"}
        )
        print("✅ TaskConfig creation works")
        
        print("🎉 Basic HUD functionality test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in basic HUD test: {e}")

async def main():
    print("=" * 60)
    print("HUD SDK 2048 Example Test")
    print("=" * 60)
    print("Note: The full 2048 example requires Docker and a built image.")
    print("To run the full example:")
    print("1. Install Docker")
    print("2. Build image: docker build -t hud-text-2048 environments/text_2048") 
    print("3. Uncomment the docker_version() call below")
    print("=" * 60)
    
    # Test basic functionality first
    await test_basic_hud()
    
    # Uncomment below to test with Docker (requires Docker + built image)
    # await docker_version()


if __name__ == "__main__":
    print("=" * 40)
    print("This example runs a local 2048 game in Docker.")
    print("=" * 40)
    asyncio.run(main())
