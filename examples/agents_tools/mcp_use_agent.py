#!/usr/bin/env python3
"""MCP-Use agent example for HUD tools via HTTP."""

import asyncio
from dotenv import load_dotenv
import hud
from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient

load_dotenv()

# Configuration
BASE_URL = "http://localhost:8040/mcp"


async def main():
    """Run MCP-Use agent with HUD tools."""
    # Configure MCP client to connect to the router
    config = {
        "mcpServers": {
            "hud": {
                "url": BASE_URL
            }
        }
    }
    
    # Create client and agent
    client = MCPClient.from_dict(config)
    llm = ChatOpenAI(model="gpt-4o")
    agent = MCPAgent(llm=llm, client=client, max_steps=30, verbose=True)
    
    try:
        # Run the agent
        query = "Take a screenshot, then use bash to show the current date and time"
        print(f"\n🤖 Running: {query}\n")
        
        with hud.trace("mcp_use_agent"):
            result = await agent.run(query)
        print(f"\n✅ Result: {result}")
        
    finally:
        await client.close_all_sessions()


if __name__ == "__main__":
    print(f"🚀 Connecting to MCP router at {BASE_URL}")
    asyncio.run(main()) 