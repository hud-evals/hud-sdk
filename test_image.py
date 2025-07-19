#!/usr/bin/env python3
"""
Simple test to check if qacontroller:latest image works with gym.make
"""

import asyncio
from hud.types import CustomGym
import hud.gym as gym


async def test_image():
    """Test using qacontroller:latest image."""
    
    print("Testing qacontroller:latest image...")
    
    # Create gym with pre-built image
    test_gym = CustomGym(
        type="public",
        location="local",
        image_or_build_context="qacontroller:latest"
    )
    
    try:
        # This should use the existing image without building
        env = await gym.make(test_gym)
        print(f"Success! Created environment successfully")
        await env.close()
        return True
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_image())
    if success:
        print("✓ qacontroller:latest image works!")
    else:
        print("✗ qacontroller:latest image failed!")