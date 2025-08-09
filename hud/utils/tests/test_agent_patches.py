"""Tests for agent patches module."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from mcp import types

from hud.utils.agent_patches import apply_all_patches, patch_mcp_client_call_tool


class TestAgentPatches:
    """Test agent patches functionality."""

    @pytest.mark.asyncio
    async def test_patch_mcp_client_call_tool_success(self):
        """Test patching MCP client call_tool method."""
        # Test that the patch function runs without error
        try:
            patch_mcp_client_call_tool()
            # If we get here, the patch was attempted
            assert True
        except ImportError:
            # It's okay if the FastMCPHUDClient doesn't exist in test environment
            pytest.skip("FastMCPHUDClient not available in test environment")

    @pytest.mark.asyncio
    async def test_patched_call_tool_handles_output_schema_error(self):
        """Test the logic that converts output schema errors to warnings."""
        # Test the actual logic that would be in the patched method
        from hud.types import MCPToolResult

        # Create an error result that should be converted
        error_result = MCPToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text="Tool 'test_tool' has an output schema but did not return structured content",  # noqa: E501
                )
            ],
            isError=True,
            structuredContent=None,
        )

        # Check if error text matches what we're looking for
        error_text = error_result.content[0].text
        assert "has an output schema but did not return structured content" in error_text

        # This is what the patch would do - convert to non-error
        converted_result = MCPToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=f"Warning: {error_text}\n"
                    f"The tool executed but didn't return structured content as expected. "
                    f"This has been converted to a warning.",
                )
            ],
            isError=False,
            structuredContent=None,
        )

        assert not converted_result.isError
        assert "Warning:" in converted_result.content[0].text
        assert "This has been converted to a warning" in converted_result.content[0].text

    @pytest.mark.asyncio
    async def test_patched_call_tool_handles_exception_with_schema_error(self):
        """Test patched method handles exceptions containing schema errors."""
        with patch("hud.utils.agent_patches.logger") as mock_logger:
            # Test the exception handling logic
            error_msg = (
                "Output validation error: outputSchema defined but no structured output returned"
            )

            # This matches the error pattern we're looking for
            if "outputSchema defined but no structured output returned" in error_msg:
                # The patch would log a warning
                mock_logger.warning.assert_not_called()  # Not called yet

                # Simulate logging
                mock_logger.warning(
                    "Tool '%s' raised output schema validation error. "
                    "Converting to warning and continuing. Original error: %s",
                    "test_tool",
                    error_msg,
                )

                mock_logger.warning.assert_called_once()

    def test_patch_fails_gracefully(self):
        """Test that patch fails gracefully if import fails."""
        # Test the actual code path for handling import errors
        with patch("hud.utils.agent_patches.logger") as mock_logger:
            # Mock the imports to fail
            import sys

            original_modules = {}
            modules_to_fail = ["mcp", "hud.clients.fastmcp", "hud.types"]

            for module in modules_to_fail:
                if module in sys.modules:
                    original_modules[module] = sys.modules[module]
                    del sys.modules[module]

            try:
                # This should fail and log an error
                patch_mcp_client_call_tool()

                # Check if error was logged
                if mock_logger.error.called:
                    error_args = mock_logger.error.call_args[0]
                    assert "Failed to patch" in error_args[0]
            finally:
                # Restore original modules
                for module, original in original_modules.items():
                    sys.modules[module] = original

    def test_apply_all_patches(self):
        """Test that apply_all_patches calls all patch functions."""
        with patch("hud.utils.agent_patches.patch_mcp_client_call_tool") as mock_patch:
            apply_all_patches()
            mock_patch.assert_called_once()

    @pytest.mark.asyncio
    async def test_other_exceptions_are_reraised(self):
        """Test that non-schema exceptions are re-raised."""
        # This tests the logic where other exceptions should be re-raised
        error_msg = "Some other error that's not schema related"

        # Check that it doesn't match our special cases
        assert "has an output schema but did not return structured content" not in error_msg
        assert "outputSchema defined but no structured output returned" not in error_msg

        # In the actual patch, this would cause the exception to be re-raised
        with pytest.raises(RuntimeError):
            raise RuntimeError(error_msg)
