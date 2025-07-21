from __future__ import annotations

import asyncio
import inspect
from functools import wraps
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from mcp.server.fastmcp import FastMCP


def register_instance_tool(mcp: FastMCP, name: str, instance: Any) -> Callable[..., Any]:
    """Register ``instance.__call__`` as a FastMCP tool.

    Parameters
    ----------
    mcp:
        A :class:`mcp.server.fastmcp.FastMCP` instance.
    name:
        Public tool name.
    instance:
        Object with an ``async def __call__`` (or sync) implementing the tool.
    """

    call_fn = instance.__call__
    sig = inspect.signature(call_fn)

    # Remove *args/**kwargs so Pydantic doesn't treat them as required fields
    from typing import Any as _Any

    filtered = [
        p.replace(kind=p.POSITIONAL_OR_KEYWORD, annotation=_Any)
        for p in sig.parameters.values()
        if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]

    public_sig = inspect.Signature(parameters=filtered, return_annotation=_Any)

    @wraps(call_fn)
    async def _wrapper(*args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        result = call_fn(*args, **kwargs)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    _wrapper.__signature__ = public_sig  # type: ignore[attr-defined]

    return mcp.tool(name=name)(_wrapper)
