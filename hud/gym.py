from typing import Any, Optional
from hud.environment import Environment
from hud.run import make_run
import datetime


async def make(id: str, *, metadata: Optional[dict[str, Any]] = None, run_id: Optional[str] = None) -> Environment:
    """
    Create a new Environment.

    Returns:
        Environment: A new Gym instance
    """
    
    if run_id is None:
        run = await make_run(
            name=f"env-{id}-{datetime.datetime.now().isoformat()}",
            gym_id=id,
            metadata=metadata,
            evalset_id="unknown",
        )
        run_id = run.id

    env = Environment(
        run_id=run_id,
        config=self.config,
        adapter=self.adapter,
        metadata=metadata or {},
    )
    await env.create_environment()

    return env