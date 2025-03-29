from typing import Any, Optional
from hud.settings import settings
from hud.environment import Environment
from hud.run import make_run
import datetime

from hud.server import make_request


async def list_environments(*, api_key: Optional[str]) -> list[str]:
    """
    List all available gyms.

    Returns:
        list[str]: List of gym IDs
    """
    if api_key is None:
        api_key = settings.api_key
        
    # API call to get gyms
    data = await make_request(
        method="GET", url=f"{settings.base_url}/gyms", api_key=api_key
    )
    return data["gyms"]

async def make(
    id_or_name: str,
    *, 
    config: Optional[dict[str, Any]] = None, 
    metadata: Optional[dict[str, Any]] = None, 
    run_id: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Environment:
    """
    Create a new Environment.

    Returns:
        Environment: A new Gym instance
    """
    if api_key is None:
        api_key = settings.api_key
    
    # need to load gym id, since id_or_name can be either
    # API call to get gym info
    data = await make_request(
        method="GET",
        url=f"{settings.base_url}/gyms/{id_or_name}",
        api_key=api_key,
    )
    
    id = data["id"]
    
    if run_id is None:
        run = await make_run(
            name=f"env-{id}-{datetime.datetime.now().isoformat()}",
            gym_id=id,
        )
        run_id = run.id

    env = Environment(
        run_id=run_id,
        id=id,
        config=config,
        metadata=metadata,
    )
    await env.create_environment()
    await env.wait_for_ready()
    return env