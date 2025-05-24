from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from hud.server import make_request
from hud.settings import settings

if TYPE_CHECKING:
    from hud.env.environment import Environment  # Import Environment for type hinting

logger = logging.getLogger(__name__)


async def upload_env_telemetry(
    environment: Environment,
    results: Any,
    api_key: str | None = None,
) -> None:
    """
    Sends telemetry data (results from a cloud runner) to the HUD telemetry upload endpoint.
    """
    environment_id = environment.client.env_id  # type: ignore

    if not api_key:
        api_key = settings.api_key

    if not api_key:
        raise ValueError("API key must be provided either as an argument or set in hud.settings.")

    endpoint_url = f"{settings.base_url}/v2/environments/{environment_id}/telemetry-upload"

    request_payload = {
        "results": {
            "steps": results,
        }
    }

    logger.debug("Sending telemetry to %s for env_id: %s", endpoint_url, environment_id)

    try:
        await make_request(
            method="POST",
            url=endpoint_url,
            json=request_payload,
            api_key=api_key,
        )
        logger.info("Successfully uploaded telemetry for environment_id: %s", environment_id)
    except Exception as e:
        logger.error(
            "Failed to upload telemetry for environment_id: %s. Error: %s", environment_id, e
        )
        raise
