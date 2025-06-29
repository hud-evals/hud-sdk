from __future__ import annotations

import asyncio
import logging
import time
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from pathlib import Path

import httpx

from hud.settings import settings
from hud.utils.common import Observation

logger = logging.getLogger("hud.telemetry")

@dataclass
class RequestTracker:
    method: str
    url: str
    queued_at: float
    started_at: float | None = None
    completed_at: float | None = None
    status_code: int | None = None
    worker_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "url": self.url,
            "queued_at": datetime.fromtimestamp(self.queued_at).isoformat(),
            "started_at": datetime.fromtimestamp(self.started_at).isoformat() if self.started_at else None,
            "completed_at": datetime.fromtimestamp(self.completed_at).isoformat() if self.completed_at else None,
            "queue_time": self.started_at - self.queued_at if self.started_at else None,
            "execution_time": self.completed_at - self.started_at if self.completed_at and self.started_at else None,
            "total_time": self.completed_at - self.queued_at if self.completed_at else None,
            "status_code": self.status_code,
            "worker_id": self.worker_id
        }

class AsyncLogger:
    _instance = None
    _queue: asyncio.Queue = asyncio.Queue()
    _worker_tasks: list[asyncio.Task] = []
    _client: httpx.AsyncClient | None = None
    NUM_WORKERS = 5  # Number of concurrent workers
    
    # Tracking file setup
    TRACKING_FILE = Path("request_tracking.jsonl")

    def __init__(self):
        raise RuntimeError("Use AsyncLogger.get_instance()")

    @classmethod
    def get_instance(cls) -> AsyncLogger:
        if cls._instance is None:
            cls._instance = super(AsyncLogger, cls).__new__(cls)
            cls._instance._start_workers()
        return cls._instance

    def _start_workers(self) -> None:
        if not self._worker_tasks:
            # Create shared HTTP client with connection pooling
            limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
            self._client = httpx.AsyncClient(limits=limits, timeout=30.0)
            
            # Start multiple workers
            for i in range(self.NUM_WORKERS):
                task = asyncio.create_task(self._process_queue(i))
                self._worker_tasks.append(task)

    async def _process_queue(self, worker_id: int) -> None:
        while True:
            try:
                item = await self._queue.get()
                if item is None:  # Shutdown sentinel
                    self._queue.task_done()
                    break

                method, url, data, tracker = item
                tracker.started_at = time.time()
                tracker.worker_id = worker_id
                
                await self._send_to_server(method, url, data, tracker)
                self._queue.task_done()
            except Exception as e:
                logger.exception("Error processing queue item: %s", e)

    async def _send_to_server(self, method: str, url: str, data: dict[str, Any], tracker: RequestTracker) -> None:
        if not self._client:
            return

        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {settings.api_key}",
            }
            
            response = await self._client.post(
                url,
                json=data,
                headers=headers,
            )

            tracker.completed_at = time.time()
            tracker.status_code = response.status_code

            # Write tracking info to file
            with open(self.TRACKING_FILE, "a") as f:
                f.write(json.dumps(tracker.to_dict()) + "\n")

            if response.status_code >= 200 and response.status_code < 300:
                logger.debug(
                    "Successfully sent %s request to %s. Status: %s",
                    method,
                    url,
                    response.status_code,
                )
            else:
                logger.warning(
                    "Failed to send %s request to %s: HTTP %s - %s",
                    method,
                    url,
                    response.status_code,
                    response.text,
                )
        except Exception as e:
            logger.exception("Error sending %s request to %s: %s", method, url, e)

    async def log_observation(self, env_id: str, observation: Observation) -> None:
        """Queue an observation to be logged to the telemetry service."""
        if not settings.telemetry_enabled:
            return

        url = f"{settings.base_url}/v2/environments/{env_id}/log_observation"
        data = observation.to_json()
        tracker = RequestTracker(
            method="observation",
            url=url,
            queued_at=time.time()
        )
        await self._queue.put(("observation", url, data, tracker))

    async def log_score(self, env_id: str, score: float) -> None:
        """Queue a score to be logged to the telemetry service."""
        if not settings.telemetry_enabled:
            return

        url = f"{settings.base_url}/v2/environments/{env_id}/log_score"
        data = {"score": score}
        tracker = RequestTracker(
            method="score",
            url=url,
            queued_at=time.time()
        )
        await self._queue.put(("score", url, data, tracker))

    async def shutdown(self) -> None:
        """Shutdown the logger, waiting for queued items to complete."""
        # Send shutdown sentinel to all workers
        for _ in range(len(self._worker_tasks)):
            await self._queue.put(None)
        
        # Wait for all workers to complete
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks)
            self._worker_tasks.clear()
        
        # Close the HTTP client
        if self._client:
            await self._client.aclose()
            self._client = None 
