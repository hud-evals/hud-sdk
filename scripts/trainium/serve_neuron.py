"""FastAPI server that hosts VLMNeuronAgent on a Trainium instance.

Run with:
    MODEL_NAME=Qwen/Qwen2.5-7B-Instruct python scripts/trainium/serve_neuron.py

Environment variables honoured (see .env.example).
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

from fastapi import FastAPI
import uvicorn

# Ensure project root is on sys.path so `hud` is importable when executed
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from hud.utils.common import Observation  # noqa: E402  (after path tweak)
from hud.rl.types import Batch  # noqa: E402
from .vlm_neuron_agent import VLMNeuronAgent  # noqa: E402

app = FastAPI(title="Qwen-Trainium VLM Server")

agent: VLMNeuronAgent | None = None


@app.on_event("startup")
async def _startup():
    global agent

    agent = VLMNeuronAgent(
        model_name=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"),
        use_lora=False,
        device_map="xla",
        load_in_8bit=False,
    )

    # Compile / load model and run a tiny warm-up forward
    agent._setup_model()
    await agent.sample(Observation(text="Hello"))


@app.post("/sample")
async def sample(request: dict):
    """Generate a single ActionSample."""
    obs_dict = request.get("observation", {})
    obs = Observation(**obs_dict)
    result = await agent.sample(obs)  # type: ignore[arg-type]
    return {"action_sample": result.model_dump()}


@app.post("/update")
async def update(request: dict):
    """Apply an RL update batch on the server side."""
    batch_dict = request.get("batch", {})
    batch = Batch(**batch_dict)
    stats = await agent.update(batch)  # type: ignore[arg-type]
    # Cast to float for JSON serialisation
    return {"stats": {k: float(v) for k, v in stats.items()}}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        log_level="info",
    ) 