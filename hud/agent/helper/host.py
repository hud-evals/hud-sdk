"""Helper server to expose VLMAgent via HTTP endpoints for distributed RL training."""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional
import os
import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

from hud.agent.vlm import VLMAgent
from hud.utils.common import Observation
from hud.rl.types import Batch

logger = logging.getLogger(__name__)

# Global agent instance
agent: Optional[VLMAgent] = None


# Pydantic models for API
class ObservationRequest(BaseModel):
    text: Optional[str] = None
    screenshot: Optional[str] = None


class SampleRequest(BaseModel):
    observation: ObservationRequest
    verbose: bool = False


class BatchRequest(BaseModel):
    observations: list[ObservationRequest]
    texts: list[str]
    advantages: list[float]
    returns: list[float]
    old_log_probs: Optional[list[float]] = None
    metadata: Optional[dict[str, Any]] = None


class UpdateRequest(BaseModel):
    batch: BatchRequest


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize agent on startup."""
    global agent
    
    # Load configuration from environment or config file
    config = load_agent_config()
    
    logger.info("Initializing VLMAgent with config: %s", config)
    
    # Create agent
    agent = VLMAgent(**config)
    
    # Warm up the model by loading it
    logger.info("Warming up model...")
    agent._setup_model()
    logger.info("Model ready!")
    
    yield
    
    # Cleanup would go here if needed
    logger.info("Shutting down...")


app = FastAPI(
    title="VLM Agent Server",
    description="HTTP endpoints for VLMAgent sampling and updates",
    lifespan=lifespan
)


def load_agent_config() -> dict:
    """Load agent configuration from environment or file.
    
    Priority:
    1. CONFIG_PATH environment variable pointing to JSON file
    2. Individual environment variables
    3. Default configuration
    """
    # Check for config file
    config_path = os.environ.get("AGENT_CONFIG_PATH")
    if config_path and os.path.exists(config_path):
        logger.info("Loading config from %s", config_path)
        with open(config_path) as f:
            return json.load(f)
    
    # Build from environment variables
    config = {
        "model_name": os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"),
        "device_map": os.environ.get("DEVICE_MAP", "auto"),
        "load_in_8bit": os.environ.get("LOAD_IN_8BIT", "true").lower() == "true",
        "use_lora": os.environ.get("USE_LORA", "true").lower() == "true",
        "lora_rank": int(os.environ.get("LORA_RANK", "16")),
        "lora_alpha": int(os.environ.get("LORA_ALPHA", "32")),
        "lora_dropout": float(os.environ.get("LORA_DROPOUT", "0.1")),
        "learning_rate": float(os.environ.get("LEARNING_RATE", "1e-5")),
        "max_new_tokens": int(os.environ.get("MAX_NEW_TOKENS", "512")),
        "temperature": float(os.environ.get("TEMPERATURE", "0.7")),
    }
    
    # Optional system prompt
    if "SYSTEM_PROMPT" in os.environ:
        config["system_prompt"] = os.environ["SYSTEM_PROMPT"]
    
    # Optional LoRA target modules (comma-separated)
    if "LORA_TARGET_MODULES" in os.environ:
        config["lora_target_modules"] = os.environ["LORA_TARGET_MODULES"].split(",")
    
    return config


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "agent_loaded": agent is not None,
        "model_name": agent.model_name if agent else None
    }


@app.post("/sample")
async def sample(request: SampleRequest):
    """Sample an action from the agent.
    
    Returns ActionSample with text, log probabilities, and parsed actions.
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        # Convert request to Observation
        obs = Observation(
            text=request.observation.text,
            screenshot=request.observation.screenshot
        )
        
        # Sample from agent
        action_sample = await agent.sample(obs, verbose=request.verbose)
        
        # Serialize ActionSample
        # Convert CLA actions to dicts for transport
        action_dicts = []
        if action_sample.actions:
            for action in action_sample.actions:
                action_dicts.append(action.model_dump())
        
        return {
            "action_sample": {
                "text": action_sample.text,
                "log_probs": action_sample.log_probs,
                "tokens": action_sample.tokens,
                "total_log_prob": action_sample.total_log_prob,
                "actions": action_dicts,
                "raw_actions": action_sample.raw_actions,
                "done": action_sample.done,
                "metadata": action_sample.metadata
            }
        }
        
    except Exception as e:
        logger.exception("Error in /sample")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update")
async def update(request: UpdateRequest):
    """Perform gradient update on a batch.
    
    Returns training statistics.
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        # Convert request to Batch
        observations = []
        for obs_req in request.batch.observations:
            observations.append(Observation(
                text=obs_req.text,
                screenshot=obs_req.screenshot
            ))
        
        batch = Batch(
            observations=observations,
            texts=request.batch.texts,
            advantages=request.batch.advantages,
            returns=request.batch.returns,
            old_log_probs=request.batch.old_log_probs,
            metadata=request.batch.metadata
        )
        
        # Perform update
        stats = await agent.update(batch)
        
        return {"stats": stats}
        
    except Exception as e:
        logger.exception("Error in /update")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/save_checkpoint")
async def save_checkpoint(path: str = "checkpoint"):
    """Save model checkpoint (LoRA weights only if using LoRA)."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        if agent._model is None:
            raise HTTPException(status_code=400, detail="Model not loaded")
        
        # Save model checkpoint
        agent._model.save_pretrained(path)
        if agent._tokenizer:
            agent._tokenizer.save_pretrained(path)
        
        return {"status": "saved", "path": path}
        
    except Exception as e:
        logger.exception("Error saving checkpoint")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run server
    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "0.0.0.0")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    ) 