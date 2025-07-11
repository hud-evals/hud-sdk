#!/usr/bin/env python3
"""Launch VLMAgent server on AWS Trainium."""

import os
import sys
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our Neuron setup
from trainium_qwen_setup import setup_neuron_environment, create_neuron_vlm_agent

# Setup Neuron environment
setup_neuron_environment()

# Create FastAPI app
app = FastAPI(title="Neuron VLM Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent instance
agent = None

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    global agent
    
    # Create Neuron-compatible agent
    NeuronVLMAgent = create_neuron_vlm_agent()
    
    agent = NeuronVLMAgent(
        model_name=os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"),
        use_lora=False,  # LoRA not supported on Neuron yet
        max_new_tokens=256,
        temperature=0.7,
        system_prompt="You are a helpful math tutor. Solve problems step by step."
    )
    
    print("Warming up model...")
    # Warm up with a dummy request
    from hud.utils.common import Observation
    obs = Observation(text="What is 2+2?")
    await agent.sample(obs)
    print("Model ready!")

@app.post("/sample")
async def sample(request: dict):
    """Handle sample requests."""
    from hud.utils.common import Observation
    
    # Create observation
    obs = Observation(**request.get("observation", {"text": ""}))
    
    # Generate sample
    action_sample = await agent.sample(obs)
    
    # Convert to dict
    return {
        "text": action_sample.text,
        "log_probs": action_sample.log_probs,
        "tokens": action_sample.tokens,
        "total_log_prob": action_sample.total_log_prob,
        "done": action_sample.done,
        "metadata": action_sample.metadata,
    }

@app.post("/update")
async def update(request: dict):
    """Handle update requests."""
    from hud.rl.types import Batch
    
    # Create batch from request
    batch = Batch(**request)
    
    # Perform update
    stats = await agent.update(batch)
    
    # Ensure all values are floats
    return {k: float(v) for k, v in stats.items()}

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "device": "neuron"}

if __name__ == "__main__":
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        log_level="info"
    )
