#!/bin/bash
# Setup script for Trainium instance

set -e

echo "🚀 Setting up Trainium instance for Qwen model"

# Activate Neuron environment
source /opt/aws_neuron_venv_pytorch/bin/activate

# Verify Neuron
echo "✓ Checking Neuron installation..."
python -c "import torch_neuronx; print('Neuron SDK ready!')"
neuron-ls

# Install additional dependencies
echo "✓ Installing dependencies..."
pip install transformers accelerate peft fastapi uvicorn httpx

# Clone HUD SDK if not present
if [ ! -d "hud-sdk" ]; then
    echo "✓ Cloning HUD SDK..."
    git clone https://github.com/hud-ai/hud-sdk.git
    cd hud-sdk
    pip install -e .
    cd ..
fi

# Create cache directory
echo "✓ Setting up Neuron cache..."
export NEURON_COMPILE_CACHE_URL=/home/ubuntu/neuron_cache
mkdir -p $NEURON_COMPILE_CACHE_URL

echo "✅ Setup complete! Ready to launch server."
echo ""
echo "To start the server:"
echo "  export MODEL_NAME='Qwen/Qwen2.5-0.5B-Instruct'"
echo "  python launch_neuron_server.py"
