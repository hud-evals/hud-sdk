#!/usr/bin/env bash
# Launch an AWS Trainium instance (trn1.2xlarge) and configure it to run
# the Qwen VLM server automatically via cloud-init.
#
# Prerequisites:
#   • AWS CLI configured with credentials + default region
#   • An existing key-pair ($KEY_NAME) for SSH access
#   • A security-group ($SG_ID) that allows inbound TCP/8000 from your IP
#
# Usage:
#   SG_ID=sg-xxxx KEY_NAME=my-key ./launch_trainium_instance.sh
#
set -euo pipefail

# -------- adjustable parameters --------
AMI_ID="ami-0df24e148655a9132"   # Deep Learning AMI Neuron PyTorch 2.1 (Ubuntu 22.04)
INSTANCE_TYPE="trn1.2xlarge"
SUBNET_ID=${SUBNET_ID:-"subnet-xxxxxxxx"}
SG_ID=${SG_ID:?"provide SG_ID env var"}
KEY_NAME=${KEY_NAME:?"provide KEY_NAME env var"}
TAG="qwen-vlm-server"

# cloud-init user-data script installs deps & launches server on boot
read -r -d '' USER_DATA <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

# Activate neuron venv
source /opt/aws_neuron_venv_pytorch/bin/activate

# Clone / pull repo (assumes public Github; replace if private)
if [[ ! -d "$HOME/hud-sdk" ]]; then
  git clone https://github.com/hud-evals/hud-sdk.git $HOME/hud-sdk
else
  cd $HOME/hud-sdk && git pull --rebase
fi
cd $HOME/hud-sdk

pip install --upgrade pip
pip install --upgrade "optimum[neuronx]" transformers fastapi uvicorn httpx safetensors accelerate peft
pip install -e .

# Export runtime env (edit as needed)
export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
export NEURON_RT_NUM_CORES=2
export XLA_USE_BF16=1
export NEURON_COMPILE_CACHE_URL="$HOME/neuron_cache"
export PORT=8000

# Launch server in tmux so it keeps running
sudo apt-get update && sudo apt-get install -y tmux
su - ubuntu -c "tmux new-session -d -s vlm 'python scripts/trainium/serve_neuron.py'"
EOF

# Launch instance
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SG_ID" \
  --subnet-id "$SUBNET_ID" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$TAG}]" \
  --user-data "$USER_DATA" \
  --query 'Instances[0].InstanceId' --output text)

echo "Launched instance: $INSTANCE_ID (waiting for public IP)"

aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"
IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --query "Reservations[0].Instances[0].PublicIpAddress" --output text)

echo "Instance is ready: $IP"
echo "Health check: curl http://$IP:8000/health"
echo "SSH: ssh -i $KEY_NAME.pem ubuntu@$IP" 