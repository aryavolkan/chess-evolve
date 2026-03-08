#!/bin/bash
# Start a W&B hyperparameter sweep for chess-evolve
# Creates sweep and launches worker agents

set -e

WORKERS=${1:-3}  # Default 3 workers
SWEEP_NAME=${2:-"chess-evolve-sweep-$(date +%Y%m%d-%H%M)"}

echo "🎮 Starting Chess-Evolve W&B Sweep"
echo "   Workers: $WORKERS"
echo "   Name: $SWEEP_NAME"
echo ""

# Create sweep
echo "📊 Creating W&B sweep..."
SWEEP_ID=$(python3 -c "
import wandb
from sweep_config import sweep_config

# Add sweep name
sweep_config['name'] = '$SWEEP_NAME'

sweep_id = wandb.sweep(
    sweep_config,
    project='chess-evolve'
)

print(sweep_id)
")

echo "✓ Sweep created: $SWEEP_ID"
echo "🔗 https://wandb.ai/aryavolkan-personal/chess-evolve/sweeps/$SWEEP_ID"
echo ""

# Launch workers
echo "🚀 Launching $WORKERS workers..."
for i in $(seq 1 $WORKERS); do
    nohup python3 train_wandb.py --sweep $SWEEP_ID > worker${i}.log 2>&1 &
    PID=$!
    echo "   Worker $i (PID $PID) → worker${i}.log"
    sleep 2
done

echo ""
echo "✅ Sweep running!"
echo ""
echo "Monitor:"
echo "  Web:  https://wandb.ai/aryavolkan-personal/chess-evolve/sweeps/$SWEEP_ID"
echo "  Logs: tail -f worker*.log"
echo "  Stop: pkill -f 'train_wandb.py --sweep $SWEEP_ID'"
echo ""
echo "Sweep ID: $SWEEP_ID"
