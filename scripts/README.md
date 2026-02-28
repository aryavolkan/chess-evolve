# Scripts

Helper scripts for training and W&B integration.

For full documentation see:
- [Training Guide](../docs/TRAINING.md) — running training, config reference, metrics, sweeps
- [Improving Training](../docs/IMPROVING_TRAINING.md) — hyperparameter tuning, diagnosing issues
- [Architecture](../docs/ARCHITECTURE.md) — system design and data flow

## Quick Reference

```bash
# Headless test (3 generations)
bash scripts/test_headless.sh

# Single W&B run
python train_wandb.py

# W&B sweep
python train_wandb.py --sweep <sweep-id>

# With custom config
python train_wandb.py --config my_config.json
```

## Metrics Path

Godot writes metrics to the user data directory:
- **Linux**: `~/.local/share/godot/app_userdata/Chess Evolve/metrics.json`
- **macOS**: `~/Library/Application Support/Godot/app_userdata/Chess Evolve/metrics.json`
