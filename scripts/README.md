# Chess-Evolve Training Scripts

Scripts for headless training and W&B integration.

## Quick Start

### Test Headless Mode

```bash
bash scripts/test_headless.sh
```

Runs a quick 3-generation test to verify headless training works.

### Single W&B Run

```bash
# Install wandb if needed
pip3 install wandb

# Login (first time only)
wandb login

# Run training
python3 scripts/train_wandb.py
```

Options:
- `--visible`: Show Godot window (default: headless)
- `--project <name>`: W&B project name (default: chess-evolve-neuroevolution)
- `--entity <name>`: W&B entity (default: aryavolkan-personal)

### W&B Sweep

```bash
# Create sweep (returns sweep ID)
wandb sweep scripts/sweep_config.yaml

# Run agents (in separate terminals)
wandb agent <entity>/<project>/<sweep-id>
```

## Config

Training config is written to:
```
~/Library/Application Support/Godot/app_userdata/Chess Evolve/sweep_config.json
```

Default config in `train_wandb.py`:
- population_size: 30
- hidden_size: 64
- elite_count: 3
- games_per_individual: 2
- max_moves_per_game: 100
- max_generations: 100
- mutation_rate: 0.15
- mutation_strength: 0.3

## Metrics

Godot writes metrics to:
```
~/Library/Application Support/Godot/app_userdata/Chess Evolve/metrics.json
```

Updated each generation with:
- generation
- white_best, white_avg
- black_best, black_avg
- best_fitness, avg_fitness
- games_played

## How It Works

1. Python script writes config to `user://sweep_config.json`
2. Launches Godot with `--auto-train` flag
3. Godot detects flag and runs headless training loop
4. Each generation, Godot writes metrics.json
5. Python polls metrics.json and logs to W&B

## Troubleshooting

**Training doesn't start:**
- Check Godot path in script (`/opt/homebrew/bin/godot`)
- Verify project path is correct
- Look for errors in Godot output

**No metrics file:**
- Training may have crashed early
- Check Godot console output
- Try running with `--visible` to see errors

**W&B not logging:**
- Check `wandb login` succeeded
- Verify project/entity names
- Check network connection
