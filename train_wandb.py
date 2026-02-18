#!/usr/bin/env python3
"""
Chess-Evolve W&B-tracked training with hyperparameter sweeps.
Uses shared Godot+W&B utilities for the launch→poll→log pipeline.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.expanduser("~/shared-evolve-utils"))
from godot_wandb import run_training  # noqa: E402

import wandb  # noqa: E402

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Default config (overridden by W&B sweep)
DEFAULT_CONFIG = {
    "population_size": 30,
    "hidden_size": 64,
    "elite_count": 3,
    "crossover_rate": 0.70,
    "mutation_rate": 0.25,
    "mutation_strength": 0.12,
    "games_per_individual": 2,
    "max_generations": 100,
    "max_moves_per_game": 100,
    "input_size": 389,
    "output_size": 128,
}

PROJECT_PATH = os.path.expanduser("~/Projects/chess-evolve")

CHESS_LOG_KEYS = [
    "generation", "white_best", "white_avg", "black_best", "black_avg",
    "best_fitness", "avg_fitness", "games_played",
]


def do_training(config=None, visible=False):
    """Run a single training session."""
    merged = DEFAULT_CONFIG.copy()
    if config:
        merged.update(config)
    run_training(
        config=merged,
        project_path=PROJECT_PATH,
        app_name="chess-evolve",
        wandb_project="chess-evolve",
        wandb_tags=["chess", "neuroevolution", "coevolution"],
        visible=visible,
        log_keys=CHESS_LOG_KEYS,
    )


def sweep_agent(sweep_id: str):
    """W&B sweep agent."""
    def train_fn():
        config = dict(wandb.config)
        do_training(config, visible=False)

    wandb.agent(sweep_id, function=train_fn, count=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chess-Evolve W&B training")
    parser.add_argument("--sweep", type=str, help="W&B sweep ID to join")
    parser.add_argument("--visible", action="store_true", help="Show Godot window")
    parser.add_argument("--config", type=str, help="JSON config file")
    args = parser.parse_args()

    custom_config = None
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            custom_config = json.load(f)
        print(f"✓ Loaded config from {args.config}")

    if args.sweep:
        print(f"🔄 Joining W&B sweep: {args.sweep}")
        sweep_agent(args.sweep)
    else:
        do_training(custom_config, visible=args.visible)
