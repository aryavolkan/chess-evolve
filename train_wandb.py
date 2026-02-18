#!/usr/bin/env python3
"""
Chess-Evolve W&B-tracked training with hyperparameter sweeps.
Uses shared Godot+W&B utilities for the launch→poll→log pipeline.
"""
import argparse
import json
import os
import sys

# Set W&B environment variables
os.environ['WANDB_ENTITY'] = 'aryavolkan-personal'
os.environ['WANDB_PROJECT'] = 'chess-evolve'

sys.path.insert(0, os.path.expanduser("~/projects/shared-evolve-utils"))
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
    "games_per_individual": 2,  # Legacy parameter, replaced by tournament_opponents
    "tournament_opponents": 4,   # Number of opponents each individual plays
    "use_tournament": True,      # Enable tournament-based selection
    "tournament_mode": "round_robin",  # Tournament type: 'round_robin' or 'swiss'
    "max_generations": 100,
    "max_moves_per_game": 100,
    "input_size": 389,
    "output_size": 128,
}

PROJECT_PATH = os.path.expanduser("~/projects/chess-evolve")
GODOT_PATH = os.environ.get("GODOT_PATH", "/usr/local/bin/godot")

CHESS_LOG_KEYS = [
    "generation", "white_best", "white_avg", "black_best", "black_avg",
    "best_fitness", "avg_fitness", "games_played",
    # New metrics
    "white_win_rate", "white_draw_rate", "white_loss_rate",
    "black_win_rate", "black_draw_rate", "black_loss_rate",
    "white_hof_size", "black_hof_size",
    "white_tournament_score_best", "white_tournament_score_avg",
    "black_tournament_score_best", "black_tournament_score_avg",
    "total_games_this_gen", "avg_game_length",
    "white_material_avg", "black_material_avg",
    "generation_time_sec",
]


def do_training(config=None, visible=False):
    """Run a single training session."""
    merged = DEFAULT_CONFIG.copy()
    if config:
        merged.update(config)
    run_training(
        config=merged,
        project_path=PROJECT_PATH,
        app_name="Chess Evolve",
        wandb_project="chess-evolve",
        wandb_tags=["chess", "neuroevolution", "coevolution"],
        visible=visible,
        godot_path=GODOT_PATH,
        log_keys=CHESS_LOG_KEYS,
    )


def sweep_agent(sweep_id: str):
    """W&B sweep agent."""
    def train_fn():
        # When called by wandb.agent, wandb has already been initialized
        # with the sweep configuration. We need to extract the config.
        import wandb
        
        # Initialize a new run with the sweep config
        run = wandb.init(project="chess-evolve", entity="aryavolkan-personal")
        config = dict(run.config)
        
        # Merge with default config
        merged = DEFAULT_CONFIG.copy()
        merged.update(config)
        
        # Run the actual training
        print(f"Starting training with config: {merged}")
        
        # Call run_training directly (it will NOT call wandb.init again)
        from godot_wandb import godot_user_dir, read_metrics, poll_metrics, launch_godot, write_config
        
        user_dir = godot_user_dir("Chess Evolve")
        metrics_path = user_dir / "metrics.json"
        config_path = user_dir / "sweep_config.json"
        
        max_gens = merged.get("max_generations", 100)
        print(f"\n🎮 Starting training (pop={merged.get('population_size', '?')}, "
              f"gens={max_gens})\n")
        
        write_config(merged, config_path)
        
        godot_proc = launch_godot(
            project_path=PROJECT_PATH,
            godot_path=GODOT_PATH,
            visible=False,
            metrics_path=metrics_path,
        )
        
        try:
            poll_metrics(
                run, metrics_path,
                max_generations=max_gens,
                poll_interval=2.0,
                max_stale=300,
                log_keys=CHESS_LOG_KEYS,
            )
        finally:
            if godot_proc:
                godot_proc.terminate()
                godot_proc.wait(timeout=10)

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
        # Add entity and project to sweep ID if not already present
        sweep_id = args.sweep
        if "/" not in sweep_id:
            sweep_id = f"aryavolkan-personal/chess-evolve/{sweep_id}"
        print(f"🔄 Joining W&B sweep: {sweep_id}")
        sweep_agent(sweep_id)
    else:
        do_training(custom_config, visible=args.visible)
