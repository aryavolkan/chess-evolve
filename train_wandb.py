#!/usr/bin/env python3
"""
Chess-Evolve W&B-tracked training with hyperparameter sweeps.
Polls Godot metrics and logs to Weights & Biases.
"""
import wandb
import subprocess
import json
import os
import time
import sys
import argparse
from pathlib import Path

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Default config (will be overridden by W&B sweep)
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
    "input_size": 389,   # Board encoding size (fixed)
    "output_size": 128   # Move selection output (fixed)
}

# Paths
GODOT_PATH = "/opt/homebrew/bin/godot"
PROJECT_PATH = os.path.expanduser("~/Projects/chess-evolve")
GODOT_USER_DIR = os.path.expanduser("~/Library/Application Support/Godot/app_userdata/chess-evolve")
METRICS_PATH = os.path.join(GODOT_USER_DIR, "metrics.json")
CONFIG_PATH = os.path.join(GODOT_USER_DIR, "sweep_config.json")

# Polling settings
POLL_INTERVAL = 5  # seconds between metrics checks
MAX_WAIT_FOR_START = 30  # seconds to wait for training to start


def write_config(config):
    """Write config for Godot to read"""
    os.makedirs(GODOT_USER_DIR, exist_ok=True)
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Config written to {CONFIG_PATH}")


def read_metrics():
    """Read current metrics from Godot's JSON file"""
    try:
        if not os.path.exists(METRICS_PATH):
            return None
        with open(METRICS_PATH, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def launch_godot(visible=False):
    """Launch Godot chess training process"""
    # Clear old metrics
    if os.path.exists(METRICS_PATH):
        os.remove(METRICS_PATH)
        print("✓ Cleared old metrics")

    # Build command
    cmd = [GODOT_PATH, "--path", PROJECT_PATH]
    
    if not visible:
        cmd.extend(["--headless", "--rendering-driver", "dummy"])
    
    cmd.extend(["--", "--auto-train"])
    
    print(f"Launching: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    return process


def wait_for_training_start(timeout=MAX_WAIT_FOR_START):
    """Wait for metrics.json to appear (training started)"""
    print(f"⏳ Waiting for training to start (timeout: {timeout}s)...")
    start = time.time()
    
    while time.time() - start < timeout:
        if os.path.exists(METRICS_PATH):
            print("✓ Training started!")
            return True
        time.sleep(1)
    
    print("✗ Timeout waiting for training start")
    return False


def poll_and_log_metrics(wandb_run, max_gens=100):
    """Poll metrics.json and log to W&B"""
    last_gen = -1
    print(f"📊 Polling metrics every {POLL_INTERVAL}s...")
    
    while True:
        metrics = read_metrics()
        
        if metrics is None:
            time.sleep(POLL_INTERVAL)
            continue
        
        gen = metrics.get("generation", -1)
        
        # New generation data available
        if gen > last_gen:
            print(f"\nGen {gen}: W_best={metrics.get('white_best', 0):.1f}, "
                  f"W_avg={metrics.get('white_avg', 0):.1f}, "
                  f"B_best={metrics.get('black_best', 0):.1f}, "
                  f"B_avg={metrics.get('black_avg', 0):.1f}")
            
            # Log to W&B
            wandb_run.log({
                "generation": gen,
                "white_best": metrics.get("white_best", 0),
                "white_avg": metrics.get("white_avg", 0),
                "black_best": metrics.get("black_best", 0),
                "black_avg": metrics.get("black_avg", 0),
                "games_played": metrics.get("games_played", 0)
            })
            
            last_gen = gen
        
        # Check if training complete
        if gen >= max_gens - 1:
            print(f"\n✓ Training complete! Reached generation {gen}")
            break
        
        time.sleep(POLL_INTERVAL)
    
    return metrics


def run_training(config=None, visible=False):
    """Run a single training session with W&B logging"""
    # Merge config with defaults
    if config is None:
        config = DEFAULT_CONFIG.copy()
    else:
        merged = DEFAULT_CONFIG.copy()
        merged.update(config)
        config = merged
    
    # Initialize W&B
    run = wandb.init(
        project="chess-evolve",
        config=config,
        tags=["chess", "neuroevolution", "coevolution"]
    )
    
    print(f"\n🎮 Starting Chess-Evolve training:")
    print(f"   Pop: {config['population_size']}, Hidden: {config['hidden_size']}, Elite: {config['elite_count']}")
    print(f"   Mutation: rate={config['mutation_rate']:.3f}, strength={config['mutation_strength']:.3f}")
    print(f"   Crossover: {config['crossover_rate']:.3f}")
    print(f"   Games per individual: {config['games_per_individual']}")
    print(f"   Max generations: {config['max_generations']}\n")
    
    # Write config for Godot
    write_config(config)
    
    # Launch training
    process = launch_godot(visible=visible)
    
    # Wait for training to start
    if not wait_for_training_start():
        process.kill()
        run.finish(exit_code=1)
        return
    
    # Poll and log metrics
    try:
        final_metrics = poll_and_log_metrics(run, config["max_generations"])
        
        # Wait for process to finish
        print("\n⏳ Waiting for Godot to finish...")
        time.sleep(5)
        process.terminate()
        process.wait(timeout=10)
        
        # Log final summary
        if final_metrics:
            run.summary["final_white_best"] = final_metrics.get("white_best", 0)
            run.summary["final_black_best"] = final_metrics.get("black_best", 0)
            run.summary["total_games"] = final_metrics.get("games_played", 0)
        
        print("\n✅ Training complete!")
        run.finish(exit_code=0)
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        process.kill()
        run.finish(exit_code=130)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        process.kill()
        run.finish(exit_code=1)


def sweep_agent():
    """W&B sweep agent - run training with sweep-provided config"""
    def train_fn():
        # W&B sweep provides config via wandb.config
        config = dict(wandb.config)
        run_training(config, visible=False)
    
    wandb.agent(sweep_id, function=train_fn, count=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chess-Evolve W&B training")
    parser.add_argument("--sweep", type=str, help="W&B sweep ID to join")
    parser.add_argument("--visible", action="store_true", help="Show Godot window (not headless)")
    parser.add_argument("--config", type=str, help="JSON config file (overrides defaults)")
    
    args = parser.parse_args()
    
    # Load custom config if provided
    custom_config = None
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
        print(f"✓ Loaded config from {args.config}")
    
    if args.sweep:
        # Join existing sweep
        sweep_id = args.sweep
        print(f"🔄 Joining W&B sweep: {sweep_id}")
        sweep_agent()
    else:
        # Single run with config
        run_training(custom_config, visible=args.visible)
