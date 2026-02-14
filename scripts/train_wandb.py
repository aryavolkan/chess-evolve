#!/usr/bin/env python3
"""
Single W&B-tracked chess-evolve training run with live logging.
Polls metrics.json for reliability (Godot buffers stdout).
"""
import wandb
import subprocess
import json
import os
import time
import sys
from pathlib import Path

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Default config for chess evolution
DEFAULT_CONFIG = {
    "population_size": 30,
    "input_size": 389,  # Board encoding size
    "hidden_size": 64,
    "output_size": 128,  # Move probability outputs
    "elite_count": 3,
    "games_per_individual": 2,
    "max_moves_per_game": 100,
    "max_generations": 100,
    "mutation_rate": 0.15,
    "mutation_strength": 0.3,
}

# Paths
GODOT_PATH = "/opt/homebrew/bin/godot"  # Homebrew install
PROJECT_PATH = os.path.expanduser("~/Projects/chess-evolve")
GODOT_USER_DIR = os.path.expanduser("~/Library/Application Support/Godot/app_userdata/Chess Evolve")
METRICS_PATH = os.path.join(GODOT_USER_DIR, "metrics.json")
CONFIG_PATH = os.path.join(GODOT_USER_DIR, "sweep_config.json")

# Polling settings
POLL_INTERVAL = 5  # seconds between checks
MAX_WAIT_FOR_START = 30  # seconds to wait for training start
COMPLETION_WAIT = 5  # seconds to wait after training completes


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
    """Launch Godot training process"""
    # Clear old metrics
    if os.path.exists(METRICS_PATH):
        os.remove(METRICS_PATH)
        print("✓ Cleared old metrics")

    # Build command
    cmd = [GODOT_PATH, "--path", PROJECT_PATH]
    
    if not visible:
        cmd.extend(["--headless", "--rendering-driver", "dummy"])
    
    cmd.extend(["--", "--auto-train"])
    
    print(f"🚀 Launching Godot: {' '.join(cmd)}")
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
    
    print("❌ Training did not start in time")
    return False


def poll_metrics(run, max_generations):
    """Poll metrics.json and log to W&B"""
    last_gen = -1
    stale_count = 0
    max_stale = 60  # 5 minutes of no progress
    
    while last_gen < max_generations - 1:
        metrics = read_metrics()
        
        if metrics and "generation" in metrics:
            gen = metrics["generation"]
            
            if gen > last_gen:
                # New generation!
                last_gen = gen
                stale_count = 0
                
                # Log to W&B
                log_data = {
                    "generation": gen,
                    "white_best": metrics.get("white_best", 0),
                    "white_avg": metrics.get("white_avg", 0),
                    "black_best": metrics.get("black_best", 0),
                    "black_avg": metrics.get("black_avg", 0),
                    "best_fitness": metrics.get("best_fitness", 0),
                    "avg_fitness": metrics.get("avg_fitness", 0),
                    "games_played": metrics.get("games_played", 0),
                }
                
                wandb.log(log_data)
                
                print(f"Gen {gen}: W_best={log_data['white_best']:.1f}, "
                      f"W_avg={log_data['white_avg']:.1f}, "
                      f"B_best={log_data['black_best']:.1f}, "
                      f"B_avg={log_data['black_avg']:.1f}")
            else:
                stale_count += 1
                if stale_count >= max_stale:
                    print("❌ Training appears stuck")
                    return False
        
        time.sleep(POLL_INTERVAL)
    
    print("✓ Training complete!")
    return True


def main():
    # Parse args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--visible", action="store_true", help="Show Godot window")
    parser.add_argument("--project", default="chess-evolve-neuroevolution", help="W&B project")
    parser.add_argument("--entity", default="aryavolkan-personal", help="W&B entity")
    args = parser.parse_args()
    
    # Merge config from W&B sweep (if running in one) or use defaults
    config = DEFAULT_CONFIG.copy()
    
    # Initialize W&B
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        config=config,
        tags=["chess-evolve", "coevolution"]
    )
    
    # Write config for Godot
    write_config(wandb.config.as_dict())
    
    # Launch Godot
    process = launch_godot(visible=args.visible)
    
    try:
        # Wait for training to start
        if not wait_for_training_start():
            print("❌ Training failed to start")
            run.finish(exit_code=1)
            return 1
        
        # Poll and log metrics
        success = poll_metrics(run, wandb.config.max_generations)
        
        # Wait a bit for final metrics
        time.sleep(COMPLETION_WAIT)
        
        # Final metrics
        final = read_metrics()
        if final:
            print("\n📊 Final Results:")
            print(f"  Generations: {final.get('generation', 0)}")
            print(f"  White Best: {final.get('white_best', 0):.1f}")
            print(f"  Black Best: {final.get('black_best', 0):.1f}")
            print(f"  Total Games: {final.get('games_played', 0)}")
        
        # Cleanup
        process.terminate()
        process.wait(timeout=5)
        
        run.finish(exit_code=0 if success else 1)
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        process.terminate()
        process.wait(timeout=5)
        run.finish(exit_code=130)
        return 130
    
    except Exception as e:
        print(f"❌ Error: {e}")
        process.terminate()
        process.wait(timeout=5)
        run.finish(exit_code=1)
        return 1


if __name__ == "__main__":
    sys.exit(main())
