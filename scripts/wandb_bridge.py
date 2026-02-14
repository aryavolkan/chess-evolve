#!/usr/bin/env python3
"""Simple bridge that streams metrics.json updates into Weights & Biases.

Run this while Chess Evolve training is active (UI or headless). The script polls
`metrics.json`, logs new generations to W&B, and prints a concise console summary.
"""
import argparse
import json
import os
import time
from pathlib import Path

import wandb

DEFAULT_PROJECT = "chess-evolve"
DEFAULT_ENTITY = None  # falls back to logged-in user
DEFAULT_RUN_NAME = "local-training"
DEFAULT_POLL_INTERVAL = 2.0

# Godot stores user data per-project: ~/Library/Application Support/Godot/app_userdata/<ProjectName>
DEFAULT_USER_DIR = Path.home() / "Library" / "Application Support" / "Godot" / "app_userdata" / "Chess Evolve"
DEFAULT_METRICS_PATH = DEFAULT_USER_DIR / "metrics.json"


def read_metrics(path: Path) -> dict | None:
    try:
        with path.open() as fp:
            return json.load(fp)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Stream Chess Evolve metrics into W&B")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="W&B project name")
    parser.add_argument("--entity", default=DEFAULT_ENTITY, help="W&B entity (team/user)")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME, help="Name for this run")
    parser.add_argument("--metrics-path", default=str(DEFAULT_METRICS_PATH), help="Path to metrics.json")
    parser.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL, help="Seconds between polls")
    args = parser.parse_args()

    metrics_path = Path(args.metrics_path).expanduser()
    if not metrics_path.exists():
        print(f"Waiting for metrics at {metrics_path} ...")

    run = wandb.init(project=args.project, entity=args.entity, name=args.run_name, config={
        "metrics_path": str(metrics_path)
    })
    print(f"Streaming metrics from {metrics_path}\nRun URL: {run.url}")

    last_generation = -1
    try:
        while True:
            metrics = read_metrics(metrics_path)
            if metrics:
                generation = metrics.get("generation", -1)
                if generation > last_generation:
                    last_generation = generation
                    wandb.log(metrics)
                    print(
                        f"Gen {generation:4d} | "
                        f"W Best {metrics.get('white_best', 0):7.2f} | "
                        f"B Best {metrics.get('black_best', 0):7.2f} | "
                        f"Avg {metrics.get('avg_fitness', 0):7.2f} | "
                        f"Games {metrics.get('games_played', 0)}"
                    )
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("\nStopping bridge (Ctrl+C)")
    finally:
        run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
