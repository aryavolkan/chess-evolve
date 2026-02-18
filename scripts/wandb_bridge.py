#!/usr/bin/env python3
"""Stream Chess Evolve metrics.json updates into Weights & Biases.

Run this while Chess Evolve training is active (UI or headless). The script polls
`metrics.json`, logs new generations to W&B, and prints a concise console summary.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.expanduser("~/Projects/shared-evolve-utils"))
import wandb  # noqa: E402
from godot_wandb import godot_user_dir, poll_metrics, read_metrics  # noqa: E402

CHESS_LOG_KEYS = [
    "generation", "white_best", "white_avg", "black_best", "black_avg",
    "best_fitness", "avg_fitness", "games_played",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Stream Chess Evolve metrics into W&B")
    parser.add_argument("--project", default="chess-evolve", help="W&B project name")
    parser.add_argument("--entity", default=None, help="W&B entity (team/user)")
    parser.add_argument("--run-name", default=None, help="Name for this run")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Seconds between polls")
    args = parser.parse_args()

    metrics_path = godot_user_dir("Chess Evolve") / "metrics.json"

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name or f"bridge-{int(time.time())}",
        tags=["chess-evolve", "bridge"],
    )

    print(f"W&B run: {run.url}")
    print(f"Watching: {metrics_path}")
    print("Press Ctrl+C to stop.\n")

    try:
        poll_metrics(
            run, metrics_path,
            max_generations=9999,
            poll_interval=args.poll_interval,
            max_stale=300,
            log_keys=CHESS_LOG_KEYS,
        )
    except KeyboardInterrupt:
        print("\nStopped by user")

    final = read_metrics(metrics_path)
    if final:
        wandb.summary["final_generation"] = final.get("generation", 0)
        wandb.summary["final_white_best"] = final.get("white_best", 0)
        wandb.summary["final_black_best"] = final.get("black_best", 0)

    wandb.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
