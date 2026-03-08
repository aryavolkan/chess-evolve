#!/usr/bin/env python3
"""Stream Chess Evolve metrics.json updates into Weights & Biases.

Run this while Chess Evolve training is active (UI or headless). The script polls
`metrics.json`, logs new generations to W&B, and prints a concise console summary.
"""
import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

shared_utils = Path.home() / "Projects/shared-evolve-utils"
if shared_utils.exists():
    sys.path.insert(0, str(shared_utils))

import wandb  # noqa: E402

from godot_wandb import godot_user_dir, poll_metrics, read_metrics  # noqa: E402

CHESS_LOG_KEYS = [
    "generation",
    # Fitness
    "white_best",
    "white_avg",
    "black_best",
    "black_avg",
    "combined_best",
    # Games
    "total_games_this_gen",
    "avg_game_length",
    "games_per_sec",
    "moves_per_sec",
    "generation_time_sec",
    # Outcome rates
    "white_win_rate",
    "white_draw_rate",
    "black_win_rate",
    "black_draw_rate",
    # Tournament scores
    "white_tournament_score_best",
    "white_tournament_score_avg",
    "black_tournament_score_best",
    "black_tournament_score_avg",
    # Material
    "white_material_avg",
    "black_material_avg",
    # Hall of Fame
    "white_hof_size",
    "black_hof_size",
    # NEAT topology metrics (per-color)
    "white_species_count",
    "black_species_count",
    "white_depth_avg",
    "black_depth_avg",
    "white_width_avg",
    "black_width_avg",
    "white_connections_avg",
    "black_connections_avg",
    "white_hidden_nodes_avg",
    "black_hidden_nodes_avg",
    # Fitness component breakdowns
    "white_fitness_outcome",
    "white_fitness_material",
    "white_fitness_mobility",
    "white_fitness_king_safety",
    "white_fitness_opp_king_safety",
    "white_fitness_king_danger",
    "white_fitness_captures",
    "white_fitness_move_penalty",
    "black_fitness_outcome",
    "black_fitness_material",
    "black_fitness_mobility",
    "black_fitness_king_safety",
    "black_fitness_opp_king_safety",
    "black_fitness_king_danger",
    "black_fitness_captures",
    "black_fitness_move_penalty",
    # King danger metrics
    "white_king_danger_avg",
    "black_king_danger_avg",
    # Benchmark vs random (absolute progress)
    "bench_white_win_rate",
    "bench_white_material_adv",
    "bench_black_win_rate",
    "bench_black_material_adv",
    "bench_avg_win_rate",
    # Stockfish benchmark
    "sf_white_win_rate",
    "sf_black_win_rate",
    "sf_avg_game_length",
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
