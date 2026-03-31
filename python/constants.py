"""Shared constants for chess-evolve training infrastructure.

Centralizes CHESS_LOG_KEYS and shared-evolve-utils path discovery to avoid
duplication across train_wandb.py, overnight-agent, and scripts.
"""

import os
import sys


def find_shared_evolve_utils() -> str:
    """Find the shared-evolve-utils directory, probing standard locations."""
    return next(
        (p for p in [
            os.path.expanduser("~/projects/shared-evolve-utils"),
            os.path.expanduser("~/Projects/shared-evolve-utils"),
            os.path.expanduser("~/shared-evolve-utils"),
        ] if os.path.isdir(p)),
        "",
    )


def add_shared_evolve_utils_to_path() -> str:
    """Find shared-evolve-utils and add to sys.path. Returns the path found."""
    path = find_shared_evolve_utils()
    if path and path not in sys.path:
        sys.path.insert(0, path)
    return path


# Canonical list of all metric keys logged to W&B.
# Shared between train_wandb.py, overnight-agent, and scripts/wandb_bridge.py.
CHESS_LOG_KEYS = [
    "generation",
    # Fitness
    "white_best",
    "white_avg",
    "black_best",
    "black_avg",
    "combined_best",
    "best_fitness",
    "avg_fitness",
    # Games
    "total_games_this_gen",
    "games_played",
    "avg_game_length",
    "games_per_sec",
    "moves_per_sec",
    "generation_time_sec",
    # Outcome rates
    "white_win_rate",
    "white_draw_rate",
    "black_win_rate",
    "black_draw_rate",
    "white_loss_rate",
    "black_loss_rate",
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
    "sf_white_avg_cpl",
    "sf_black_avg_cpl",
    # Stockfish fitness signal
    "sf_fitness_white_avg",
    "sf_fitness_black_avg",
    # Curriculum
    "curriculum_stage",
    # Puzzle metrics (stage 0)
    "puzzle_white_accuracy_best",
    "puzzle_white_accuracy_avg",
    "puzzle_black_accuracy_best",
    "puzzle_black_accuracy_avg",
    "puzzle_accuracy_best",
    "puzzle_white_soft_score_best",
    "puzzle_white_soft_score_avg",
    "puzzle_black_soft_score_best",
    "puzzle_black_soft_score_avg",
    "puzzle_soft_score_best",
    "puzzle_max_rating",
    # Puzzle benchmark (global, comparable across runs)
    "puzzle_bench_white_accuracy",
    "puzzle_bench_black_accuracy",
    "puzzle_bench_accuracy",
    "elo_estimate",
]
