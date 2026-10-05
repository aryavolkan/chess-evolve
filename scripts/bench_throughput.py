#!/usr/bin/env python3
"""Measure the Rust CPU backend's raw game-simulation throughput.

Plays batches of games between random fixed-topology genomes through
``chess_cpu.simulate_games_batch`` -- the same call the trainer makes every
generation -- and reports games/s and moves/s. Nothing is learned here; this
is a backend/hardware benchmark for comparing machines, thread counts and
builds, so the numbers in the README can be reproduced.

Usage:
    python scripts/bench_throughput.py                  # 150 games x 3 rounds
    python scripts/bench_throughput.py --games 600 --rounds 5
    RAYON_NUM_THREADS=1 python scripts/bench_throughput.py   # single-core baseline
    python scripts/bench_throughput.py --json           # machine-readable output

Requires the chess-cpu crate: ``cd rust/chess-cpu && maturin develop --release``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time

import numpy as np

try:
    import chess_cpu
except ImportError:  # pragma: no cover - exercised only without the crate
    sys.exit("chess_cpu is not installed. Build it first: cd rust/chess-cpu && maturin develop --release")

INPUT_SIZE = 389   # 6 piece planes x 64 squares + side to move + 4 castling rights
OUTPUT_SIZE = 4096  # one logit per (from_square, to_square) pair


def genome_size(input_size: int, hidden_size: int, output_size: int) -> int:
    """Flat weight count for input -> hidden -> output with biases (matches CPUTrainer)."""
    return input_size * hidden_size + hidden_size + hidden_size * output_size + output_size


def random_population(rng: np.random.Generator, n: int, size: int, input_size: int) -> np.ndarray:
    """He-scaled random genomes, same init the trainer uses for its benchmark population."""
    scale = np.float32((2.0 / input_size) ** 0.5)
    return (rng.standard_normal((n, size)).astype(np.float32) * scale).astype(np.float32)


def run_batch(white: np.ndarray, black: np.ndarray, pairings: list[tuple[int, int]], args: argparse.Namespace) -> tuple[float, int]:
    """Simulate one batch; return (seconds, total moves)."""
    gsize = white.shape[1]
    t0 = time.perf_counter()
    results = chess_cpu.simulate_games_batch(
        white.tobytes(),
        black.tobytes(),
        gsize,
        pairings,
        input_size=INPUT_SIZE,
        hidden_size=args.hidden,
        output_size=OUTPUT_SIZE,
        max_moves=args.max_moves,
        temperature=args.temperature,
    )
    dt = time.perf_counter() - t0
    return dt, sum(r["move_count"] for r in results)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int, default=150, help="games per round (default: 150 = pop 30 x 5 opponents)")
    parser.add_argument("--rounds", type=int, default=3, help="timed rounds after one warm-up (default: 3)")
    parser.add_argument("--max-moves", type=int, default=100, help="move cap per game (default: 100)")
    parser.add_argument("--hidden", type=int, default=64, help="hidden layer width (default: 64)")
    parser.add_argument("--temperature", type=float, default=0.5, help="move-selection softmax temperature (default: 0.5)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the random genomes (default: 0)")
    parser.add_argument("--json", action="store_true", help="print one JSON object instead of a table")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    gsize = genome_size(INPUT_SIZE, args.hidden, OUTPUT_SIZE)
    n = max(1, math.ceil(math.sqrt(args.games)))
    white = random_population(rng, n, gsize, INPUT_SIZE)
    black = random_population(rng, n, gsize, INPUT_SIZE)
    pairings = [(w, b) for w in range(n) for b in range(n)][: args.games]

    threads = os.environ.get("RAYON_NUM_THREADS") or str(os.cpu_count())

    run_batch(white, black, pairings[: min(len(pairings), 16)], args)  # warm-up (thread pool, page-in)

    rounds = [run_batch(white, black, pairings, args) for _ in range(args.rounds)]
    games_per_s = [len(pairings) / dt for dt, _ in rounds]
    moves_per_s = [moves / dt for dt, moves in rounds]
    summary = {
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "rayon_threads": threads,
        "genome_size": gsize,
        "games_per_round": len(pairings),
        "max_moves": args.max_moves,
        "hidden_size": args.hidden,
        "rounds": [
            {"seconds": round(dt, 4), "moves": moves, "games_per_s": round(g, 1), "moves_per_s": round(m, 1)}
            for (dt, moves), g, m in zip(rounds, games_per_s, moves_per_s, strict=True)
        ],
        "median_games_per_s": round(statistics.median(games_per_s), 1),
        "median_moves_per_s": round(statistics.median(moves_per_s), 1),
        "avg_game_length": round(sum(m for _, m in rounds) / (len(pairings) * len(rounds)), 1),
    }

    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    print(f"chess_cpu throughput  |  {platform.machine()}, {os.cpu_count()} CPUs, RAYON_NUM_THREADS={threads}")
    print(f"genome: {INPUT_SIZE}->{args.hidden}->{OUTPUT_SIZE} ({gsize:,} weights)  games/round: {len(pairings)}  max_moves: {args.max_moves}")
    print()
    print(f"{'round':>5}  {'seconds':>8}  {'moves':>8}  {'games/s':>9}  {'moves/s':>10}")
    for i, r in enumerate(summary["rounds"], 1):
        print(f"{i:>5}  {r['seconds']:>8.3f}  {r['moves']:>8}  {r['games_per_s']:>9.1f}  {r['moves_per_s']:>10.1f}")
    print()
    print(f"median: {summary['median_games_per_s']:.1f} games/s, {summary['median_moves_per_s']:.1f} moves/s, avg game length {summary['avg_game_length']} plies")
    return 0


if __name__ == "__main__":
    sys.exit(main())
