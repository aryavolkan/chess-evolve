#!/usr/bin/env python3
"""W&B sweep worker for Chess-Evolve (headless Godot training)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_SHARED = next(
    p for p in [
        os.path.expanduser("~/projects/shared-evolve-utils"),
        os.path.expanduser("~/Projects/shared-evolve-utils"),
    ]
    if os.path.isdir(p)
)
sys.path.insert(0, _SHARED)
import wandb  # noqa: E402
from godot_wandb import (  # noqa: E402
    SweepWorker,
    define_step_metric,
    godot_user_dir,
    launch_godot,
    log_final_summary,
    read_metrics,
    run_sweep_agent,
    wait_for_metrics,
)
from global_elite import GlobalElitePool  # noqa: E402

PROJECT_PATH = os.environ.get("CHESS_EVOLVE_PROJECT_PATH", os.path.expanduser("~/Projects/chess-evolve"))
APP_NAME = os.environ.get("CHESS_EVOLVE_APP_NAME", "Chess Evolve")
GODOT_PATH = os.environ.get("GODOT_PATH", "/opt/homebrew/bin/godot")

DEFAULT_CONFIG = {
    "population_size": 20,
    "hidden_size": 64,
    "elite_count": 3,
    "crossover_rate": 0.70,
    "mutation_rate": 0.25,
    "mutation_strength": 0.12,
    "games_per_individual": 2,
    "max_generations": 50,
    "max_moves_per_game": 40,
    "input_size": 389,
    "output_size": 128,
    "use_minimax": False,
    "use_tournament": True,
    "tournament_opponents": 2,
}

CHESS_LOG_KEYS = [
    "generation",
    "white_best",
    "white_avg",
    "black_best",
    "black_avg",
    "best_fitness",
    "avg_fitness",
    "games_played",
    "total_games_this_gen",
    "avg_game_length",
    "games_per_sec",
    "moves_per_sec",
    "white_win_rate",
    "white_draw_rate",
    "white_loss_rate",
    "black_win_rate",
    "black_draw_rate",
    "black_loss_rate",
    "white_hof_size",
    "black_hof_size",
    "white_tournament_score_best",
    "white_tournament_score_avg",
    "black_tournament_score_best",
    "black_tournament_score_avg",
    "white_material_avg",
    "black_material_avg",
    "generation_time_sec",
]


def poll_metrics_with_timeout(
    run: wandb.sdk.wandb_run.Run,
    metrics_path: Path,
    max_generations: int,
    poll_interval: float,
    max_stale: int,
    timeout_minutes: float | None,
    log_keys: list[str] | None = None,
) -> tuple[dict | None, str]:
    """Tail metrics.jsonl with a hard timeout.

    Returns (final_metrics, status) where status is one of:
    "complete", "stale", "timeout".
    """
    jsonl_path = Path(str(metrics_path) + "l")
    file_pos = 0
    last_gen = -1
    stale_count = 0
    final_metrics = None
    start = time.time()

    while True:
        if timeout_minutes and (time.time() - start) > timeout_minutes * 60:
            return final_metrics, "timeout"

        new_records = []
        if jsonl_path.exists():
            try:
                with open(jsonl_path) as f:
                    f.seek(file_pos)
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            new_records.append(json.loads(line))
                        except Exception:
                            pass
                    file_pos = f.tell()
            except OSError:
                pass

        if not new_records and not jsonl_path.exists():
            snapshot = read_metrics(metrics_path)
            if snapshot and snapshot.get("generation", -1) > last_gen:
                new_records = [snapshot]

        for metrics in new_records:
            gen = metrics.get("generation", -1)
            if gen <= last_gen:
                continue
            last_gen = gen
            stale_count = 0
            final_metrics = metrics

            if log_keys:
                log_data = {k: metrics.get(k, 0) for k in log_keys}
            else:
                log_data = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}

            run.log(log_data)

        if last_gen >= max_generations - 1:
            return final_metrics, "complete"

        if not new_records:
            stale_count += 1
            if stale_count >= max_stale:
                return final_metrics, "stale"

        time.sleep(poll_interval)


def run_training_once(
    config: dict,
    project: str,
    entity: str | None,
    visible: bool,
    poll_interval: float,
    max_stale: int,
    timeout_minutes: float | None,
    worker_id: str | None = None,
) -> None:
    merged = DEFAULT_CONFIG.copy()
    merged.update(config or {})

    user_dir = godot_user_dir(APP_NAME)
    worker = SweepWorker(user_dir, worker_id=worker_id)
    worker.write_config(merged)
    worker.clear_metrics()

    elite_pool = GlobalElitePool(user_dir)

    # --- Global Elite: write seed file for Godot to read at startup ---
    seed_path = elite_pool.write_seed_file(worker.worker_id)
    if seed_path:
        pool_stats = elite_pool.stats()
        print(
            f"🧬 Global elite pool: {pool_stats['total_elites']} genomes from "
            f"{pool_stats['contributor_count']} run(s), "
            f"top fitness={pool_stats['top_fitness']:.2f}"
        )
        print(f"🧬 Seeding {min(5, pool_stats['total_elites'])} elite genome(s) into this run")
    else:
        print("🧬 No global elites found; starting fresh")

    run = None
    process = None

    try:
        run = wandb.init(
            project=project,
            entity=entity,
            config=merged,
            tags=["chess-evolve", "sweep"],
        )
        define_step_metric()
        if run is not None:
            print(f"🔗 W&B run: {run.url}")
            # Log elite pool stats to W&B so we can track cross-run improvement
            if seed_path:
                run.log({"global_elite/pool_size": pool_stats["total_elites"],
                         "global_elite/top_fitness": pool_stats["top_fitness"],
                         "global_elite/avg_fitness": pool_stats["avg_fitness"],
                         "global_elite/contributors": pool_stats["contributor_count"]})

        process = launch_godot(
            PROJECT_PATH,
            godot_path=GODOT_PATH,
            visible=visible,
            metrics_path=worker.metrics_path,
            worker_id=worker.worker_id,
        )

        if not wait_for_metrics(worker.metrics_path, timeout=120.0):
            print("❌ Metrics file never appeared; terminating run")
            if process is not None:
                process.kill()
            if run is not None:
                run.finish(exit_code=1)
            return

        max_gens = int(merged.get("max_generations", 50))
        final, status = poll_metrics_with_timeout(
            run,
            worker.metrics_path,
            max_generations=max_gens,
            poll_interval=poll_interval,
            max_stale=max_stale,
            timeout_minutes=timeout_minutes,
            log_keys=CHESS_LOG_KEYS,
        )

        if status == "timeout":
            print("⏰ Training timeout exceeded")
        elif status == "stale":
            print("❌ Training stalled (no new metrics)")

        log_final_summary(run, final)
        run.finish(exit_code=0 if status == "complete" else 1)
    except Exception as exc:
        print(f"❌ Training crashed: {exc}")
        if run is not None:
            run.finish(exit_code=1)
        raise
    finally:
        if process is not None:
            try:
                process.terminate()
                process.wait(timeout=10)
            except Exception:
                process.kill()

        # --- Global Elite: harvest this run's best genomes ---
        contrib_path = Path(user_dir) / f"elite_contrib_{worker.worker_id}.json"
        if contrib_path.exists():
            try:
                new_genomes = json.loads(contrib_path.read_text()).get("elites", [])
                if new_genomes:
                    kept = elite_pool.update_contrib(worker.worker_id, new_genomes)
                    print(f"🧬 Contributed {len(new_genomes)} genome(s) to global elite pool (total stored: {kept})")
                contrib_path.unlink()
            except (json.JSONDecodeError, OSError) as exc:
                print(f"⚠️  Could not harvest elite contributions: {exc}")

        # Clean up the seed file so stale genomes aren't re-read next time.
        elite_pool.cleanup_seed_file(worker.worker_id)

        worker.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser(description="Chess-Evolve sweep worker")
    parser.add_argument("--sweep-id", required=True, help="W&B sweep ID")
    parser.add_argument("--project", default="chess-evolve", help="W&B project")
    parser.add_argument("--entity", default=None, help="W&B entity")
    parser.add_argument("--count", type=int, default=1, help="Runs per agent")
    parser.add_argument("--visible", action="store_true", help="Show Godot window")
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--max-stale", type=int, default=300)
    parser.add_argument("--timeout-minutes", type=float, default=45.0)
    parser.add_argument("--worker-id", default=None, help="Optional fixed worker id")
    args = parser.parse_args()

    def train_fn():
        try:
            run_training_once(
                dict(wandb.config),
                project=args.project,
                entity=args.entity,
                visible=args.visible,
                poll_interval=args.poll_interval,
                max_stale=args.max_stale,
                timeout_minutes=args.timeout_minutes,
                worker_id=args.worker_id,
            )
        except Exception as exc:
            print(f"❌ Sweep worker failed: {exc}")

    run_sweep_agent(args.sweep_id, args.project, train_fn, count=args.count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
