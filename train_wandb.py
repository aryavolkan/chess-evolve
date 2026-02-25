#!/usr/bin/env python3
"""
Chess-Evolve W&B-tracked training with hyperparameter sweeps.
Uses shared Godot+W&B utilities for the launch→poll→log pipeline.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

_SHARED = next(
    (p for p in [
        os.path.expanduser("~/projects/shared-evolve-utils"),
        os.path.expanduser("~/Projects/shared-evolve-utils"),
        os.path.expanduser("~/shared-evolve-utils"),
    ] if os.path.isdir(p)),
    "",
)
if _SHARED:
    sys.path.insert(0, _SHARED)
_OVERNIGHT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "overnight-agent")
sys.path.insert(0, _OVERNIGHT)
import wandb  # noqa: E402
from godot_wandb import (  # noqa: E402
    SweepWorker,
    define_step_metric,
    godot_user_dir,
    launch_godot,
    log_final_summary,
    poll_metrics,
    run_training,
    wait_for_metrics,
)
from global_elite import GlobalElitePool  # noqa: E402

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Default config (overridden by W&B sweep)
DEFAULT_CONFIG = {
    "population_size": 20,
    "hidden_size": 64,
    "elite_count": 3,
    "crossover_rate": 0.70,
    "mutation_rate": 0.25,
    "mutation_strength": 0.12,
    "games_per_individual": 2,
    "max_generations": 50,
    "max_moves_per_game": 40,    # capped for GDScript performance
    "input_size": 389,
    "output_size": 4096,
    "use_minimax": False,        # minimax too slow in GDScript; direct NN output
    "use_tournament": True,
    "tournament_opponents": 2,   # 2 opponents/individual keeps gen time ~10s
    "immigration_rate": 0.1,     # fraction of population replaced with random individuals each gen
    "fitness_sharing_sigma": 25.0, # genetic distance threshold for fitness sharing
    "tournament_k": 2,           # tournament selection size (2 = less pressure than 3)
}

_PROJECT_PATH_DEFAULT = next(
    (p for p in [
        os.path.expanduser("~/projects/chess-evolve"),
        os.path.expanduser("~/Projects/chess-evolve"),
        os.path.expanduser("~/chess-evolve"),
        os.path.dirname(os.path.abspath(__file__)),
    ] if os.path.isfile(os.path.join(p, "project.godot"))),
    os.path.dirname(os.path.abspath(__file__)),
)
PROJECT_PATH = os.environ.get("CHESS_EVOLVE_PROJECT_PATH", _PROJECT_PATH_DEFAULT)

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
    "combined_best",
]


def _chess_metric_transform(log_data: dict) -> dict:
    """Add combined_best = min(white_best, black_best) for balanced optimization."""
    wb = log_data.get("white_best", 0)
    bb = log_data.get("black_best", 0)
    log_data["combined_best"] = min(wb, bb)
    return log_data


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
        log_keys=CHESS_LOG_KEYS,
    )


GODOT_PATH = os.environ.get(
    "GODOT_PATH",
    next((p for p in [
        os.path.expanduser("~/.local/bin/godot"),
        "/usr/local/bin/godot",
        "/opt/homebrew/bin/godot",
    ] if os.path.isfile(p)), "godot"),
)
USER_DIR = godot_user_dir("Chess Evolve")

_worker: SweepWorker = None


def sweep_train_fn():
    """Run one sweep trial with per-worker isolation."""
    global _worker
    _worker = SweepWorker(USER_DIR)

    run = wandb.init(project="chess-evolve", tags=["chess", "neuroevolution", "coevolution"])
    define_step_metric()

    # Merge defaults with sweep overrides
    config = DEFAULT_CONFIG.copy()
    config.update(dict(run.config))
    max_gens = config.get("max_generations", 50)

    _worker.clear_metrics()
    _worker.write_config(config)

    # --- Global Elite: seed this run from shared pool ---
    elite_pool = GlobalElitePool(USER_DIR)
    seed_path = elite_pool.write_seed_file(_worker.worker_id)
    if seed_path:
        pool_stats = elite_pool.stats()
        print(
            f"🧬 Global elite pool: {pool_stats['total_elites']} genomes from "
            f"{pool_stats['contributor_count']} run(s), "
            f"top fitness={pool_stats['top_fitness']:.2f}"
        )
        run.log({"global_elite/pool_size": pool_stats["total_elites"],
                 "global_elite/top_fitness": pool_stats["top_fitness"],
                 "global_elite/avg_fitness": pool_stats["avg_fitness"],
                 "global_elite/contributors": pool_stats["contributor_count"]})
    else:
        print("🧬 No global elites found; starting fresh")

    print(f"\n🎮 Worker {_worker.worker_id}: pop={config.get('population_size')}, gens={max_gens}")

    proc = launch_godot(
        PROJECT_PATH,
        godot_path=GODOT_PATH,
        visible=False,
        metrics_path=_worker.metrics_path,
        worker_id=_worker.worker_id,
    )

    try:
        if not wait_for_metrics(_worker.metrics_path, timeout=120.0):
            print("❌ Metrics file never appeared; terminating run")
            proc.kill()
            run.finish(exit_code=1)
            _worker.cleanup()
            return

        final = poll_metrics(run, _worker.metrics_path, max_gens, log_keys=CHESS_LOG_KEYS, metric_transform=_chess_metric_transform)

        # Wait for Godot to exit gracefully — it writes elite contrib in its quit handler
        try:
            proc.wait(timeout=20)
        except Exception:
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
        proc = None  # mark as handled

        # --- Global Elite: harvest this run's best genomes ---
        contrib_path = Path(USER_DIR) / f"elite_contrib_{_worker.worker_id}.json"
        if contrib_path.exists():
            try:
                new_genomes = json.loads(contrib_path.read_text()).get("elites", [])
                if new_genomes:
                    kept = elite_pool.update_contrib(_worker.worker_id, new_genomes)
                    print(f"🧬 Contributed {len(new_genomes)} genome(s) to global elite pool (total stored: {kept})")
                    artifact = wandb.Artifact(
                        f"elite-population-{_worker.worker_id}",
                        type="elite-population",
                        description=f"Elite genomes from worker {_worker.worker_id}",
                    )
                    artifact.add_file(str(contrib_path))
                    run.log_artifact(artifact)
            except (json.JSONDecodeError, OSError) as exc:
                print(f"⚠️  Could not harvest elite contributions: {exc}")

        elite_pool.cleanup_seed_file(_worker.worker_id)

        log_final_summary(run, final)
        print(f"✅ Worker {_worker.worker_id}: training complete!")
        run.finish(exit_code=0)

    except KeyboardInterrupt:
        print(f"\n🛑 Worker {_worker.worker_id}: interrupted")
        if proc is not None:
            proc.kill()
        run.finish(exit_code=130)
    except Exception as e:
        print(f"\n❌ Worker {_worker.worker_id}: error: {e}")
        if proc is not None:
            proc.kill()
        run.finish(exit_code=1)
    finally:
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        elite_pool.cleanup_seed_file(_worker.worker_id)
        _worker.cleanup()


def sweep_agent(sweep_id: str, count: int = 50):
    """W&B sweep agent with per-worker isolation."""
    wandb.agent(sweep_id, function=sweep_train_fn, count=count, project="chess-evolve")


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
