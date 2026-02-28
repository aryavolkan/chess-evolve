#!/usr/bin/env python3
"""
Chess-Evolve W&B-tracked training with hyperparameter sweeps.
Uses shared Godot+W&B utilities for the launch→poll→log pipeline.
Supports multiple backends: Rust CPU, PyTorch GPU/CPU, Godot.
"""
import argparse
import json
import os
import sys
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
from global_elite import GlobalElitePool  # noqa: E402
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

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Default config (overridden by W&B sweep)
DEFAULT_CONFIG = {
    "population_size": 30,
    "hidden_size": 64,
    "elite_count": 2,
    "crossover_rate": 0.70,
    "mutation_rate": 0.25,
    "mutation_strength": 0.12,
    "games_per_individual": 2,
    "max_generations": 50,
    "max_moves_per_game": 100,
    "input_size": 389,
    "output_size": 4096,
    "use_minimax": False,        # minimax too slow in GDScript; direct NN output
    "use_tournament": True,
    "tournament_opponents": 5,   # 5 opponents/individual for stronger selection pressure
    "immigration_rate": 0.1,     # fraction of population replaced with random individuals each gen
    "fitness_sharing_sigma": 0.08, # RMS genetic distance threshold for fitness sharing
    "tournament_k": 2,           # tournament selection size (2 = less pressure than 3)
    "use_opening_book": True,
    "opening_book_depth": 6,
    "move_temperature": 0.5,     # softmax temperature for move selection (0 = deterministic)
    # NEAT-specific defaults
    "use_neat": False,
    "neat_add_node_rate": 0.15,
    "neat_add_connection_rate": 0.25,
    "neat_initial_connections_per_output": 10,
    "neat_target_species_count": 5,
    "neat_seed_genome_path": "",
    "neat_save_genome_path": "user://neat_best_genome.json",
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
    # Elo tracking
    "white_hof_avg_elo",
    "black_hof_avg_elo",
    "white_hof_top_elo",
    "black_hof_top_elo",
    "white_elo_min",
    "white_elo_p25",
    "white_elo_median",
    "white_elo_p75",
    "white_elo_max",
    "black_elo_min",
    "black_elo_p25",
    "black_elo_median",
    "black_elo_p75",
    "black_elo_max",
    # NEAT topology metrics
    "neat_hidden_nodes_avg",
    "neat_connections_avg",
    "neat_species_count",
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


# --- Backend detection (lightweight, no heavy imports) ---

def _detect_rust() -> bool:
    """Check if Rust CPU training crates (chess_cpu + evolve_ga) are available."""
    try:
        import importlib.util
        return (importlib.util.find_spec("chess_cpu") is not None
                and importlib.util.find_spec("evolve_ga") is not None
                and os.path.isfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpu_trainer.py")))
    except Exception:
        return False


def _detect_pytorch() -> bool:
    """Check if PyTorch + python-chess + gpu_trainer are importable."""
    try:
        import importlib.util
        for mod in ("torch", "chess"):
            if importlib.util.find_spec(mod) is None:
                return False
        return os.path.isfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu_trainer.py"))
    except Exception:
        return False


_USE_RUST = _detect_rust()
_USE_PYTORCH = _detect_pytorch()


# --- Training backends ---

def _run_rust_training(run, config, max_gens, elite_pool):
    """Run training via Rust CPU trainer (chess_cpu + evolve_ga)."""
    from cpu_trainer import CPUTrainer

    print("🦀 Rust CPU training mode")

    trainer = CPUTrainer(config, _worker.metrics_path)

    def on_generation(metrics):
        log_data = {k: metrics.get(k, 0) for k in CHESS_LOG_KEYS if k in metrics}
        log_data = _chess_metric_transform(log_data)
        run.log(log_data)
        gen = metrics.get("generation", 0)
        t = metrics.get("generation_time_sec", 0)
        gps = metrics.get("games_per_sec", 0)
        wb = metrics.get("white_best", 0)
        bb = metrics.get("black_best", 0)
        print(f"  gen {gen}: w_best={wb:.2f} b_best={bb:.2f} {t:.2f}s ({gps:.0f} games/s)")

    final = trainer.train(max_generations=max_gens, on_generation=on_generation)

    _harvest_elites(run, elite_pool)
    log_final_summary(run, final)
    print(f"✅ Worker {_worker.worker_id}: Rust CPU training complete!")
    run.finish(exit_code=0)


def _run_pytorch_training(run, config, max_gens, elite_pool, device="cuda"):
    """Run training via PyTorch trainer (GPU or CPU, no Godot)."""
    from gpu_trainer import GPUTrainer

    print(f"🚀 PyTorch training mode (device: {device})")

    trainer = GPUTrainer(config, _worker.metrics_path, device=device)

    def on_generation(metrics):
        log_data = {k: metrics.get(k, 0) for k in CHESS_LOG_KEYS if k in metrics}
        log_data = _chess_metric_transform(log_data)
        run.log(log_data)
        gen = metrics.get("generation", 0)
        t = metrics.get("generation_time_sec", 0)
        gps = metrics.get("games_per_sec", 0)
        wb = metrics.get("white_best", 0)
        bb = metrics.get("black_best", 0)
        print(f"  gen {gen}: w_best={wb:.2f} b_best={bb:.2f} {t:.2f}s ({gps:.0f} games/s)")

    final = trainer.train(max_generations=max_gens, on_generation=on_generation)

    _harvest_elites(run, elite_pool)
    log_final_summary(run, final)
    print(f"✅ Worker {_worker.worker_id}: PyTorch training complete!")
    run.finish(exit_code=0)


def _run_godot_training(run, config, max_gens, elite_pool):
    """Run training via Godot subprocess (original path)."""
    proc = launch_godot(
        PROJECT_PATH,
        godot_path=GODOT_PATH,
        visible=False,
        metrics_path=_worker.metrics_path,
        worker_id=_worker.worker_id,
    )

    try:
        # NEAT networks are much slower than fixed-topology; allow more time
        use_neat = config.get("use_neat", False)
        start_timeout = 600.0 if use_neat else 120.0
        if not wait_for_metrics(_worker.metrics_path, timeout=start_timeout):
            print("❌ Metrics file never appeared; terminating run")
            proc.kill()
            run.finish(exit_code=1)
            _worker.cleanup()
            return

        # max_stale scales with population: large pops take minutes per gen
        # NEAT is ~10x slower per individual due to variable topology
        pop_size = config.get("population_size", 30)
        per_individual = 10 if use_neat else 3
        stale_timeout = max(120, pop_size * per_individual)
        final = poll_metrics(run, _worker.metrics_path, max_gens, log_keys=CHESS_LOG_KEYS, metric_transform=_chess_metric_transform, max_stale=stale_timeout)

        # Wait for Godot to exit gracefully
        try:
            proc.wait(timeout=20)
        except Exception:
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
        proc = None

        _harvest_elites(run, elite_pool)
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


def _harvest_elites(run, elite_pool):
    """Harvest elite contributions from this run and upload to W&B."""
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

    # Determine backend: Rust > PyTorch GPU > PyTorch CPU > Godot
    # Skip torch import when CUDA disabled (saves ~3GB RSS for Rust/Godot workers)
    _has_cuda = False
    if _USE_PYTORCH and os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        try:
            import torch
            _has_cuda = torch.cuda.is_available()
        except ImportError:
            pass

    try:
        if _has_cuda:
            _run_pytorch_training(run, config, max_gens, elite_pool, device="cuda")
        elif _USE_RUST:
            _run_rust_training(run, config, max_gens, elite_pool)
        elif _USE_PYTORCH:
            _run_pytorch_training(run, config, max_gens, elite_pool, device="cpu")
        else:
            print("⚙️  Godot training mode")
            _run_godot_training(run, config, max_gens, elite_pool)
    except KeyboardInterrupt:
        print(f"\n🛑 Worker {_worker.worker_id}: interrupted")
        run.finish(exit_code=130)
    except Exception as e:
        print(f"\n❌ Worker {_worker.worker_id}: error: {e}")
        import traceback
        traceback.print_exc()
        run.finish(exit_code=1)
    finally:
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
