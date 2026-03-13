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
_PYTHON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "python")
sys.path.insert(0, _PYTHON)
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
    "output_size": 384,
    "use_minimax": False,        # minimax too slow in GDScript; direct NN output
    "use_tournament": True,
    "tournament_opponents": 5,   # 5 opponents/individual for stronger selection pressure
    "immigration_rate": 0.1,     # fraction of population replaced with random individuals each gen
    "fitness_sharing_sigma": 0.08, # RMS genetic distance threshold for fitness sharing
    "tournament_k": 2,           # tournament selection size (2 = less pressure than 3)
    "use_opening_book": True,
    "opening_book_depth": 6,
    "move_temperature": 0.5,     # softmax temperature for move selection (0 = deterministic)
    # Fitness tuning
    "draw_bonus": 3.0,              # reward surviving to draw (was 0.0)
    "capture_weight": 0.2,          # reduced — random trading shouldn't dominate
    "opp_king_safety_weight": 0.0,  # removed — penalizes opponent having intact pawns
    "benchmark_fitness_weight": 0.3, # blend absolute fitness to counter coevo oscillation
    # NEAT-specific defaults
    "use_neat": True,
    "neat_add_node_rate": 0.15,
    "neat_add_connection_rate": 0.25,
    "neat_initial_connections_per_output": 1,
    "neat_target_species_count": 5,
    "neat_seed_genome_path": "",
    "neat_save_genome_path": "user://neat_best_genome.json",
    # Stockfish CPL fitness signal
    "sf_fitness_weight": 0.4,       # stronger SF signal for better gradient
    "sf_fitness_interval": 1,       # every generation for consistent signal
    "sf_fitness_top_n": 50,         # test top 50 genomes per color
    # Puzzle curriculum
    "puzzle_stage0_file": "data/puzzles/stage0_puzzles.json",
    "puzzle_stage1_file": "data/puzzles/stage1_puzzles.json",
    "puzzle_batch_size": 50,
    "puzzle_accuracy_threshold": 0.5,
    "puzzle_temperature": 0.1,
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
]


def _chess_metric_transform(log_data: dict) -> dict:
    """Add combined_best = min(white_best, black_best) for balanced optimization."""
    wb = log_data.get("white_best", 0)
    bb = log_data.get("black_best", 0)
    log_data["combined_best"] = min(wb, bb)
    return log_data


def do_training(config=None, visible=False):
    """Run a single training session.

    Auto-detects backend: Rust CPU > PyTorch GPU > PyTorch CPU > Godot.
    """
    global _worker
    merged = DEFAULT_CONFIG.copy()
    if config:
        merged.update(config)

    # Check if a non-Godot backend is available
    _has_cuda = False
    if _USE_PYTORCH and os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        try:
            import torch
            _has_cuda = torch.cuda.is_available()
        except ImportError:
            pass

    if _has_cuda or _USE_RUST or _USE_PYTORCH:
        # Use sweep infrastructure for worker isolation + W&B logging
        _worker = SweepWorker(USER_DIR)
        run = wandb.init(
        project="chess-evolve",
        tags=["chess", "neuroevolution", "coevolution"],
        settings=wandb.Settings(init_timeout=300),
    )
        define_step_metric()
        run.config.update(merged)
        max_gens = merged.get("max_generations", 50)
        _worker.clear_metrics()
        _worker.write_config(merged)

        print(f"\n🎮 Training: pop={merged.get('population_size')}, gens={max_gens}")

        try:
            if _has_cuda:
                _run_pytorch_training(run, merged, max_gens, device="cuda")
            elif _USE_RUST:
                _run_rust_training(run, merged, max_gens)
            elif _USE_PYTORCH:
                _run_pytorch_training(run, merged, max_gens, device="cpu")
        except KeyboardInterrupt:
            print(f"\n🛑 Worker {_worker.worker_id}: interrupted")
            run.finish(exit_code=130)
        except Exception as e:
            print(f"\n❌ Worker {_worker.worker_id}: error: {e}")
            import traceback
            traceback.print_exc()
            run.finish(exit_code=1)
        finally:
            _worker.cleanup()
    else:
        # Godot fallback
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
                and os.path.isfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), "python", "cpu_trainer.py")))
    except Exception:
        return False


def _detect_neat_rust() -> bool:
    """Check if NEAT Rust crate (neat_ga) is available for NEAT training."""
    try:
        import importlib.util
        return (importlib.util.find_spec("neat_ga") is not None
                and importlib.util.find_spec("chess_cpu") is not None
                and os.path.isfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), "python", "neat_cpu_trainer.py")))
    except Exception:
        return False


def _detect_pytorch() -> bool:
    """Check if PyTorch + python-chess + gpu_trainer are importable."""
    try:
        import importlib.util
        for mod in ("torch", "chess"):
            if importlib.util.find_spec(mod) is None:
                return False
        return os.path.isfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), "python", "gpu_trainer.py"))
    except Exception:
        return False


_USE_RUST = _detect_rust()
_USE_NEAT_RUST = _detect_neat_rust()
_USE_PYTORCH = _detect_pytorch()


# --- Training backends ---

def _make_generation_callback(run):
    """Create a per-generation callback that logs metrics to W&B and prints progress."""
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
    return on_generation


def _run_rust_training(run, config, max_gens):
    """Run training via Rust CPU trainer (chess_cpu + evolve_ga or neat_ga)."""
    if config.get("use_neat", False) and _USE_NEAT_RUST:
        from neat_cpu_trainer import NeatCPUTrainer
        print("🦀 Rust NEAT CPU training mode")
        trainer = NeatCPUTrainer(config, _worker.metrics_path)
    else:
        from cpu_trainer import CPUTrainer
        print("🦀 Rust CPU training mode")
        trainer = CPUTrainer(config, _worker.metrics_path)

    final = trainer.train(max_generations=max_gens, on_generation=_make_generation_callback(run))

    log_final_summary(run, final)
    print(f"✅ Worker {_worker.worker_id}: Rust CPU training complete!")
    run.finish(exit_code=0)


def _run_pytorch_training(run, config, max_gens, device="cuda"):
    """Run training via PyTorch trainer (GPU or CPU, no Godot)."""
    from gpu_trainer import GPUTrainer

    print(f"🚀 PyTorch training mode (device: {device})")

    trainer = GPUTrainer(config, _worker.metrics_path, device=device)

    final = trainer.train(max_generations=max_gens, on_generation=_make_generation_callback(run))

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

        _harvest_godot_elites(run, elite_pool)
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


def _harvest_godot_elites(run, elite_pool):
    """Harvest elite contributions from a Godot run and upload to W&B."""
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

    run = wandb.init(
        project="chess-evolve",
        tags=["chess", "neuroevolution", "coevolution"],
        settings=wandb.Settings(init_timeout=300),
    )
    define_step_metric()

    # Merge defaults with sweep overrides
    config = DEFAULT_CONFIG.copy()
    config.update(dict(run.config))
    max_gens = config.get("max_generations", 50)

    # Seed NEAT trials from best genome if available
    if config.get("use_neat", False):
        _seed_path = Path("neat_best_genomes.json")
        if _seed_path.exists():
            config["seed_genome_path"] = str(_seed_path)
            print(f"🧬 Seeding NEAT population from {_seed_path}")

    _worker.clear_metrics()
    _worker.write_config(config)

    print(f"\n🎮 Worker {_worker.worker_id}: pop={config.get('population_size')}, gens={max_gens}")

    # Determine backend: Rust > PyTorch GPU > PyTorch CPU > Godot
    _has_cuda = False
    if _USE_PYTORCH and os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        try:
            import torch
            _has_cuda = torch.cuda.is_available()
        except ImportError:
            pass

    try:
        if _has_cuda:
            _run_pytorch_training(run, config, max_gens, device="cuda")
        elif _USE_RUST:
            _run_rust_training(run, config, max_gens)
        elif _USE_PYTORCH:
            _run_pytorch_training(run, config, max_gens, device="cpu")
        else:
            # Godot path still uses GlobalElitePool
            print("⚙️  Godot training mode")
            elite_pool = GlobalElitePool(USER_DIR)
            elite_pool.write_seed_file(_worker.worker_id)
            _run_godot_training(run, config, max_gens, elite_pool)
            elite_pool.cleanup_seed_file(_worker.worker_id)
    except KeyboardInterrupt:
        print(f"\n🛑 Worker {_worker.worker_id}: interrupted")
        run.finish(exit_code=130)
    except Exception as e:
        print(f"\n❌ Worker {_worker.worker_id}: error: {e}")
        import traceback
        traceback.print_exc()
        run.finish(exit_code=1)
    finally:
        _worker.cleanup()


def sweep_agent(sweep_id: str, count: int = 50):
    """W&B sweep agent with per-worker isolation."""
    wandb.agent(sweep_id, function=sweep_train_fn, count=count, project="chess-evolve")


def do_chained_training(config=None, visible=False, max_chains=10):
    """Run training in a loop, seeding each run from the previous best.

    Stops when benchmark win rate does not improve between runs.
    Only overwrites the seed file when the new run improves on the previous best.
    """
    save_path = Path(config.get("save_genome_path", "neat_best_genomes.json")) if config else Path("neat_best_genomes.json")
    # Trainer writes to this temp path; we promote to save_path only on improvement
    candidate_path = save_path.with_suffix(".candidate.json")
    best_bench_wr = 0.0

    # Check if seed file already has a baseline
    if save_path.exists():
        try:
            prev = json.loads(save_path.read_text())
            best_bench_wr = prev.get("bench_avg_win_rate", prev.get("bench_white_win_rate", 0.0))
            print(f"  Seed file found: previous bench_avg_win_rate={best_bench_wr:.3f}")
        except (json.JSONDecodeError, OSError):
            pass

    for chain_idx in range(max_chains):
        print(f"\n{'='*60}")
        print(f"  Chain run {chain_idx + 1}/{max_chains}")
        print(f"  Previous best bench_white_win_rate: {best_bench_wr:.3f}")
        print(f"{'='*60}\n")

        # Point seed_genome_path at best-so-far save file (if it exists)
        run_config = dict(config) if config else {}
        if save_path.exists() and chain_idx > 0:
            run_config["seed_genome_path"] = str(save_path)
        # Trainer saves to candidate path so we don't overwrite best on regression
        run_config["save_genome_path"] = str(candidate_path)

        do_training(run_config, visible=visible)

        # Check if benchmark improved
        if candidate_path.exists():
            try:
                result = json.loads(candidate_path.read_text())
                new_bench_wr = result.get("bench_avg_win_rate", result.get("bench_white_win_rate", 0.0))
                print(f"\n  Chain run {chain_idx + 1} bench_avg_win_rate: {new_bench_wr:.3f} (prev: {best_bench_wr:.3f})")
                if new_bench_wr > best_bench_wr + 0.01:
                    best_bench_wr = new_bench_wr
                    # Promote candidate to best
                    candidate_path.rename(save_path)
                    print("  Improvement! Saved as best seed. Continuing to next chain run...")
                else:
                    # Discard candidate, keep previous best
                    candidate_path.unlink(missing_ok=True)
                    print(f"  No improvement (delta={new_bench_wr - best_bench_wr:.3f}). Stopping chain.")
                    break
            except (json.JSONDecodeError, OSError):
                print("  Could not read save file. Stopping chain.")
                break
        else:
            print("  No save file produced. Stopping chain.")
            break

    print(f"\n  Chained training complete. Best bench_avg_win_rate: {best_bench_wr:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chess-Evolve W&B training")
    parser.add_argument("--sweep", type=str, help="W&B sweep ID to join")
    parser.add_argument("--visible", action="store_true", help="Show Godot window")
    parser.add_argument("--config", type=str, help="JSON config file")
    parser.add_argument("--chain", type=int, default=0, metavar="N",
                        help="Auto-chain up to N runs, seeding each from previous best")
    args = parser.parse_args()

    custom_config = None
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            custom_config = json.load(f)
        print(f"✓ Loaded config from {args.config}")

    if args.sweep:
        print(f"🔄 Joining W&B sweep: {args.sweep}")
        sweep_agent(args.sweep)
    elif args.chain > 0:
        do_chained_training(custom_config, visible=args.visible, max_chains=args.chain)
    else:
        do_training(custom_config, visible=args.visible)
