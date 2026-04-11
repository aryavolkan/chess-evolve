#!/usr/bin/env python3
"""Test: multiprocessing.Process + wandb.init in child → crash?"""
import sys, os, multiprocessing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def train_in_child(result_q, run_num):
    """Runs inside multiprocessing.Process WITH wandb."""
    import wandb
    from pathlib import Path
    from neat_cpu_trainer import NeatCPUTrainer

    config = {
        "population_size": 150, "hidden_size": 64, "games_per_individual": 2,
        "mutation_rate": 0.25, "mutation_strength": 0.12, "max_generations": 250,
        "tournament_k": 5, "tournament_opponents": 10, "use_neat": True,
        "curriculum_stage": 2, "checkmate_bonus": 15, "input_size": 389,
        "output_size": 384, "max_moves_per_game": 100, "eval_temperature": 0.0,
        "benchmark_size": 50, "draw_bonus": 1.5, "sf_fitness_weight": 0.18,
        "sf_fitness_top_n": 50, "sf_bench_interval": 5, "transition_gens": 3,
        "neat_initial_connections_per_output": 1, "neat_target_species_count": 5,
        "immigration_rate": 0.14, "crossover_rate": 0.76, "move_temperature": 0.39,
        "neat_add_node_rate": 0.17, "neat_add_connection_rate": 0.25,
    }

    run = wandb.init(project="chess-evolve", config=config, tags=["debug-mp-wandb"],
                     name=f"debug-mp-{run_num}")
    trainer = NeatCPUTrainer(config, Path(f"/tmp/debug_mp_wandb_metrics_{run_num}"))

    def on_gen(gen_data):
        gen = gen_data.get("generation", 0)
        run.log(gen_data, step=gen)
        if gen % 50 == 0:
            print(f"  [child {run_num}] gen {gen}")

    try:
        result = trainer.train(max_generations=250, on_generation=on_gen)
        run.finish()
        result_q.put(("ok", result.get("generation", 0)))
    except Exception as exc:
        print(f"\n  [child {run_num}] *** CRASH gen={exc}: {type(exc).__module__}.{type(exc).__qualname__}: {exc!r}")
        import traceback
        traceback.print_exc()
        run.finish(exit_code=1)
        result_q.put(("error", f"gen=? {type(exc).__qualname__}"))

if __name__ == "__main__":
    os.environ["RAYON_NUM_THREADS"] = "4"

    results = []
    for i in range(5):
        print(f"\n=== Run {i+1}/5 ===")
        q = multiprocessing.Queue()
        p = multiprocessing.Process(target=train_in_child, args=(q, i+1))
        p.start()
        p.join(timeout=600)
        if p.is_alive():
            p.kill()
            results.append("timeout")
            print("  TIMEOUT")
        elif not q.empty():
            status, detail = q.get_nowait()
            results.append(status)
            print(f"  Result: {status} ({detail})")
        else:
            results.append(f"exit-{p.exitcode}")
            print(f"  Exit code: {p.exitcode}")

    print(f"\nSummary: {results}")
    ok = results.count("ok")
    err = results.count("error")
    print(f"  {ok}/5 ok, {err}/5 errors")
