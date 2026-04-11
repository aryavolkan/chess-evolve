#!/usr/bin/env python3
"""Run real NeatCPUTrainer WITH wandb but WITHOUT sweep agent."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import wandb
from pathlib import Path
from neat_cpu_trainer import NeatCPUTrainer

config = {
    "population_size": 150,
    "hidden_size": 64,
    "games_per_individual": 2,
    "mutation_rate": 0.25,
    "mutation_strength": 0.12,
    "max_generations": 250,
    "tournament_k": 5,
    "tournament_opponents": 10,
    "use_neat": True,
    "curriculum_stage": 2,
    "checkmate_bonus": 15,
    "input_size": 389,
    "output_size": 384,
    "max_moves_per_game": 100,
    "eval_temperature": 0.0,
    "benchmark_size": 50,
    "draw_bonus": 1.5,
    "sf_fitness_weight": 0.18,
    "sf_fitness_top_n": 50,
    "sf_bench_interval": 5,
    "transition_gens": 3,
    "neat_initial_connections_per_output": 1,
    "neat_target_species_count": 5,
    "immigration_rate": 0.14,
    "crossover_rate": 0.76,
    "move_temperature": 0.39,
    "neat_add_node_rate": 0.17,
    "neat_add_connection_rate": 0.25,
}

run = wandb.init(project="chess-evolve", config=config, tags=["debug-bare-exception"])
trainer = NeatCPUTrainer(config, Path("/tmp/debug_wandb_metrics"))

def on_gen(gen_data):
    gen = gen_data.get("generation", 0)
    run.log(gen_data, step=gen)
    if gen % 10 == 0:
        print(f"  gen {gen}: w_best={gen_data.get('white_best', 0):.2f} b_best={gen_data.get('black_best', 0):.2f}")

print("Starting training WITH wandb (no sweep agent)...")
try:
    result = trainer.train(max_generations=250, on_generation=on_gen)
    print(f"Completed 250 gens!")
except Exception as exc:
    print(f"\n*** CRASH: {type(exc).__module__}.{type(exc).__qualname__}: {exc!r}")
    import traceback
    traceback.print_exc()
    run.finish(exit_code=1)
    sys.exit(1)

run.finish()
