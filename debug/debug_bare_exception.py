#!/usr/bin/env python3
"""Minimal reproducer for bare Exception() from simulate_neat_games_batch.

Runs NEAT evolution in a loop, calling simulate_neat_games_batch each gen,
to isolate when/why the bare Exception() is raised.
"""
import sys
import json
import chess_cpu
import neat_ga

POP_SIZE = 150
OUTPUT_SIZE = 384
MAX_MOVES = 100
TEMPERATURE = 0.05  # low temp that the crashes happen at
NUM_OPP = 30

# Create a NEAT population
config = {
    "population_size": POP_SIZE,
    "input_size": 389,
    "output_size": OUTPUT_SIZE,
    "weight_mutate_rate": 0.8,
    "weight_perturb_rate": 0.9,
    "weight_perturb_strength": 0.12,
    "weight_reset_range": 1.0,
    "add_node_rate": 0.17,
    "add_connection_rate": 0.25,
    "crossover_rate": 0.76,
    "survival_fraction": 0.5,
    "elite_fraction": 0.1,
    "target_species_count": 5,
    "compatibility_threshold": 3.0,
    "excess_coefficient": 1.0,
    "disjoint_coefficient": 1.0,
    "weight_coefficient": 0.4,
    "threshold_step": 0.3,
    "min_species_protected": 2,
    "stagnation_limit": 15,
    "interspecies_crossover_rate": 0.01,
}

def make_pop(cfg):
    cfg_json = json.dumps(cfg)
    init = neat_ga.create_population_with_tracker(cfg_json)
    pop = neat_ga.NeatPopulation.from_json(
        init["population"], cfg_json, init["tracker"], None,
    )
    return pop

print(f"Creating populations (pop={POP_SIZE}, out={OUTPUT_SIZE})...")
white_pop = make_pop(config)
black_pop = make_pop(config)

# Create fixed random opponents
import random
random.seed(42)
opp_pop = make_pop(config)
opp_genomes = opp_pop.get_genomes_json()[:NUM_OPP]

pairings = [(w, b) for w in range(POP_SIZE) for b in range(NUM_OPP)]

print(f"Starting evolution loop (temp={TEMPERATURE}, pairings={len(pairings)})...")
for gen in range(1, 501):
    w_genomes = white_pop.get_genomes_json()
    b_genomes = black_pop.get_genomes_json()

    # Measure genome complexity
    avg_len = sum(len(g) for g in w_genomes) / POP_SIZE
    max_len = max(len(g) for g in w_genomes)

    try:
        w_results = chess_cpu.simulate_neat_games_batch(
            w_genomes, opp_genomes, pairings,
            output_size=OUTPUT_SIZE, max_moves=MAX_MOVES,
            temperature=TEMPERATURE,
            mercy_min_moves=30, mercy_material_threshold=10.0,
        )
    except Exception as exc:
        print(f"\n*** WHITE CRASH gen={gen}: {type(exc).__module__}.{type(exc).__qualname__}: {exc!r}")
        print(f"    avg_genome_json_len={avg_len:.0f} max={max_len}")
        print(f"    __cause__={exc.__cause__!r} __context__={exc.__context__!r}")
        print(f"    __traceback__={exc.__traceback__}")
        # Try with fewer pairings
        try:
            chess_cpu.simulate_neat_games_batch(
                w_genomes[:10], opp_genomes[:2], [(w, b) for w in range(10) for b in range(2)],
                output_size=OUTPUT_SIZE, max_moves=MAX_MOVES,
                temperature=TEMPERATURE,
                mercy_min_moves=30, mercy_material_threshold=10.0,
            )
            print("    Small batch SUCCEEDED — issue is scale-dependent")
        except Exception as e2:
            print(f"    Small batch ALSO FAILED: {type(e2).__qualname__}: {e2!r}")
        sys.exit(1)

    try:
        b_pairings = [(w, b) for w in range(NUM_OPP) for b in range(POP_SIZE)]
        b_results = chess_cpu.simulate_neat_games_batch(
            opp_genomes, b_genomes, b_pairings,
            output_size=OUTPUT_SIZE, max_moves=MAX_MOVES,
            temperature=TEMPERATURE,
            mercy_min_moves=30, mercy_material_threshold=10.0,
        )
    except Exception as exc:
        print(f"\n*** BLACK CRASH gen={gen}: {type(exc).__module__}.{type(exc).__qualname__}: {exc!r}")
        print(f"    avg_genome_json_len={avg_len:.0f} max={max_len}")
        sys.exit(1)

    # Simple fitness: count wins
    w_fit = [0.0] * POP_SIZE
    b_fit = [0.0] * POP_SIZE
    for r in w_results:
        if r["result"] == 1:
            w_fit[r["white_idx"]] += 1.0
    for r in b_results:
        if r["result"] == -1:
            b_fit[r["black_idx"]] += 1.0

    w_stats = white_pop.evolve(w_fit)
    b_stats = black_pop.evolve(b_fit)

    if gen % 10 == 0:
        print(f"  gen {gen}: w_best={w_stats['best_fitness']:.1f} b_best={b_stats['best_fitness']:.1f} "
              f"avg_json={avg_len:.0f} conns={w_stats['avg_connections']:.0f}")
