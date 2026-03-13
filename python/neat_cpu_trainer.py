"""
NEAT CPU-based chess neuroevolution trainer using Rust backends.

Uses neat_ga for NEAT evolution (speciation, crossover, topology mutation)
and chess_cpu for parallel game simulation with sparse neural networks.
Drop-in replacement for CPUTrainer when use_neat=True.
"""
from __future__ import annotations

import heapq
import json
import os
import random
import shutil
import time
from collections.abc import Callable
from pathlib import Path

import chess_cpu
import neat_ga

from fitness import (
    aggregate_game_stats,
    blend_fitness,
    compute_fitness,
    compute_fitness_breakdown,
    compute_outcome_rates,
    compute_tournament_scores,
    merge_fitness_weights,
)


class NeatCPUTrainer:
    """NEAT-based training loop using Rust chess simulation + NEAT GA operators.

    Unlike CPUTrainer which uses fixed-topology dense networks, NEAT evolves
    variable-topology sparse networks. Genomes cross the Python/Rust boundary
    as JSON strings.
    """

    def __init__(self, config: dict, metrics_path: Path):
        self.config = config
        self.metrics_path = Path(metrics_path)

        # Population parameters
        self.pop_size = config.get("population_size", 50)
        self.input_size = config.get("input_size", 389)
        self.output_size = config.get("output_size", 4096)
        self.max_moves = config.get("max_moves_per_game", 200)
        self.temperature = config.get("move_temperature", 0.5)
        self.tournament_opponents = config.get("tournament_opponents", 5)

        # Mercy rule
        self.mercy_min_moves = config.get("mercy_min_moves", 30)
        self.mercy_material_threshold = config.get("mercy_material_threshold", 12.0)

        # Fitness weights (shared defaults from fitness.py, config-overridable)
        self.fitness_weights = merge_fitness_weights(config)

        # NEAT config for Rust
        self.neat_config = {
            "input_count": self.input_size,
            "output_count": self.output_size,
            "population_size": self.pop_size,
            "initial_connections_per_output": config.get("initial_connections_per_output",
                                                          config.get("neat_initial_connections_per_output", 5)),
            "compatibility_threshold": config.get("compatibility_threshold", 3.0),
            "c1_excess": config.get("c1_excess", 1.0),
            "c2_disjoint": config.get("c2_disjoint", 1.0),
            "c3_weight_diff": config.get("c3_weight_diff", 0.4),
            "target_species_count": config.get("target_species_count",
                                                config.get("neat_target_species_count", 4)),
            "threshold_step": config.get("threshold_step", 0.3),
            "weight_mutate_rate": config.get("weight_mutate_rate", 0.8),
            "weight_perturb_rate": config.get("weight_perturb_rate", 0.9),
            "weight_perturb_strength": config.get("weight_perturb_strength", 0.3),
            "weight_reset_range": config.get("weight_reset_range", 2.0),
            "add_node_rate": config.get("add_node_rate",
                                         config.get("neat_add_node_rate", 0.10)),
            "add_connection_rate": config.get("add_connection_rate",
                                               config.get("neat_add_connection_rate", 0.20)),
            "disable_connection_rate": config.get("disable_connection_rate", 0.01),
            "add_node_count": config.get("add_node_count", 1),
            "add_connection_count": config.get("add_connection_count", 1),
            "prune_rate": config.get("prune_rate", 0.1),
            "complexity_cost": config.get("complexity_cost", 0.0),
            "elite_fraction": config.get("elite_fraction", 0.1),
            "survival_fraction": config.get("survival_fraction", 0.5),
            "crossover_rate": config.get("crossover_rate", 0.75),
            "interspecies_crossover_rate": config.get("interspecies_crossover_rate", 0.001),
            "disabled_gene_inherit_rate": config.get("disabled_gene_inherit_rate", 0.75),
            "stagnation_threshold": config.get("stagnation_threshold", 15),
            "stagnation_kill_threshold": config.get("stagnation_kill_threshold", 25),
            "min_species_protected": config.get("min_species_protected", 2),
        }

        # Hall of Fame (stored as JSON strings)
        self.white_hof: list[tuple[float, str]] = []
        self.black_hof: list[tuple[float, str]] = []
        self.hof_max_size = self.pop_size

        # Benchmark fitness blending: fraction of selection fitness from benchmark vs random
        self.benchmark_fitness_weight = config.get("benchmark_fitness_weight", 0.0)

        # Curriculum learning: stage 0 = coevolution (default), stage 1 = vs random
        self.curriculum_stage = config.get("curriculum_stage", 0)
        self.curriculum_random_opponents = config.get("curriculum_random_opponents", 30)
        self.curriculum_promotion_threshold = config.get("curriculum_promotion_threshold", 0.80)

        # Seed genome paths: if set, initialize population from saved best topology
        _seed = config.get("seed_genome_path", "")
        self.seed_genome_path = Path(_seed) if _seed else None
        self.save_genome_path = Path(config.get("save_genome_path", "neat_best_genomes.json"))

        # Fixed random benchmark population for absolute progress measurement.
        self.benchmark_size = 20
        self.benchmark_genomes = self._init_benchmark()

        # Stockfish benchmark: play best genome vs Stockfish every N generations
        self.sf_bench_interval = config.get("sf_bench_interval", 10)
        self.sf_bench_games = config.get("sf_bench_games", 6)
        self.sf_skill_level = config.get("sf_skill_level", 0)
        self.sf_move_time = config.get("sf_move_time", 0.05)  # seconds per move
        self._stockfish_path = shutil.which("stockfish") or os.environ.get("STOCKFISH_PATH", "")

        # Stockfish fitness signal: CPL-based fitness bonus for top N genomes
        self.sf_fitness_weight = config.get("sf_fitness_weight", 0.0)
        self.sf_fitness_interval = config.get("sf_fitness_interval", 5)
        self.sf_fitness_top_n = config.get("sf_fitness_top_n", 10)

        if self._stockfish_path:
            parts = [f"skill {self.sf_skill_level}", f"bench every {self.sf_bench_interval} gens"]
            if self.sf_fitness_weight > 0:
                parts.append(f"fitness weight {self.sf_fitness_weight}")
            print(f"  Stockfish found: {self._stockfish_path} ({', '.join(parts)})")
        else:
            print("  Stockfish not found; sf_bench disabled")

    def _load_seed(self) -> dict | None:
        """Load seed genomes from file if it exists."""
        if self.seed_genome_path and self.seed_genome_path.exists():
            try:
                return json.loads(self.seed_genome_path.read_text())
            except (json.JSONDecodeError, OSError) as e:
                print(f"  Warning: could not load seed genome: {e}")
        return None

    def _save_best(self, white_pop: list[str], black_pop: list[str],
                   white_fitness: list[float], black_fitness: list[float],
                   bench_w_wr: float, bench_b_wr: float):
        """Save best white and black genomes independently when each improves."""
        prev = {}
        if self.save_genome_path.exists():
            try:
                prev = json.loads(self.save_genome_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass

        prev_w_wr = prev.get("bench_white_win_rate", 0.0)
        prev_b_wr = prev.get("bench_black_win_rate", 0.0)

        w_best_idx = max(range(len(white_fitness)), key=lambda i: white_fitness[i])
        b_best_idx = max(range(len(black_fitness)), key=lambda i: black_fitness[i])

        updated = False
        if bench_w_wr > prev_w_wr:
            prev["white"] = white_pop[w_best_idx]
            prev["bench_white_win_rate"] = bench_w_wr
            print(f"  White seed updated: bench_wr={bench_w_wr:.3f} (was {prev_w_wr:.3f})")
            updated = True
        else:
            print(f"  White seed kept: bench_wr={bench_w_wr:.3f} <= existing {prev_w_wr:.3f}")

        if bench_b_wr > prev_b_wr:
            prev["black"] = black_pop[b_best_idx]
            prev["bench_black_win_rate"] = bench_b_wr
            print(f"  Black seed updated: bench_wr={bench_b_wr:.3f} (was {prev_b_wr:.3f})")
            updated = True
        else:
            print(f"  Black seed kept: bench_wr={bench_b_wr:.3f} <= existing {prev_b_wr:.3f}")

        # Merge current HoF with previously saved HoF, keep top pop_size by fitness
        for color_key, hof in [("white_hof", self.white_hof), ("black_hof", self.black_hof)]:
            # Current run's HoF: list of (fitness, genome_json)
            merged: dict[str, float] = {}
            for fitness, genome_json in hof:
                if genome_json not in merged or fitness > merged[genome_json]:
                    merged[genome_json] = fitness
            # Previously saved HoF has no fitness scores; use -inf so
            # current-run genomes win ties but old ones fill remaining slots
            for genome_json in prev.get(color_key, []):
                if genome_json not in merged:
                    merged[genome_json] = float("-inf")
            # Sort by fitness descending, keep enough to seed a full population
            ranked = sorted(merged.items(), key=lambda x: -x[1])[:self.pop_size]
            prev[color_key] = [genome for genome, _ in ranked]
        updated = True  # HoF update always triggers save

        if updated:
            prev["bench_avg_win_rate"] = (
                prev.get("bench_white_win_rate", 0.0) + prev.get("bench_black_win_rate", 0.0)
            ) / 2
            try:
                self.save_genome_path.write_text(json.dumps(prev))
            except OSError as e:
                print(f"  Warning: could not save best genomes: {e}")

    def _init_benchmark(self) -> list[str]:
        """Create a fixed random NEAT population for absolute progress measurement."""
        bench_config = dict(self.neat_config)
        bench_config["population_size"] = self.benchmark_size
        config_json = json.dumps(bench_config)
        return neat_ga.create_population(config_json)

    def _init_random_opponents(self, count: int) -> list[str]:
        """Create a fixed pool of random NEAT genomes for curriculum training."""
        opp_config = dict(self.neat_config)
        opp_config["population_size"] = count
        config_json = json.dumps(opp_config)
        return neat_ga.create_population(config_json)

    def _generate_pairings(self) -> list[tuple[int, int]]:
        """Generate pairings: each white plays tournament_opponents random blacks."""
        pairings = []
        for w_idx in range(self.pop_size):
            opponents = random.sample(range(self.pop_size),
                                      min(self.tournament_opponents, self.pop_size))
            for b_idx in opponents:
                pairings.append((w_idx, b_idx))
        return pairings

    def _compute_fitness(
        self, results: list[dict], pop_size: int, color: int,
    ) -> list[float]:
        """Compute fitness for each individual of the given color."""
        return compute_fitness(results, pop_size, color, self.fitness_weights)

    def _compute_fitness_breakdown(
        self, results: list[dict], color: int,
    ) -> dict[str, float]:
        """Compute average contribution of each fitness component across all games."""
        return compute_fitness_breakdown(results, color, self.fitness_weights)

    def _compute_outcome_rates(
        self, results: list[dict], color: int,
    ) -> tuple[float, float, float]:
        """Compute win/draw/loss rates for a color."""
        return compute_outcome_rates(results, color)

    def _compute_tournament_scores(
        self, results: list[dict], pop_size: int, color: int,
    ) -> list[float]:
        """Compute tournament scores: 1.0 for win, 0.5+material_bonus for draw, 0.0 for loss."""
        return compute_tournament_scores(results, pop_size, color)

    def _update_hof(
        self, hof: list[tuple[float, str]], population: list[str], fitness: list[float],
    ) -> list[tuple[float, str]]:
        """Update Hall of Fame with best individual from current generation."""
        best_idx = max(range(len(fitness)), key=lambda i: fitness[i])
        best_fit = fitness[best_idx]
        best_genome = population[best_idx]

        if len(hof) < self.hof_max_size or best_fit > hof[-1][0]:
            hof.append((best_fit, best_genome))
            hof.sort(key=lambda x: -x[0])
            if len(hof) > self.hof_max_size:
                hof.pop()

        return hof

    def _benchmark_vs_random(
        self, white_pop: list[str], black_pop: list[str],
        white_fitness: list[float], black_fitness: list[float],
    ) -> tuple[float, float, float, float]:
        """Test top individuals against fixed random benchmark.

        Returns (white_win_rate, white_material_adv, black_win_rate, black_material_adv).
        """
        n_test = min(10, self.pop_size)

        # O(n) heap-select instead of O(n log n) sort
        w_top_idx = heapq.nlargest(n_test, range(self.pop_size), key=lambda i: white_fitness[i])
        b_top_idx = heapq.nlargest(n_test, range(self.pop_size), key=lambda i: black_fitness[i])

        top_white = [white_pop[i] for i in w_top_idx]
        top_black = [black_pop[i] for i in b_top_idx]

        # White evolved vs random benchmark (as black)
        w_pairings = [(w, b) for w in range(n_test) for b in range(self.benchmark_size)]
        w_results = chess_cpu.simulate_neat_games_batch(
            top_white,
            self.benchmark_genomes,
            w_pairings,
            output_size=self.output_size,
            max_moves=self.max_moves,
            temperature=self.temperature,
        )
        w_wins = sum(1 for r in w_results if r["result"] == 1)
        w_mat = sum(r["white_material"] - r["black_material"] for r in w_results) / max(1, len(w_results))
        w_wr = w_wins / max(1, len(w_results))

        # Benchmark (as white) vs black evolved
        b_pairings = [(w, b) for w in range(self.benchmark_size) for b in range(n_test)]
        b_results = chess_cpu.simulate_neat_games_batch(
            self.benchmark_genomes,
            top_black,
            b_pairings,
            output_size=self.output_size,
            max_moves=self.max_moves,
            temperature=self.temperature,
        )
        b_wins = sum(1 for r in b_results if r["result"] == -1)
        b_mat = sum(r["black_material"] - r["white_material"] for r in b_results) / max(1, len(b_results))
        b_wr = b_wins / max(1, len(b_results))

        return w_wr, w_mat, b_wr, b_mat

    def _play_game_vs_stockfish(
        self, genome_json: str, genome_is_white: bool,
        engine, compute_cpl: bool = False,
    ) -> tuple[str, int, float]:
        """Play one game between a NEAT genome and Stockfish.

        Returns (result, move_count, avg_cpl) where result is "win", "draw",
        or "loss" from the genome's perspective. avg_cpl is the average
        centipawn loss per genome move (0.0 if compute_cpl is False).
        """
        import chess
        import chess.engine

        from lichess_bot import SparseNetwork, pick_move  # noqa: F811

        network = SparseNetwork(json.loads(genome_json))
        board = chess.Board()
        cpl_values: list[float] = []

        for _ in range(self.max_moves):
            if board.is_game_over():
                break

            is_genome_turn = (board.turn == chess.WHITE and genome_is_white) or (
                board.turn == chess.BLACK and not genome_is_white
            )

            if is_genome_turn:
                # Evaluate position before genome's move for CPL
                if compute_cpl:
                    try:
                        info_before = engine.analyse(
                            board, chess.engine.Limit(time=self.sf_move_time),
                        )
                        score_before = info_before["score"].pov(
                            chess.WHITE if genome_is_white else chess.BLACK,
                        )
                    except Exception:
                        score_before = None

                move = pick_move(network, board, self.output_size)
                board.push(move)

                # Evaluate position after genome's move for CPL
                if compute_cpl and score_before is not None:
                    try:
                        info_after = engine.analyse(
                            board, chess.engine.Limit(time=self.sf_move_time),
                        )
                        score_after = info_after["score"].pov(
                            chess.WHITE if genome_is_white else chess.BLACK,
                        )
                        cp_before = score_before.score(mate_score=10000)
                        cp_after = score_after.score(mate_score=10000)
                        if cp_before is not None and cp_after is not None:
                            # CPL = how much worse the position got (from genome's view)
                            cpl = max(0, cp_before - cp_after)
                            cpl_values.append(cpl)
                    except Exception:
                        pass
            else:
                result = engine.play(board, chess.engine.Limit(time=self.sf_move_time))
                board.push(result.move)

        move_count = len(board.move_stack)
        avg_cpl = sum(cpl_values) / max(1, len(cpl_values)) if cpl_values else 0.0
        outcome = board.outcome()
        if outcome is None:
            return "draw", move_count, avg_cpl
        if outcome.winner is None:
            return "draw", move_count, avg_cpl
        genome_won = (outcome.winner == chess.WHITE) == genome_is_white
        return ("win" if genome_won else "loss"), move_count, avg_cpl

    def _benchmark_vs_stockfish(
        self, white_pop: list[str], black_pop: list[str],
        white_fitness: list[float], black_fitness: list[float],
    ) -> tuple[float, float, float, float, float]:
        """Play best genomes against Stockfish.

        Returns (white_win_rate, black_win_rate, avg_game_length,
                 white_avg_cpl, black_avg_cpl).
        """
        if not self._stockfish_path:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        import chess.engine

        w_best_idx = max(range(len(white_fitness)), key=lambda i: white_fitness[i])
        b_best_idx = max(range(len(black_fitness)), key=lambda i: black_fitness[i])

        try:
            engine = chess.engine.SimpleEngine.popen_uci(self._stockfish_path)
            engine.configure({"Skill Level": self.sf_skill_level})
        except Exception as e:
            print(f"  ⚠ Could not start Stockfish: {e}")
            return 0.0, 0.0, 0.0, 0.0, 0.0

        try:
            w_wins = 0
            b_wins = 0
            total_moves = 0
            w_cpls: list[float] = []
            b_cpls: list[float] = []
            n = self.sf_bench_games

            # Best white genome plays as white vs Stockfish
            for _ in range(n):
                result, moves, cpl = self._play_game_vs_stockfish(
                    white_pop[w_best_idx], genome_is_white=True,
                    engine=engine, compute_cpl=True,
                )
                if result == "win":
                    w_wins += 1
                total_moves += moves
                w_cpls.append(cpl)

            # Best black genome plays as black vs Stockfish
            for _ in range(n):
                result, moves, cpl = self._play_game_vs_stockfish(
                    black_pop[b_best_idx], genome_is_white=False,
                    engine=engine, compute_cpl=True,
                )
                if result == "win":
                    b_wins += 1
                total_moves += moves
                b_cpls.append(cpl)

            w_wr = w_wins / n
            b_wr = b_wins / n
            avg_len = total_moves / (2 * n)
            w_avg_cpl = sum(w_cpls) / max(1, len(w_cpls))
            b_avg_cpl = sum(b_cpls) / max(1, len(b_cpls))
            return w_wr, b_wr, avg_len, w_avg_cpl, b_avg_cpl
        finally:
            engine.quit()

    def _compute_sf_fitness(
        self, population: list[str], coevo_fitness: list[float],
        color: int,
    ) -> list[float]:
        """Compute Stockfish-based fitness for top N genomes.

        Plays each top genome 1 fast game vs Stockfish (no CPL analysis).
        Fitness based on outcome + survival length (how many moves before losing).
        Non-tested genomes inherit the median SF fitness to avoid penalizing them.
        """
        if not self._stockfish_path:
            return [0.0] * len(population)

        import chess.engine

        n = len(population)
        genome_is_white = color == 0

        # Test top N by coevolution fitness (heap-select, O(n) vs O(n log n) sort)
        top_n = min(self.sf_fitness_top_n, n)
        top_indices = heapq.nlargest(top_n, range(n), key=lambda i: coevo_fitness[i])

        try:
            engine = chess.engine.SimpleEngine.popen_uci(self._stockfish_path)
            engine.configure({"Skill Level": self.sf_skill_level})
        except Exception as e:
            print(f"  ⚠ SF fitness: could not start Stockfish: {e}")
            return [0.0] * n

        tested_scores: list[float] = []
        sf_fit = [0.0] * n
        try:
            for idx in top_indices:
                result, moves, cpl = self._play_game_vs_stockfish(
                    population[idx], genome_is_white=genome_is_white,
                    engine=engine, compute_cpl=True,
                )
                # CPL-based fitness: lower CPL = better play
                # Map CPL to 0-10 range: CPL=0 → 10, CPL>=2000 → 0
                cpl_score = max(0.0, 10.0 * (1.0 - cpl / 2000.0)) if cpl > 0 else 0.0
                # Outcome bonus
                outcome = 5.0 if result == "win" else (2.0 if result == "draw" else 0.0)
                sf_fit[idx] = cpl_score + outcome
                tested_scores.append(sf_fit[idx])
        finally:
            engine.quit()

        # Non-tested genomes get median score so they aren't penalized
        if tested_scores:
            tested_scores.sort()
            median = tested_scores[len(tested_scores) // 2]
            tested_set = set(top_indices)
            for i in range(n):
                if i not in tested_set:
                    sf_fit[i] = median

        return sf_fit

    def _benchmark_fitness_all(
        self, population: list[str], color: int,
    ) -> list[float]:
        """Compute per-individual fitness from games against benchmark opponents.

        Opponents = fixed random benchmark + Hall of Fame from the opposing color.
        As HoF accumulates stronger genomes, the benchmark escalates automatically.
        """
        # Build opponent pool: random benchmark + opposing HoF
        opponents = list(self.benchmark_genomes)
        if color == 0:
            # white individuals need black opponents → use black HoF
            opponents.extend(genome for _, genome in self.black_hof)
        else:
            # black individuals need white opponents → use white HoF
            opponents.extend(genome for _, genome in self.white_hof)
        num_opp = len(opponents)

        n = len(population)
        if color == 0:
            pairings = [(w, b) for w in range(n) for b in range(num_opp)]
        else:
            pairings = [(w, b) for w in range(num_opp) for b in range(n)]

        try:
            if color == 0:
                results = chess_cpu.simulate_neat_games_batch(
                    population, opponents, pairings,
                    output_size=self.output_size, max_moves=self.max_moves,
                    temperature=self.temperature,
                    mercy_min_moves=self.mercy_min_moves,
                    mercy_material_threshold=self.mercy_material_threshold,
                )
            else:
                results = chess_cpu.simulate_neat_games_batch(
                    opponents, population, pairings,
                    output_size=self.output_size, max_moves=self.max_moves,
                    temperature=self.temperature,
                    mercy_min_moves=self.mercy_min_moves,
                    mercy_material_threshold=self.mercy_material_threshold,
                )
        except Exception as e:
            print(f"  ⚠ Rust benchmark panic, using fallback draws: {e}")
            results = [
                {
                    "white_idx": w, "black_idx": b, "result": 2,
                    "move_count": 0, "white_material": 0.0,
                    "black_material": 0.0, "white_mobility": 0,
                    "black_mobility": 0, "white_king_safety": 0.0,
                    "black_king_safety": 0.0,
                    "white_king_danger": 0.0, "black_king_danger": 0.0,
                }
                for w, b in pairings
            ]

        return self._compute_fitness(results, n, color=color)

    def _genome_stats(self, population: list[str]) -> tuple[float, float]:
        """Compute average connections and nodes across population."""
        total_conns = 0
        total_nodes = 0
        for genome_json in population:
            g = json.loads(genome_json)
            total_conns += sum(1 for c in g["connections"] if c["enabled"])
            total_nodes += len(g["nodes"])
        n = max(1, len(population))
        return total_conns / n, total_nodes / n

    def train(
        self,
        max_generations: int,
        on_generation: Callable[[dict], None] | None = None,
    ) -> dict:
        """Run the NEAT training loop.

        Args:
            max_generations: Number of generations to train.
            on_generation: Callback called after each generation with metrics dict.

        Returns:
            Final metrics dict from the last generation.
        """
        config_json = json.dumps(self.neat_config)

        # Initialize populations via Rust (seeded or random)
        seed = self._load_seed()

        # Check seed output_size compatibility: skip seeds if mismatched
        if seed:
            check_key = "white" if "white" in seed else None
            if check_key:
                try:
                    g = json.loads(seed[check_key])
                    seed_outputs = sum(1 for n in g["nodes"] if n["node_type"] == 2)
                    if seed_outputs != self.output_size:
                        print(f"  ⚠ Seed output_size={seed_outputs} != config={self.output_size}; ignoring seeds")
                        seed = None
                except (json.JSONDecodeError, KeyError):
                    pass

        white_hof_seeds = (seed or {}).get("white_hof", [])
        black_hof_seeds = (seed or {}).get("black_hof", [])

        if white_hof_seeds:
            print(f"  Seeding white from {len(white_hof_seeds)} HoF genomes")
            init_result = neat_ga.create_multi_seeded_population(config_json, white_hof_seeds)
            white_pop: list[str] = init_result["population"]
            white_tracker_json: str = init_result["tracker"]
        elif seed and "white" in seed:
            print("  Seeding white from best genome")
            init_result = neat_ga.create_seeded_population(config_json, seed["white"])
            white_pop: list[str] = init_result["population"]
            white_tracker_json: str = init_result["tracker"]
        else:
            init_result = neat_ga.create_population_with_tracker(config_json)
            white_pop: list[str] = init_result["population"]
            white_tracker_json: str = init_result["tracker"]

        if black_hof_seeds:
            print(f"  Seeding black from {len(black_hof_seeds)} HoF genomes")
            init_result = neat_ga.create_multi_seeded_population(config_json, black_hof_seeds)
            black_pop: list[str] = init_result["population"]
            black_tracker_json: str = init_result["tracker"]
        elif seed and "black" in seed:
            print("  Seeding black from best genome")
            init_result = neat_ga.create_seeded_population(config_json, seed["black"])
            black_pop: list[str] = init_result["population"]
            black_tracker_json: str = init_result["tracker"]
        else:
            init_result = neat_ga.create_population_with_tracker(config_json)
            black_pop: list[str] = init_result["population"]
            black_tracker_json: str = init_result["tracker"]

        white_species_json = "[]"
        black_species_json = "[]"
        white_config_json = config_json
        black_config_json = config_json

        # Curriculum stage 1: create fixed random opponent pools
        use_curriculum = self.curriculum_stage >= 1
        if use_curriculum:
            n_opp = self.curriculum_random_opponents
            curriculum_black_opp = self._init_random_opponents(n_opp)
            curriculum_white_opp = self._init_random_opponents(n_opp)
            print(f"  Curriculum stage {self.curriculum_stage}: {n_opp} random opponents per color")

        last_metrics = {}
        total_games = 0

        for gen in range(1, max_generations + 1):
            gen_start = time.time()

            if use_curriculum:
                # Stage 1: each individual plays all random opponents
                # White pop (as white) vs random black opponents
                w_pairings = [(w, b) for w in range(self.pop_size)
                              for b in range(len(curriculum_black_opp))]
                w_results = chess_cpu.simulate_neat_games_batch(
                    white_pop, curriculum_black_opp, w_pairings,
                    output_size=self.output_size, max_moves=self.max_moves,
                    temperature=self.temperature,
                    mercy_min_moves=self.mercy_min_moves,
                    mercy_material_threshold=self.mercy_material_threshold,
                )
                # Random white opponents vs black pop (as black)
                b_pairings = [(w, b) for w in range(len(curriculum_white_opp))
                              for b in range(self.pop_size)]
                b_results = chess_cpu.simulate_neat_games_batch(
                    curriculum_white_opp, black_pop, b_pairings,
                    output_size=self.output_size, max_moves=self.max_moves,
                    temperature=self.temperature,
                    mercy_min_moves=self.mercy_min_moves,
                    mercy_material_threshold=self.mercy_material_threshold,
                )

                white_fitness = self._compute_fitness(w_results, self.pop_size, color=0)
                black_fitness = self._compute_fitness(b_results, self.pop_size, color=1)

                # Combine results for metrics reporting
                results = w_results + b_results
                num_games = len(results)
            else:
                # Standard coevolution
                pairings = self._generate_pairings()

                try:
                    results = chess_cpu.simulate_neat_games_batch(
                        white_pop,
                        black_pop,
                        pairings,
                        output_size=self.output_size,
                        max_moves=self.max_moves,
                        temperature=self.temperature,
                        mercy_min_moves=self.mercy_min_moves,
                        mercy_material_threshold=self.mercy_material_threshold,
                    )
                except Exception as e:
                    print(f"  ⚠ Rust batch panic at gen {gen}, using fallback draws: {e}")
                    results = [
                        {
                            "white_idx": w, "black_idx": b, "result": 2,
                            "move_count": 0, "white_material": 0.0,
                            "black_material": 0.0, "white_mobility": 0,
                            "black_mobility": 0, "white_king_safety": 0.0,
                            "black_king_safety": 0.0,
                        }
                        for w, b in pairings
                    ]

                num_games = len(results)

                # Compute coevolution fitness
                white_fitness = self._compute_fitness(results, self.pop_size, color=0)
                black_fitness = self._compute_fitness(results, self.pop_size, color=1)

                # Blend benchmark fitness into selection signal
                bw = self.benchmark_fitness_weight
                if bw > 0:
                    w_bench_fit = self._benchmark_fitness_all(white_pop, color=0)
                    b_bench_fit = self._benchmark_fitness_all(black_pop, color=1)
                    white_fitness = blend_fitness(white_fitness, w_bench_fit, bw)
                    black_fitness = blend_fitness(black_fitness, b_bench_fit, bw)

            gen_time = time.time() - gen_start
            total_games += num_games

            # Stockfish CPL-based fitness signal (every N generations)
            sf_w_avg_cpl_fit, sf_b_avg_cpl_fit = 0.0, 0.0
            sf_w = self.sf_fitness_weight
            if sf_w > 0 and self._stockfish_path and gen % self.sf_fitness_interval == 0:
                sf_white_fit = self._compute_sf_fitness(white_pop, white_fitness, color=0)
                sf_black_fit = self._compute_sf_fitness(black_pop, black_fitness, color=1)
                white_fitness = blend_fitness(white_fitness, sf_white_fit, sf_w)
                black_fitness = blend_fitness(black_fitness, sf_black_fit, sf_w)
                sf_w_avg_cpl_fit = sum(sf_white_fit) / max(1, sum(1 for f in sf_white_fit if f > 0))
                sf_b_avg_cpl_fit = sum(sf_black_fit) / max(1, sum(1 for f in sf_black_fit if f > 0))

            # Parsimony pressure: penalize complexity (enabled connections).
            # Use fast string count instead of full JSON parse per genome.
            # Rust serde produces "enabled":true (compact), Python json.dumps
            # produces "enabled": true (with space). Count both.
            cc = self.neat_config.get("complexity_cost", 0.0)
            if cc > 0:
                for i, gj in enumerate(white_pop):
                    white_fitness[i] -= cc * (gj.count('"enabled":true') + gj.count('"enabled": true'))
                for i, gj in enumerate(black_pop):
                    black_fitness[i] -= cc * (gj.count('"enabled":true') + gj.count('"enabled": true'))

            # Tournament scores
            if use_curriculum:
                white_tourn = self._compute_tournament_scores(w_results, self.pop_size, color=0)
                black_tourn = self._compute_tournament_scores(b_results, self.pop_size, color=1)
            else:
                white_tourn = self._compute_tournament_scores(results, self.pop_size, color=0)
                black_tourn = self._compute_tournament_scores(results, self.pop_size, color=1)

            # Update Hall of Fame
            self.white_hof = self._update_hof(self.white_hof, white_pop, white_fitness)
            self.black_hof = self._update_hof(self.black_hof, black_pop, black_fitness)

            # Evolve via Rust NEAT
            w_result = neat_ga.evolve_neat_generation(
                white_pop, white_fitness,
                white_species_json, white_config_json, white_tracker_json,
            )
            white_pop = w_result["population"]
            white_species_json = w_result["species"]
            white_tracker_json = w_result["tracker"]
            white_config_json = w_result["config"]
            w_stats = w_result["stats"]

            b_result = neat_ga.evolve_neat_generation(
                black_pop, black_fitness,
                black_species_json, black_config_json, black_tracker_json,
            )
            black_pop = b_result["population"]
            black_species_json = b_result["species"]
            black_tracker_json = b_result["tracker"]
            black_config_json = b_result["config"]
            b_stats = b_result["stats"]

            # Outcome rates
            if use_curriculum:
                w_win, w_draw, w_loss = self._compute_outcome_rates(w_results, color=0)
                b_win, b_draw, b_loss = self._compute_outcome_rates(b_results, color=1)
            else:
                w_win, w_draw, w_loss = self._compute_outcome_rates(results, color=0)
                b_win, b_draw, b_loss = self._compute_outcome_rates(results, color=1)

            # Aggregate game stats in a single pass
            total_moves, w_mat_avg, b_mat_avg = aggregate_game_stats(results)
            avg_game_length = total_moves / max(1, num_games)

            # Games/moves per second
            games_per_sec = num_games / max(0.001, gen_time)
            moves_per_sec = total_moves / max(0.001, gen_time)

            # Genome complexity stats
            avg_conns = (w_stats["avg_connections"] + b_stats["avg_connections"]) / 2
            avg_nodes = (w_stats["avg_nodes"] + b_stats["avg_nodes"]) / 2

            # Fitness component breakdowns
            if use_curriculum:
                w_breakdown = self._compute_fitness_breakdown(w_results, color=0)
                b_breakdown = self._compute_fitness_breakdown(b_results, color=1)
            else:
                w_breakdown = self._compute_fitness_breakdown(results, color=0)
                b_breakdown = self._compute_fitness_breakdown(results, color=1)

            # Benchmark vs random
            w_bench_wr, w_bench_mat, b_bench_wr, b_bench_mat = self._benchmark_vs_random(
                white_pop, black_pop, white_fitness, black_fitness,
            )

            # Benchmark vs Stockfish (every N generations)
            sf_w_wr, sf_b_wr, sf_avg_len = 0.0, 0.0, 0.0
            sf_w_cpl, sf_b_cpl = 0.0, 0.0
            if self._stockfish_path and self.sf_bench_interval > 0 and gen % self.sf_bench_interval == 0:
                sf_w_wr, sf_b_wr, sf_avg_len, sf_w_cpl, sf_b_cpl = self._benchmark_vs_stockfish(
                    white_pop, black_pop, white_fitness, black_fitness,
                )
                print(f"  Stockfish: w_wr={sf_w_wr:.2f} b_wr={sf_b_wr:.2f} avg_len={sf_avg_len:.0f} w_cpl={sf_w_cpl:.0f} b_cpl={sf_b_cpl:.0f}")

            # Cache fitness extremes to avoid redundant recomputation
            w_best = max(white_fitness)
            b_best = max(black_fitness)

            # Build metrics dict matching CHESS_LOG_KEYS
            metrics = {
                "generation": gen,
                # Fitness
                "white_best": w_best,
                "white_avg": sum(white_fitness) / len(white_fitness),
                "black_best": b_best,
                "black_avg": sum(black_fitness) / len(black_fitness),
                "combined_best": min(w_best, b_best),
                # Games
                "total_games_this_gen": num_games,
                "avg_game_length": avg_game_length,
                "games_per_sec": games_per_sec,
                "moves_per_sec": moves_per_sec,
                "generation_time_sec": gen_time,
                # Outcome rates
                "white_win_rate": w_win,
                "white_draw_rate": w_draw,
                "black_win_rate": b_win,
                "black_draw_rate": b_draw,
                # Tournament scores
                "white_tournament_score_best": max(white_tourn) if white_tourn else 0,
                "white_tournament_score_avg": sum(white_tourn) / max(1, len(white_tourn)),
                "black_tournament_score_best": max(black_tourn) if black_tourn else 0,
                "black_tournament_score_avg": sum(black_tourn) / max(1, len(black_tourn)),
                # Material
                "white_material_avg": w_mat_avg,
                "black_material_avg": b_mat_avg,
                # Hall of Fame
                "white_hof_size": len(self.white_hof),
                "black_hof_size": len(self.black_hof),
                # NEAT topology metrics (per-color)
                "white_species_count": w_stats["species_count"],
                "black_species_count": b_stats["species_count"],
                "white_depth_avg": w_stats.get("avg_depth", 0),
                "black_depth_avg": b_stats.get("avg_depth", 0),
                "white_width_avg": w_stats.get("avg_width", 0),
                "black_width_avg": b_stats.get("avg_width", 0),
                "white_connections_avg": w_stats["avg_connections"],
                "black_connections_avg": b_stats["avg_connections"],
                "white_hidden_nodes_avg": w_stats["avg_nodes"] - self.input_size - self.output_size,
                "black_hidden_nodes_avg": b_stats["avg_nodes"] - self.input_size - self.output_size,
                # Fitness component breakdowns
                "white_fitness_outcome": w_breakdown["outcome"],
                "white_fitness_material": w_breakdown["material"],
                "white_fitness_mobility": w_breakdown["mobility"],
                "white_fitness_king_safety": w_breakdown["king_safety"],
                "white_fitness_opp_king_safety": w_breakdown["opp_king_safety"],
                "white_fitness_king_danger": w_breakdown["king_danger"],
                "white_fitness_captures": w_breakdown["captures"],
                "white_fitness_move_penalty": w_breakdown["move_penalty"],
                "black_fitness_outcome": b_breakdown["outcome"],
                "black_fitness_material": b_breakdown["material"],
                "black_fitness_mobility": b_breakdown["mobility"],
                "black_fitness_king_safety": b_breakdown["king_safety"],
                "black_fitness_opp_king_safety": b_breakdown["opp_king_safety"],
                "black_fitness_king_danger": b_breakdown["king_danger"],
                "black_fitness_captures": b_breakdown["captures"],
                "black_fitness_move_penalty": b_breakdown["move_penalty"],
                # King danger averages
                "white_king_danger_avg": sum(g.get("white_king_danger", 0.0) for g in results) / max(1, num_games),
                "black_king_danger_avg": sum(g.get("black_king_danger", 0.0) for g in results) / max(1, num_games),
                # Benchmark vs random (absolute progress)
                "bench_white_win_rate": w_bench_wr,
                "bench_white_material_adv": w_bench_mat,
                "bench_black_win_rate": b_bench_wr,
                "bench_black_material_adv": b_bench_mat,
                "bench_avg_win_rate": (w_bench_wr + b_bench_wr) / 2,
                "curriculum_stage": self.curriculum_stage,
            }

            # Curriculum promotion check
            if use_curriculum:
                bench_wr = (w_bench_wr + b_bench_wr) / 2
                if bench_wr >= self.curriculum_promotion_threshold:
                    print(f"  🎓 Curriculum promotion! bench_avg_win_rate={bench_wr:.1%} "
                          f">= {self.curriculum_promotion_threshold:.0%}")

            # Only include SF metrics on benchmark generations
            ran_sf = (self._stockfish_path
                      and self.sf_bench_interval > 0
                      and gen % self.sf_bench_interval == 0)
            if ran_sf:
                metrics.update({
                    "sf_white_win_rate": sf_w_wr,
                    "sf_black_win_rate": sf_b_wr,
                    "sf_avg_game_length": sf_avg_len,
                    "sf_white_avg_cpl": sf_w_cpl,
                    "sf_black_avg_cpl": sf_b_cpl,
                    "sf_fitness_white_avg": sf_w_avg_cpl_fit,
                    "sf_fitness_black_avg": sf_b_avg_cpl_fit,
                })

            # Write metrics line to file
            try:
                with open(self.metrics_path, "a") as f:
                    f.write(json.dumps(metrics) + "\n")
            except OSError:
                pass

            if on_generation is not None:
                on_generation(metrics)

            last_metrics = metrics

        # Save best genomes for seeding next run
        self._save_best(white_pop, black_pop, white_fitness, black_fitness,
                        last_metrics.get("bench_white_win_rate", 0),
                        last_metrics.get("bench_black_win_rate", 0))

        return last_metrics
