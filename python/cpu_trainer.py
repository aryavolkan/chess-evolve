"""
CPU-based chess neuroevolution trainer using Rust backends.

Uses chess_cpu for parallel game simulation and evolve_ga for genetic operators.
Drop-in replacement for Godot-based training, callable from train_wandb.py.
"""
import heapq
import random
import time
from collections.abc import Callable
from pathlib import Path

import chess_cpu
import evolve_ga
import numpy as np

from fitness import (
    aggregate_game_stats,
    compute_fitness,
    compute_outcome_rates,
    compute_tournament_scores,
    merge_fitness_weights,
)
from trainer_utils import MetricsWriter, generate_pairings, update_hof


class CPUTrainer:
    """CPU-based training loop using Rust chess simulation + GA operators.

    Populations are stored as numpy float32 arrays (pop_size × genome_size)
    for memory efficiency (~4 bytes/weight vs ~28 bytes for Python floats).
    """

    def __init__(self, config: dict, metrics_path: Path):
        self.config = config
        self.metrics_path = Path(metrics_path)

        # Population parameters
        self.pop_size = config.get("population_size", 30)
        self.input_size = config.get("input_size", 389)
        self.hidden_size = config.get("hidden_size", 64)
        self.output_size = config.get("output_size", 4096)
        self.max_moves = config.get("max_moves_per_game", 200)
        self.temperature = config.get("move_temperature", 0.5)

        # GA parameters
        self.elite_count = config.get("elite_count", 2)
        self.crossover_rate = config.get("crossover_rate", 0.70)
        self.mutation_rate = config.get("mutation_rate", 0.25)
        self.mutation_strength = config.get("mutation_strength", 0.12)
        self.tournament_k = config.get("tournament_k", 2)
        self.immigration_rate = config.get("immigration_rate", 0.1)
        self.tournament_opponents = config.get("tournament_opponents", 5)

        # Speciation (weight-distance based, via evolve_ga).
        # Uses dynamic threshold adjustment to hit target species count,
        # matching NEAT's approach. Threshold adapts each generation.
        self.target_species = config.get("target_species", 20)
        self.use_fitness_sharing = config.get("use_fitness_sharing", True)

        # Mercy rule
        self.mercy_min_moves = config.get("mercy_min_moves", 30)
        self.mercy_material_threshold = config.get("mercy_material_threshold", 12.0)

        # Fitness weights (shared defaults from fitness.py, config-overridable)
        self.fitness_weights = merge_fitness_weights(config)

        # Network weight count
        ih = self.input_size * self.hidden_size
        bh = self.hidden_size
        ho = self.hidden_size * self.output_size
        bo = self.output_size
        self.genome_size = ih + bh + ho + bo

        # Initialize speciation threshold to the expected L2 distance between
        # random Xavier-initialized individuals: E[||a-b||] = 2*sqrt(N/I).
        # Dynamic adjustment each generation will tune this to hit target_species.
        expected_l2 = 2.0 * (self.genome_size / self.input_size) ** 0.5
        self.white_spec_threshold = expected_l2
        self.black_spec_threshold = expected_l2

        # Hall of Fame (stored as numpy arrays)
        self.white_hof: list[tuple[float, np.ndarray]] = []
        self.black_hof: list[tuple[float, np.ndarray]] = []
        self.hof_max_size = 10

        # Species representatives (persisted across generations)
        self.white_species_reps: list[list[float]] | None = None
        self.black_species_reps: list[list[float]] | None = None

        # Fixed random benchmark population for measuring absolute progress.
        # Without this, coevolutionary metrics hide improvement because both
        # sides improve simultaneously, keeping relative metrics flat.
        self.benchmark_size = config.get("benchmark_size", 20)
        self.benchmark_pop = self._init_benchmark()

    def _init_benchmark(self) -> np.ndarray:
        """Create a fixed random population for absolute progress measurement."""
        scale = np.float32((2.0 / self.input_size) ** 0.5)
        return np.random.randn(self.benchmark_size, self.genome_size).astype(np.float32) * scale

    def _benchmark_vs_random(
        self, white_pop: np.ndarray, black_pop: np.ndarray,
    ) -> tuple[float, float, float, float]:
        """Test top individuals from each population against fixed random benchmark.

        Returns (white_win_rate, white_material_adv, black_win_rate, black_material_adv).
        """
        # Use top 10 individuals (by recent fitness) vs all benchmark opponents
        n_test = min(10, self.pop_size)
        pairings = [(w, b) for w in range(n_test) for b in range(self.benchmark_size)]

        # White evolved vs random benchmark (as black)
        w_results = chess_cpu.simulate_games_batch(
            white_pop[:n_test].tobytes(),
            self.benchmark_pop.tobytes(),
            self.genome_size,
            pairings,
            input_size=self.input_size, hidden_size=self.hidden_size,
            output_size=self.output_size, max_moves=self.max_moves,
            temperature=self.temperature,
        )
        w_wins = sum(1 for r in w_results if r["result"] == 1)
        w_mat = sum(r["white_material"] - r["black_material"] for r in w_results) / max(1, len(w_results))
        w_wr = w_wins / max(1, len(w_results))

        # Benchmark (as white) vs black evolved
        b_pairings = [(w, b) for w in range(self.benchmark_size) for b in range(n_test)]
        b_results = chess_cpu.simulate_games_batch(
            self.benchmark_pop.tobytes(),
            black_pop[:n_test].tobytes(),
            self.genome_size,
            b_pairings,
            input_size=self.input_size, hidden_size=self.hidden_size,
            output_size=self.output_size, max_moves=self.max_moves,
            temperature=self.temperature,
        )
        b_wins = sum(1 for r in b_results if r["result"] == -1)
        b_mat = sum(r["black_material"] - r["white_material"] for r in b_results) / max(1, len(b_results))
        b_wr = b_wins / max(1, len(b_results))

        return w_wr, w_mat, b_wr, b_mat

    def _init_population(self) -> np.ndarray:
        """Initialize a population as numpy float32 array (pop_size × genome_size)."""
        scale = np.float32((2.0 / self.input_size) ** 0.5)  # Xavier init
        return np.random.randn(self.pop_size, self.genome_size).astype(np.float32) * scale

    def _generate_pairings(self) -> list[tuple[int, int]]:
        """Generate round-robin pairings: each white plays tournament_opponents random blacks."""
        return generate_pairings(self.pop_size, self.tournament_opponents)

    def _compute_fitness(
        self,
        results: list[dict],
        pop_size: int,
        color: int,
    ) -> list[float]:
        """Compute fitness for each individual of the given color."""
        return compute_fitness(results, pop_size, color, self.fitness_weights)

    def _compute_tournament_scores(
        self,
        results: list[dict],
        pop_size: int,
        color: int,
    ) -> list[float]:
        """Compute tournament scores: 1.0 for win, 0.5+material_bonus for draw, 0.0 for loss."""
        return compute_tournament_scores(results, pop_size, color)

    def _update_hof(
        self,
        hof: list[tuple[float, np.ndarray]],
        population: np.ndarray,
        fitness: list[float],
    ) -> list[tuple[float, np.ndarray]]:
        """Update Hall of Fame with best individuals from current generation."""
        return update_hof(hof, population, fitness, self.hof_max_size)

    def _apply_immigration(self, population: np.ndarray) -> np.ndarray:
        """Replace a fraction of the population with random individuals."""
        n_immigrants = max(1, int(self.pop_size * self.immigration_rate))
        # Don't replace elites (first elite_count individuals after evolution)
        replaceable = list(range(self.elite_count, self.pop_size))
        if len(replaceable) <= n_immigrants:
            return population
        targets = random.sample(replaceable, n_immigrants)
        scale = np.float32((2.0 / self.input_size) ** 0.5)
        # Batch allocation: one randn call instead of n_immigrants separate ones
        immigrants = np.random.randn(n_immigrants, self.genome_size).astype(np.float32) * scale
        population[targets] = immigrants
        return population

    def _compute_outcome_rates(
        self, results: list[dict], color: int,
    ) -> tuple[float, float, float]:
        """Compute win/draw/loss rates for a color."""
        return compute_outcome_rates(results, color)

    def _speciate_and_evolve(
        self,
        population: np.ndarray,
        fitness: list[float],
        species_reps: list[list[float]] | None,
        threshold: float,
    ) -> tuple[np.ndarray, int, list[list[float]], float]:
        """Assign species, apply fitness sharing, evolve.

        Returns (new_pop, species_count, new_reps, adjusted_threshold).
        Dynamically adjusts threshold to converge on target_species.
        """
        pop_list = population.tolist()

        # Speciation: assign individuals to species by weight-vector distance
        species_ids, new_reps = evolve_ga.assign_species(
            pop_list,
            threshold=threshold,
            representatives=species_reps,
        )

        # Prune empty species: only keep representatives that have members.
        active = set(species_ids)
        pruned_reps = [rep for sid, rep in enumerate(new_reps) if sid in active]
        species_count = len(pruned_reps)

        # Dynamic threshold adjustment (matching NEAT approach):
        # Proportional step: the farther from target, the bigger the adjustment.
        if species_count > self.target_species:
            ratio = species_count / max(1, self.target_species)
            threshold *= 1.0 + 0.2 * (ratio - 1.0)
        elif species_count < self.target_species:
            ratio = self.target_species / max(1, species_count)
            threshold *= 1.0 / (1.0 + 0.2 * (ratio - 1.0))
        threshold = max(1.0, threshold)  # floor

        # Optionally apply fitness sharing (divide by species size).
        # When disabled, still track species for metrics but use raw fitness.
        if self.use_fitness_sharing:
            selection_fit = evolve_ga.shared_fitness(fitness, species_ids, min_species_size=1)
        else:
            selection_fit = fitness

        new_pop_list = evolve_ga.evolve_generation(
            pop_list,
            selection_fit,
            elite_count=self.elite_count,
            mutation_rate=self.mutation_rate,
            mutation_strength=self.mutation_strength,
            crossover_rate=self.crossover_rate,
            tournament_k=self.tournament_k,
        )
        del pop_list
        result = np.array(new_pop_list, dtype=np.float32)
        del new_pop_list
        return result, species_count, pruned_reps, threshold

    def train(
        self,
        max_generations: int,
        on_generation: Callable[[dict], None] | None = None,
    ) -> dict:
        """Run the training loop.

        Args:
            max_generations: Number of generations to train.
            on_generation: Callback called after each generation with metrics dict.

        Returns:
            Final metrics dict from the last generation.
        """
        white_pop = self._init_population()
        black_pop = self._init_population()

        last_metrics = {}
        total_games = 0
        metrics_writer = MetricsWriter(self.metrics_path)

        for gen in range(1, max_generations + 1):
            gen_start = time.time()

            # Generate pairings
            pairings = self._generate_pairings()

            # Simulate all games in parallel via Rust.
            # Pass numpy arrays as bytes for memory efficiency.
            results = chess_cpu.simulate_games_batch(
                white_pop.tobytes(),
                black_pop.tobytes(),
                self.genome_size,
                pairings,
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                output_size=self.output_size,
                max_moves=self.max_moves,
                temperature=self.temperature,
                mercy_min_moves=self.mercy_min_moves,
                mercy_material_threshold=self.mercy_material_threshold,
            )

            gen_time = time.time() - gen_start
            num_games = len(results)
            total_games += num_games

            # Compute fitness
            white_fitness = self._compute_fitness(results, self.pop_size, color=0)
            black_fitness = self._compute_fitness(results, self.pop_size, color=1)

            # Tournament scores
            white_tourn = self._compute_tournament_scores(results, self.pop_size, color=0)
            black_tourn = self._compute_tournament_scores(results, self.pop_size, color=1)

            # Update Hall of Fame
            self.white_hof = self._update_hof(self.white_hof, white_pop, white_fitness)
            self.black_hof = self._update_hof(self.black_hof, black_pop, black_fitness)

            # Speciate + evolve populations via Rust GA operators.
            # Fitness sharing encourages diversity by penalizing crowded species.
            # Threshold adapts dynamically to hit target_species.
            white_pop, w_species_count, self.white_species_reps, self.white_spec_threshold = (
                self._speciate_and_evolve(
                    white_pop, white_fitness, self.white_species_reps,
                    self.white_spec_threshold,
                )
            )
            black_pop, b_species_count, self.black_species_reps, self.black_spec_threshold = (
                self._speciate_and_evolve(
                    black_pop, black_fitness, self.black_species_reps,
                    self.black_spec_threshold,
                )
            )

            # Immigration
            white_pop = self._apply_immigration(white_pop)
            black_pop = self._apply_immigration(black_pop)

            # Outcome rates
            w_win, w_draw, w_loss = self._compute_outcome_rates(results, color=0)
            b_win, b_draw, b_loss = self._compute_outcome_rates(results, color=1)

            # Aggregate game stats in a single pass
            total_moves, w_mat_avg, b_mat_avg = aggregate_game_stats(results)
            avg_game_length = total_moves / max(1, num_games)

            # Games/moves per second
            games_per_sec = num_games / max(0.001, gen_time)
            moves_per_sec = total_moves / max(0.001, gen_time)

            # Cache fitness extremes to avoid redundant recomputation
            w_best = max(white_fitness)
            b_best = max(black_fitness)
            w_sum = sum(white_fitness)
            b_sum = sum(black_fitness)

            # Benchmark vs fixed random opponents (absolute progress measure).
            # Select top individuals by fitness using heap (O(n) vs O(n log n) sort).
            w_top_indices = heapq.nlargest(min(10, self.pop_size), range(self.pop_size), key=lambda i: white_fitness[i])
            b_top_indices = heapq.nlargest(min(10, self.pop_size), range(self.pop_size), key=lambda i: black_fitness[i])
            w_bench_wr, w_bench_mat, b_bench_wr, b_bench_mat = self._benchmark_vs_random(
                white_pop[w_top_indices], black_pop[b_top_indices],
            )

            # Build metrics dict matching CHESS_LOG_KEYS
            metrics = {
                "generation": gen,
                "white_best": w_best,
                "white_avg": w_sum / len(white_fitness),
                "black_best": b_best,
                "black_avg": b_sum / len(black_fitness),
                "best_fitness": max(w_best, b_best),
                "avg_fitness": (w_sum + b_sum) / (2 * self.pop_size),
                "games_played": total_games,
                "total_games_this_gen": num_games,
                "avg_game_length": avg_game_length,
                "games_per_sec": games_per_sec,
                "moves_per_sec": moves_per_sec,
                "white_win_rate": w_win,
                "white_draw_rate": w_draw,
                "white_loss_rate": w_loss,
                "black_win_rate": b_win,
                "black_draw_rate": b_draw,
                "black_loss_rate": b_loss,
                "white_hof_size": len(self.white_hof),
                "black_hof_size": len(self.black_hof),
                "white_tournament_score_best": max(white_tourn) if white_tourn else 0,
                "white_tournament_score_avg": sum(white_tourn) / max(1, len(white_tourn)),
                "black_tournament_score_best": max(black_tourn) if black_tourn else 0,
                "black_tournament_score_avg": sum(black_tourn) / max(1, len(black_tourn)),
                "white_material_avg": w_mat_avg,
                "black_material_avg": b_mat_avg,
                "generation_time_sec": gen_time,
                "combined_best": min(w_best, b_best),
                # Benchmark vs random (absolute progress — NOT relative to opponent)
                "bench_white_win_rate": w_bench_wr,
                "bench_white_material_adv": w_bench_mat,
                "bench_black_win_rate": b_bench_wr,
                "bench_black_material_adv": b_bench_mat,
                # Species metrics (weight-distance speciation, not NEAT topology)
                "neat_hidden_nodes_avg": 0,  # fixed topology — no hidden node growth
                "neat_connections_avg": self.genome_size,  # fixed topology — all connections
                "neat_species_count": w_species_count + b_species_count,
            }

            metrics_writer.write(metrics)

            # Callback
            if on_generation is not None:
                on_generation(metrics)

            last_metrics = metrics

        metrics_writer.close()

        return last_metrics
