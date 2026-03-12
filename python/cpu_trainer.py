"""
CPU-based chess neuroevolution trainer using Rust backends.

Uses chess_cpu for parallel game simulation and evolve_ga for genetic operators.
Drop-in replacement for Godot-based training, callable from train_wandb.py.
"""
import gc
import json
import random
import time
from collections.abc import Callable
from pathlib import Path

import chess_cpu
import evolve_ga
import numpy as np


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

        # Fitness weights — tuned to prioritize winning over material hoarding.
        # Win + checkmate bonus (20.0) dominates material signal (~10 for 10-pt lead).
        # King safety rewards defending own king AND attacking opponent's king.
        self.win_bonus = 10.0
        self.draw_bonus = config.get("draw_bonus", 3.0)
        self.loss_penalty = -5.0
        self.capture_weight = config.get("capture_weight", 0.2)
        self.material_weight = 1.0
        self.mobility_weight = 0.3
        self.king_safety_weight = 0.5
        self.opp_king_safety_weight = config.get("opp_king_safety_weight", 0.0)
        self.king_danger_weight = 1.0
        self.move_count_penalty = -0.002
        self.checkmate_bonus = 10.0

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
        self.benchmark_size = 20
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

    def _random_individual_np(self) -> np.ndarray:
        """Create a single random weight vector as numpy float32."""
        scale = np.float32((2.0 / self.input_size) ** 0.5)
        return np.random.randn(self.genome_size).astype(np.float32) * scale

    def _generate_pairings(self) -> list[tuple[int, int]]:
        """Generate round-robin pairings: each white plays tournament_opponents random blacks."""
        pairings = []
        for w_idx in range(self.pop_size):
            opponents = random.sample(range(self.pop_size), min(self.tournament_opponents, self.pop_size))
            for b_idx in opponents:
                pairings.append((w_idx, b_idx))
        return pairings

    def _compute_fitness(
        self,
        results: list[dict],
        pop_size: int,
        color: int,
    ) -> list[float]:
        """Compute fitness for each individual of the given color.

        Matches the formula in ai/fitness.gd:evaluate_from_metrics().
        """
        fitness = [0.0] * pop_size
        game_counts = [0] * pop_size

        for game in results:
            if color == 0:
                idx = game["white_idx"]
                my_material = game["white_material"]
                opp_material = game["black_material"]
                my_king_safety = game["white_king_safety"]
                opp_king_safety = game["black_king_safety"]
                my_mobility = game["white_mobility"]
                opp_mobility = game["black_mobility"]
                my_king_danger_inflicted = game.get("white_king_danger", 0.0)
                my_captures_value = game.get("white_captures_value", 0.0)
            else:
                idx = game["black_idx"]
                my_material = game["black_material"]
                opp_material = game["white_material"]
                my_king_safety = game["black_king_safety"]
                opp_king_safety = game["white_king_safety"]
                my_mobility = game["black_mobility"]
                opp_mobility = game["white_mobility"]
                my_king_danger_inflicted = game.get("black_king_danger", 0.0)
                my_captures_value = game.get("black_captures_value", 0.0)

            result = game["result"]
            move_count = game["move_count"]
            f = 0.0

            # Game result — winning is the primary objective
            is_win = (result == 1 and color == 0) or (result == -1 and color == 1)
            is_loss = (result == -1 and color == 0) or (result == 1 and color == 1)

            if result == 2:  # Draw
                mat_adv = max(-1.0, min(1.0, (my_material - opp_material) / 10.0))
                f += self.draw_bonus * (0.5 + 0.5 * mat_adv)
            elif is_win:
                f += self.win_bonus
                f += self.checkmate_bonus
            elif is_loss:
                f += self.loss_penalty

            # Material difference — secondary continuous signal
            f += (my_material - opp_material) * self.material_weight

            # Mobility advantage — rewards piece activity
            f += (my_mobility - opp_mobility) * self.mobility_weight

            # King safety — reward protecting own king
            f += my_king_safety * self.king_safety_weight

            # Opponent king safety — reward exposing opponent's king
            f -= opp_king_safety * self.opp_king_safety_weight

            # King danger — reward putting opponent's king in danger
            f += my_king_danger_inflicted * self.king_danger_weight

            # Capture reward — reward taking material
            f += my_captures_value * self.capture_weight

            # Move count penalty
            f += move_count * self.move_count_penalty

            fitness[idx] += f
            game_counts[idx] += 1

        # Average fitness over games played
        for i in range(pop_size):
            if game_counts[i] > 0:
                fitness[i] /= game_counts[i]

        return fitness

    def _compute_tournament_scores(
        self,
        results: list[dict],
        pop_size: int,
        color: int,
    ) -> list[float]:
        """Compute tournament scores: 1.0 for win, 0.5+material_bonus for draw, 0.0 for loss."""
        scores = [0.0] * pop_size
        counts = [0] * pop_size

        for game in results:
            if color == 0:
                idx = game["white_idx"]
                my_mat = game["white_material"]
                opp_mat = game["black_material"]
            else:
                idx = game["black_idx"]
                my_mat = game["black_material"]
                opp_mat = game["white_material"]

            result = game["result"]
            is_win = (result == 1 and color == 0) or (result == -1 and color == 1)

            if is_win:
                scores[idx] += 1.0
            elif result == 2:
                mat_bonus = max(-0.25, min(0.25, (my_mat - opp_mat) / 40.0))
                scores[idx] += 0.5 + mat_bonus
            # loss = 0.0

            counts[idx] += 1

        # Average
        for i in range(pop_size):
            if counts[i] > 0:
                scores[i] /= counts[i]

        return scores

    def _update_hof(
        self,
        hof: list[tuple[float, np.ndarray]],
        population: np.ndarray,
        fitness: list[float],
    ) -> list[tuple[float, np.ndarray]]:
        """Update Hall of Fame with best individuals from current generation."""
        best_idx = max(range(len(fitness)), key=lambda i: fitness[i])
        best_fit = fitness[best_idx]
        best_genome = population[best_idx].copy()

        # Add if HoF not full or better than worst in HoF
        if len(hof) < self.hof_max_size or best_fit > hof[-1][0]:
            hof.append((best_fit, best_genome))
            hof.sort(key=lambda x: -x[0])
            if len(hof) > self.hof_max_size:
                hof.pop()

        return hof

    def _apply_immigration(self, population: np.ndarray) -> np.ndarray:
        """Replace a fraction of the population with random individuals."""
        n_immigrants = max(1, int(self.pop_size * self.immigration_rate))
        # Don't replace elites (first elite_count individuals after evolution)
        replaceable = list(range(self.elite_count, self.pop_size))
        if len(replaceable) <= n_immigrants:
            return population
        targets = random.sample(replaceable, n_immigrants)
        scale = np.float32((2.0 / self.input_size) ** 0.5)
        for idx in targets:
            population[idx] = np.random.randn(self.genome_size).astype(np.float32) * scale
        return population

    def _compute_outcome_rates(
        self, results: list[dict], color: int
    ) -> tuple[float, float, float]:
        """Compute win/draw/loss rates for a color."""
        wins = draws = losses = 0
        for game in results:
            r = game["result"]
            is_win = (r == 1 and color == 0) or (r == -1 and color == 1)
            is_loss = (r == -1 and color == 0) or (r == 1 and color == 1)
            if is_win:
                wins += 1
            elif r == 2:
                draws += 1
            elif is_loss:
                losses += 1
        total = max(1, wins + draws + losses)
        return wins / total, draws / total, losses / total

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
        gc.collect()
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

            # Average game length
            total_moves = sum(g["move_count"] for g in results)
            avg_game_length = total_moves / max(1, num_games)

            # Material averages
            w_mat_avg = sum(g["white_material"] for g in results) / max(1, num_games)
            b_mat_avg = sum(g["black_material"] for g in results) / max(1, num_games)

            # Games/moves per second
            games_per_sec = num_games / max(0.001, gen_time)
            moves_per_sec = total_moves / max(0.001, gen_time)

            # Benchmark vs fixed random opponents (absolute progress measure).
            # Sort populations by fitness so top individuals are first.
            w_order = sorted(range(self.pop_size), key=lambda i: white_fitness[i], reverse=True)
            b_order = sorted(range(self.pop_size), key=lambda i: black_fitness[i], reverse=True)
            w_bench_wr, w_bench_mat, b_bench_wr, b_bench_mat = self._benchmark_vs_random(
                white_pop[w_order], black_pop[b_order],
            )

            # Build metrics dict matching CHESS_LOG_KEYS
            metrics = {
                "generation": gen,
                "white_best": max(white_fitness),
                "white_avg": sum(white_fitness) / len(white_fitness),
                "black_best": max(black_fitness),
                "black_avg": sum(black_fitness) / len(black_fitness),
                "best_fitness": max(max(white_fitness), max(black_fitness)),
                "avg_fitness": (sum(white_fitness) + sum(black_fitness)) / (2 * self.pop_size),
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
                "combined_best": min(max(white_fitness), max(black_fitness)),
                # Elo tracking placeholders (not implemented in CPU trainer)
                "white_hof_avg_elo": 0,
                "black_hof_avg_elo": 0,
                "white_hof_top_elo": 0,
                "black_hof_top_elo": 0,
                "white_elo_min": 0,
                "white_elo_p25": 0,
                "white_elo_median": 0,
                "white_elo_p75": 0,
                "white_elo_max": 0,
                "black_elo_min": 0,
                "black_elo_p25": 0,
                "black_elo_median": 0,
                "black_elo_p75": 0,
                "black_elo_max": 0,
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

            # Write metrics line to file
            try:
                with open(self.metrics_path, "a") as f:
                    f.write(json.dumps(metrics) + "\n")
            except OSError:
                pass

            # Callback
            if on_generation is not None:
                on_generation(metrics)

            last_metrics = metrics

        return last_metrics
