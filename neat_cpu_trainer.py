"""
NEAT CPU-based chess neuroevolution trainer using Rust backends.

Uses neat_ga for NEAT evolution (speciation, crossover, topology mutation)
and chess_cpu for parallel game simulation with sparse neural networks.
Drop-in replacement for CPUTrainer when use_neat=True.
"""
import json
import random
import time
from collections.abc import Callable
from pathlib import Path

import chess_cpu
import neat_ga


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

        # Fitness weights (same as cpu_trainer.py)
        self.win_bonus = 10.0
        self.draw_bonus = 0.0
        self.loss_penalty = -5.0
        self.capture_weight = 0.5
        self.material_weight = 1.0
        self.mobility_weight = 0.3
        self.king_safety_weight = 0.5
        self.opp_king_safety_weight = 1.5
        self.king_danger_weight = 1.0
        self.move_count_penalty = -0.002
        self.checkmate_bonus = 10.0

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

        # Seed genome paths: if set, initialize population from saved best topology
        _seed = config.get("seed_genome_path", "")
        self.seed_genome_path = Path(_seed) if _seed else None
        self.save_genome_path = Path(config.get("save_genome_path", "neat_best_genomes.json"))

        # Fixed random benchmark population for absolute progress measurement.
        self.benchmark_size = 20
        self.benchmark_genomes = self._init_benchmark()

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
        """Compute fitness for each individual of the given color.

        Same formula as cpu_trainer.py.
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

            is_win = (result == 1 and color == 0) or (result == -1 and color == 1)
            is_loss = (result == -1 and color == 0) or (result == 1 and color == 1)

            if result == 2:
                mat_adv = max(-1.0, min(1.0, (my_material - opp_material) / 10.0))
                f += self.draw_bonus * (0.5 + 0.5 * mat_adv)
            elif is_win:
                f += self.win_bonus
                f += self.checkmate_bonus
            elif is_loss:
                f += self.loss_penalty

            f += (my_material - opp_material) * self.material_weight
            f += (my_mobility - opp_mobility) * self.mobility_weight
            f += my_king_safety * self.king_safety_weight
            f -= opp_king_safety * self.opp_king_safety_weight
            f += my_king_danger_inflicted * self.king_danger_weight
            f += my_captures_value * self.capture_weight
            f += move_count * self.move_count_penalty

            fitness[idx] += f
            game_counts[idx] += 1

        for i in range(pop_size):
            if game_counts[i] > 0:
                fitness[i] /= game_counts[i]

        return fitness

    def _compute_fitness_breakdown(
        self, results: list[dict], color: int,
    ) -> dict[str, float]:
        """Compute average contribution of each fitness component across all games."""
        totals = {
            "outcome": 0.0, "material": 0.0, "mobility": 0.0,
            "king_safety": 0.0, "opp_king_safety": 0.0,
            "king_danger": 0.0, "captures": 0.0, "move_penalty": 0.0,
        }
        n = len(results)
        if n == 0:
            return totals

        for game in results:
            if color == 0:
                my_mat = game["white_material"]
                opp_mat = game["black_material"]
                my_ks = game["white_king_safety"]
                opp_ks = game["black_king_safety"]
                my_mob = game["white_mobility"]
                opp_mob = game["black_mobility"]
            else:
                my_mat = game["black_material"]
                opp_mat = game["white_material"]
                my_ks = game["black_king_safety"]
                opp_ks = game["white_king_safety"]
                my_mob = game["black_mobility"]
                opp_mob = game["white_mobility"]

            result = game["result"]
            is_win = (result == 1 and color == 0) or (result == -1 and color == 1)
            is_loss = (result == -1 and color == 0) or (result == 1 and color == 1)

            if result == 2:
                mat_adv = max(-1.0, min(1.0, (my_mat - opp_mat) / 10.0))
                totals["outcome"] += self.draw_bonus * (0.5 + 0.5 * mat_adv)
            elif is_win:
                totals["outcome"] += self.win_bonus + self.checkmate_bonus
            elif is_loss:
                totals["outcome"] += self.loss_penalty

            totals["material"] += (my_mat - opp_mat) * self.material_weight
            totals["mobility"] += (my_mob - opp_mob) * self.mobility_weight
            totals["king_safety"] += my_ks * self.king_safety_weight
            totals["opp_king_safety"] -= opp_ks * self.opp_king_safety_weight
            if color == 0:
                totals["king_danger"] += game.get("white_king_danger", 0.0) * self.king_danger_weight
            else:
                totals["king_danger"] += game.get("black_king_danger", 0.0) * self.king_danger_weight
            if color == 0:
                totals["captures"] += game.get("white_captures_value", 0.0) * self.capture_weight
            else:
                totals["captures"] += game.get("black_captures_value", 0.0) * self.capture_weight
            totals["move_penalty"] += game["move_count"] * self.move_count_penalty

        return {k: v / n for k, v in totals.items()}

    def _compute_outcome_rates(
        self, results: list[dict], color: int,
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

    def _compute_tournament_scores(
        self, results: list[dict], pop_size: int, color: int,
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

            counts[idx] += 1

        for i in range(pop_size):
            if counts[i] > 0:
                scores[i] /= counts[i]

        return scores

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

        # Sort by fitness to get top individuals
        w_order = sorted(range(self.pop_size), key=lambda i: white_fitness[i], reverse=True)
        b_order = sorted(range(self.pop_size), key=lambda i: black_fitness[i], reverse=True)

        top_white = [white_pop[i] for i in w_order[:n_test]]
        top_black = [black_pop[i] for i in b_order[:n_test]]

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

        last_metrics = {}
        total_games = 0

        for gen in range(1, max_generations + 1):
            gen_start = time.time()

            # Generate pairings
            pairings = self._generate_pairings()

            # Simulate all games in parallel via Rust
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

            gen_time = time.time() - gen_start
            num_games = len(results)
            total_games += num_games

            # Compute coevolution fitness
            white_fitness = self._compute_fitness(results, self.pop_size, color=0)
            black_fitness = self._compute_fitness(results, self.pop_size, color=1)

            # Blend benchmark fitness into selection signal
            bw = self.benchmark_fitness_weight
            if bw > 0:
                w_bench_fit = self._benchmark_fitness_all(white_pop, color=0)
                b_bench_fit = self._benchmark_fitness_all(black_pop, color=1)
                white_fitness = [(1 - bw) * c + bw * b for c, b in zip(white_fitness, w_bench_fit, strict=True)]
                black_fitness = [(1 - bw) * c + bw * b for c, b in zip(black_fitness, b_bench_fit, strict=True)]

            # Parsimony pressure: penalize complexity (enabled connections)
            cc = self.neat_config.get("complexity_cost", 0.0)
            if cc > 0:
                for i, gj in enumerate(white_pop):
                    g = json.loads(gj)
                    n_conns = sum(1 for c in g["connections"] if c["enabled"])
                    white_fitness[i] -= cc * n_conns
                for i, gj in enumerate(black_pop):
                    g = json.loads(gj)
                    n_conns = sum(1 for c in g["connections"] if c["enabled"])
                    black_fitness[i] -= cc * n_conns

            # Tournament scores
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

            # Genome complexity stats
            avg_conns = (w_stats["avg_connections"] + b_stats["avg_connections"]) / 2
            avg_nodes = (w_stats["avg_nodes"] + b_stats["avg_nodes"]) / 2

            # Fitness component breakdowns
            w_breakdown = self._compute_fitness_breakdown(results, color=0)
            b_breakdown = self._compute_fitness_breakdown(results, color=1)

            # Benchmark vs random
            w_bench_wr, w_bench_mat, b_bench_wr, b_bench_mat = self._benchmark_vs_random(
                white_pop, black_pop, white_fitness, black_fitness,
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
                # Elo tracking placeholders
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
                # NEAT topology metrics (combined, backward-compatible)
                "neat_hidden_nodes_avg": avg_nodes - self.input_size - self.output_size,
                "neat_connections_avg": avg_conns,
                "neat_species_count": w_stats["species_count"] + b_stats["species_count"],
                # Per-color topology metrics
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
                # Benchmark vs random
                # King danger averages
                "white_king_danger_avg": sum(g.get("white_king_danger", 0.0) for g in results) / max(1, num_games),
                "black_king_danger_avg": sum(g.get("black_king_danger", 0.0) for g in results) / max(1, num_games),
                "bench_white_win_rate": w_bench_wr,
                "bench_white_material_adv": w_bench_mat,
                "bench_black_win_rate": b_bench_wr,
                "bench_black_material_adv": b_bench_mat,
            }

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
