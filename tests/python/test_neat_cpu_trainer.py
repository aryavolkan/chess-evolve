"""Tests for NeatCPUTrainer pure-Python logic.

Mocks the Rust backends (chess_cpu, neat_ga) so we can test fitness
computation, seed loading/saving, genome stats, HoF management, and
outcome rates without compiled extensions.
"""
from __future__ import annotations

import json
import sys
import types
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Mock Rust extension modules before importing NeatCPUTrainer.
# Track injected mocks so we can clean them up afterwards to avoid
# polluting sys.modules for other test files.
# ---------------------------------------------------------------------------

_injected = []

_chess_cpu_mock = types.ModuleType("chess_cpu")
_chess_cpu_mock.simulate_neat_games_batch = MagicMock(return_value=[])
if "chess_cpu" not in sys.modules:
    sys.modules["chess_cpu"] = _chess_cpu_mock
    _injected.append("chess_cpu")

_neat_ga_mock = types.ModuleType("neat_ga")
_neat_ga_mock.create_population = MagicMock(return_value=["{}"] * 20)
_neat_ga_mock.create_population_with_tracker = MagicMock(
    return_value={"population": ["{}"] * 50, "tracker": "{}"}
)
_neat_ga_mock.create_seeded_population = MagicMock(
    return_value={"population": ["{}"] * 50, "tracker": "{}"}
)
_neat_ga_mock.create_multi_seeded_population = MagicMock(
    return_value={"population": ["{}"] * 50, "tracker": "{}"}
)
_neat_ga_mock.evolve_neat_generation = MagicMock(
    return_value={
        "population": ["{}"] * 50,
        "species": "[]",
        "tracker": "{}",
        "config": "{}",
        "stats": {"avg_connections": 10, "avg_nodes": 400, "species_count": 3, "avg_depth": 2, "avg_width": 5},
    }
)
if "neat_ga" not in sys.modules:
    sys.modules["neat_ga"] = _neat_ga_mock
    _injected.append("neat_ga")

if "evolve_ga" not in sys.modules:
    sys.modules["evolve_ga"] = types.ModuleType("evolve_ga")
    _injected.append("evolve_ga")

from neat_cpu_trainer import NeatCPUTrainer  # noqa: E402

# Clean up mocks so other test files can detect real modules as missing
for _mod in _injected:
    sys.modules.pop(_mod, None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_game_result(
    white_idx=0, black_idx=1, result=2, move_count=40,
    white_material=10.0, black_material=10.0,
    white_mobility=5, black_mobility=5,
    white_king_safety=1.0, black_king_safety=1.0,
    white_captures_value=0.0, black_captures_value=0.0,
):
    return {
        "white_idx": white_idx, "black_idx": black_idx,
        "result": result, "move_count": move_count,
        "white_material": white_material, "black_material": black_material,
        "white_mobility": white_mobility, "black_mobility": black_mobility,
        "white_king_safety": white_king_safety, "black_king_safety": black_king_safety,
        "white_captures_value": white_captures_value,
        "black_captures_value": black_captures_value,
    }


def _make_genome(n_connections=3, n_nodes=5):
    """Create a minimal NEAT genome JSON string."""
    return json.dumps({
        "nodes": [{"id": i, "type": "hidden"} for i in range(n_nodes)],
        "connections": [{"in": 0, "out": i + 1, "weight": 0.5, "enabled": True}
                        for i in range(n_connections)],
    })


@pytest.fixture
def trainer(tmp_path):
    config = {
        "population_size": 5,
        "input_size": 4,
        "output_size": 3,
        "tournament_opponents": 3,
    }
    return NeatCPUTrainer(config, tmp_path / "metrics.jsonl")


# ===========================================================================
# _compute_fitness (same formula as CPUTrainer)
# ===========================================================================

class TestComputeFitness:
    def test_white_win(self, trainer):
        results = [_make_game_result(result=1, white_material=20.0, black_material=5.0)]
        fitness = trainer._compute_fitness(results, pop_size=2, color=0)
        assert fitness[0] > 0

    def test_draw_material_advantage(self, trainer):
        adv = [_make_game_result(result=2, white_material=20.0, black_material=5.0)]
        disadv = [_make_game_result(result=2, white_material=5.0, black_material=20.0)]
        f_adv = trainer._compute_fitness(adv, pop_size=2, color=0)
        f_disadv = trainer._compute_fitness(disadv, pop_size=2, color=0)
        assert f_adv[0] > f_disadv[0]

    def test_no_games_yields_zero(self, trainer):
        fitness = trainer._compute_fitness([], pop_size=3, color=0)
        assert fitness == [0.0, 0.0, 0.0]

    def test_captures_increase_fitness(self, trainer):
        no_cap = [_make_game_result(result=2, white_captures_value=0.0)]
        cap = [_make_game_result(result=2, white_captures_value=9.0)]
        f_no = trainer._compute_fitness(no_cap, pop_size=2, color=0)
        f_cap = trainer._compute_fitness(cap, pop_size=2, color=0)
        assert f_cap[0] > f_no[0]

    def test_draw_bonus_default(self, trainer):
        assert trainer.fitness_weights["draw_bonus"] == 3.0

    def test_loss_penalty_value(self, trainer):
        assert trainer.fitness_weights["loss_penalty"] == -5.0


# ===========================================================================
# _compute_tournament_scores
# ===========================================================================

class TestComputeTournamentScores:
    def test_win_scores_one(self, trainer):
        results = [_make_game_result(result=1)]
        scores = trainer._compute_tournament_scores(results, pop_size=2, color=0)
        assert scores[0] == 1.0

    def test_loss_scores_zero(self, trainer):
        results = [_make_game_result(result=-1)]
        scores = trainer._compute_tournament_scores(results, pop_size=2, color=0)
        assert scores[0] == 0.0


# ===========================================================================
# _update_hof (uses JSON strings instead of numpy)
# ===========================================================================

class TestUpdateHoF:
    def test_adds_best_to_empty_hof(self, trainer):
        pop = ["genome_a", "genome_b"]
        fitness = [5.0, 10.0]
        hof = trainer._update_hof([], pop, fitness)
        assert len(hof) == 1
        assert hof[0] == (10.0, "genome_b")

    def test_hof_capped(self, trainer):
        trainer.hof_max_size = 2
        pop = ["g0", "g1", "g2"]
        hof = []
        for i in range(3):
            fitness = [0.0] * 3
            fitness[i] = float(i + 1) * 10
            hof = trainer._update_hof(hof, pop, fitness)
        assert len(hof) <= 2

    def test_sorted_descending(self, trainer):
        pop = ["g0", "g1"]
        hof = []
        hof = trainer._update_hof(hof, pop, [100.0, 50.0])
        hof = trainer._update_hof(hof, pop, [200.0, 150.0])
        assert hof[0][0] >= hof[-1][0]


# ===========================================================================
# _compute_outcome_rates
# ===========================================================================

class TestComputeOutcomeRates:
    def test_mixed(self, trainer):
        results = [
            _make_game_result(result=1),
            _make_game_result(result=2),
            _make_game_result(result=-1),
            _make_game_result(result=2),
        ]
        w, d, l = trainer._compute_outcome_rates(results, color=0)
        assert abs(w - 0.25) < 1e-6
        assert abs(d - 0.50) < 1e-6
        assert abs(l - 0.25) < 1e-6

    def test_empty_no_crash(self, trainer):
        w, d, l = trainer._compute_outcome_rates([], color=0)
        assert (w, d, l) == (0.0, 0.0, 0.0)


# ===========================================================================
# _load_seed / _save_best
# ===========================================================================

class TestSeedIO:
    def test_load_seed_returns_none_when_no_file(self, trainer):
        trainer.seed_genome_path = None
        assert trainer._load_seed() is None

    def test_load_seed_returns_none_for_missing_file(self, trainer, tmp_path):
        trainer.seed_genome_path = tmp_path / "nonexistent.json"
        assert trainer._load_seed() is None

    def test_load_seed_returns_data(self, trainer, tmp_path):
        seed_file = tmp_path / "seed.json"
        seed_data = {"white": "{}", "black": "{}"}
        seed_file.write_text(json.dumps(seed_data))
        trainer.seed_genome_path = seed_file
        result = trainer._load_seed()
        assert result["white"] == "{}"

    def test_load_seed_handles_invalid_json(self, trainer, tmp_path):
        seed_file = tmp_path / "bad_seed.json"
        seed_file.write_text("{invalid json")
        trainer.seed_genome_path = seed_file
        assert trainer._load_seed() is None

    def test_save_best_creates_file(self, trainer, tmp_path):
        trainer.save_genome_path = tmp_path / "best.json"
        trainer._save_best(
            white_pop=["w_genome"], black_pop=["b_genome"],
            white_fitness=[5.0], black_fitness=[3.0],
            bench_w_wr=0.5, bench_b_wr=0.4,
        )
        assert trainer.save_genome_path.exists()
        data = json.loads(trainer.save_genome_path.read_text())
        assert data["white"] == "w_genome"
        assert data["black"] == "b_genome"
        assert abs(data["bench_avg_win_rate"] - 0.45) < 1e-6

    def test_save_best_skips_genome_on_no_improvement(self, trainer, tmp_path):
        trainer.save_genome_path = tmp_path / "best.json"
        # Write an existing seed with high per-color win rates
        trainer.save_genome_path.write_text(json.dumps({
            "white": "old_w", "black": "old_b",
            "bench_white_win_rate": 0.9, "bench_black_win_rate": 0.9,
        }))
        trainer._save_best(
            white_pop=["new_w"], black_pop=["new_b"],
            white_fitness=[1.0], black_fitness=[1.0],
            bench_w_wr=0.1, bench_b_wr=0.1,
        )
        data = json.loads(trainer.save_genome_path.read_text())
        assert data["white"] == "old_w", "should not overwrite better white seed"
        assert data["black"] == "old_b", "should not overwrite better black seed"

    def test_save_best_overwrites_on_improvement(self, trainer, tmp_path):
        trainer.save_genome_path = tmp_path / "best.json"
        trainer.save_genome_path.write_text(json.dumps({
            "white": "old_w", "black": "old_b",
            "bench_white_win_rate": 0.1, "bench_black_win_rate": 0.1,
        }))
        trainer._save_best(
            white_pop=["new_w"], black_pop=["new_b"],
            white_fitness=[1.0], black_fitness=[1.0],
            bench_w_wr=0.5, bench_b_wr=0.5,
        )
        data = json.loads(trainer.save_genome_path.read_text())
        assert data["white"] == "new_w"
        assert data["black"] == "new_b"

    def test_save_best_persists_hof(self, trainer, tmp_path):
        """HoF genomes should be saved to the genome file."""
        trainer.save_genome_path = tmp_path / "best.json"
        trainer.white_hof = [(20.0, "w_hof_1"), (10.0, "w_hof_2")]
        trainer.black_hof = [(15.0, "b_hof_1")]
        trainer._save_best(
            white_pop=["w"], black_pop=["b"],
            white_fitness=[1.0], black_fitness=[1.0],
            bench_w_wr=0.5, bench_b_wr=0.5,
        )
        data = json.loads(trainer.save_genome_path.read_text())
        assert "white_hof" in data
        assert "black_hof" in data
        assert data["white_hof"] == ["w_hof_1", "w_hof_2"]
        assert data["black_hof"] == ["b_hof_1"]

    def test_save_best_hof_merges_with_existing(self, trainer, tmp_path):
        """New HoF should merge with previously saved HoF, not overwrite."""
        trainer.save_genome_path = tmp_path / "best.json"
        # Simulate a previous run that saved strong HoF genomes
        trainer.save_genome_path.write_text(json.dumps({
            "white": "old_w", "black": "old_b",
            "bench_white_win_rate": 0.1, "bench_black_win_rate": 0.1,
            "white_hof": ["prev_w1", "prev_w2", "prev_w3"],
            "black_hof": ["prev_b1"],
        }))
        # Current run has only 1 HoF genome per color
        trainer.white_hof = [(5.0, "new_w1")]
        trainer.black_hof = [(3.0, "new_b1")]
        trainer._save_best(
            white_pop=["w"], black_pop=["b"],
            white_fitness=[1.0], black_fitness=[1.0],
            bench_w_wr=0.5, bench_b_wr=0.5,
        )
        data = json.loads(trainer.save_genome_path.read_text())
        # New genome should be first (has fitness), old ones fill remaining slots
        assert data["white_hof"][0] == "new_w1"
        assert len(data["white_hof"]) == 4  # 1 new + 3 old
        assert "prev_w1" in data["white_hof"]
        assert "prev_w2" in data["white_hof"]
        assert "prev_w3" in data["white_hof"]

    def test_save_best_hof_dedupes(self, trainer, tmp_path):
        """Same genome in current HoF and saved HoF should not duplicate."""
        trainer.save_genome_path = tmp_path / "best.json"
        trainer.save_genome_path.write_text(json.dumps({
            "white": "w", "black": "b",
            "bench_white_win_rate": 0.0, "bench_black_win_rate": 0.0,
            "white_hof": ["shared_genome", "old_only"],
            "black_hof": [],
        }))
        trainer.white_hof = [(10.0, "shared_genome"), (5.0, "new_only")]
        trainer.black_hof = []
        trainer._save_best(
            white_pop=["w"], black_pop=["b"],
            white_fitness=[1.0], black_fitness=[1.0],
            bench_w_wr=0.5, bench_b_wr=0.5,
        )
        data = json.loads(trainer.save_genome_path.read_text())
        # "shared_genome" appears once, not twice
        assert data["white_hof"].count("shared_genome") == 1
        assert len(data["white_hof"]) == 3  # shared + new_only + old_only

    def test_save_best_hof_capped_at_pop_size(self, trainer, tmp_path):
        """Merged HoF should keep at most pop_size genomes."""
        trainer.save_genome_path = tmp_path / "best.json"
        trainer.save_genome_path.write_text(json.dumps({
            "white": "w", "black": "b",
            "bench_white_win_rate": 0.0, "bench_black_win_rate": 0.0,
            "white_hof": [f"old_{i}" for i in range(5)],
            "black_hof": [],
        }))
        trainer.white_hof = [(float(i), f"new_{i}") for i in range(5, 0, -1)]
        trainer.black_hof = []
        trainer._save_best(
            white_pop=["w"], black_pop=["b"],
            white_fitness=[1.0], black_fitness=[1.0],
            bench_w_wr=0.5, bench_b_wr=0.5,
        )
        data = json.loads(trainer.save_genome_path.read_text())
        assert len(data["white_hof"]) == 5

    def test_save_best_hof_current_beats_old(self, trainer, tmp_path):
        """Current-run genomes (with fitness) should rank above old ones (no fitness)."""
        trainer.save_genome_path = tmp_path / "best.json"
        trainer.save_genome_path.write_text(json.dumps({
            "white": "w", "black": "b",
            "bench_white_win_rate": 0.0, "bench_black_win_rate": 0.0,
            "white_hof": [f"old_{i}" for i in range(5)],
            "black_hof": [],
        }))
        # 3 current genomes should take top 3 spots
        trainer.white_hof = [(30.0, "cur_a"), (20.0, "cur_b"), (10.0, "cur_c")]
        trainer.black_hof = []
        trainer._save_best(
            white_pop=["w"], black_pop=["b"],
            white_fitness=[1.0], black_fitness=[1.0],
            bench_w_wr=0.5, bench_b_wr=0.5,
        )
        data = json.loads(trainer.save_genome_path.read_text())
        # Current genomes ranked first by fitness
        assert data["white_hof"][:3] == ["cur_a", "cur_b", "cur_c"]
        # Old ones fill remaining slots up to pop_size (5)
        assert len(data["white_hof"]) == trainer.pop_size

    def test_save_best_updates_colors_independently(self, trainer, tmp_path):
        trainer.save_genome_path = tmp_path / "best.json"
        trainer.save_genome_path.write_text(json.dumps({
            "white": "old_w", "black": "old_b",
            "bench_white_win_rate": 0.8, "bench_black_win_rate": 0.2,
        }))
        # White regresses, black improves
        trainer._save_best(
            white_pop=["new_w"], black_pop=["new_b"],
            white_fitness=[1.0], black_fitness=[1.0],
            bench_w_wr=0.3, bench_b_wr=0.5,
        )
        data = json.loads(trainer.save_genome_path.read_text())
        assert data["white"] == "old_w", "white should keep old seed"
        assert data["black"] == "new_b", "black should update"
        assert data["bench_white_win_rate"] == 0.8
        assert data["bench_black_win_rate"] == 0.5


# ===========================================================================
# _genome_stats
# ===========================================================================

class TestGenomeStats:
    def test_counts_enabled_connections(self, trainer):
        genome = json.dumps({
            "nodes": [{"id": 0}, {"id": 1}, {"id": 2}],
            "connections": [
                {"in": 0, "out": 1, "weight": 0.5, "enabled": True},
                {"in": 1, "out": 2, "weight": 0.3, "enabled": False},
                {"in": 0, "out": 2, "weight": 0.1, "enabled": True},
            ],
        })
        avg_conns, avg_nodes = trainer._genome_stats([genome])
        assert avg_conns == 2.0
        assert avg_nodes == 3.0

    def test_averages_across_population(self, trainer):
        g1 = json.dumps({
            "nodes": [{"id": 0}, {"id": 1}],
            "connections": [{"in": 0, "out": 1, "weight": 0.5, "enabled": True}],
        })
        g2 = json.dumps({
            "nodes": [{"id": 0}, {"id": 1}, {"id": 2}],
            "connections": [
                {"in": 0, "out": 1, "weight": 0.5, "enabled": True},
                {"in": 1, "out": 2, "weight": 0.3, "enabled": True},
                {"in": 0, "out": 2, "weight": 0.1, "enabled": True},
            ],
        })
        avg_conns, avg_nodes = trainer._genome_stats([g1, g2])
        assert avg_conns == 2.0  # (1+3)/2
        assert avg_nodes == 2.5  # (2+3)/2

    def test_empty_population(self, trainer):
        avg_conns, avg_nodes = trainer._genome_stats([])
        assert avg_conns == 0.0
        assert avg_nodes == 0.0


# ===========================================================================
# _generate_pairings
# ===========================================================================

class TestGeneratePairings:
    def test_correct_count(self, trainer):
        pairings = trainer._generate_pairings()
        expected = trainer.pop_size * min(trainer.tournament_opponents, trainer.pop_size)
        assert len(pairings) == expected

    def test_valid_indices(self, trainer):
        for w, b in trainer._generate_pairings():
            assert 0 <= w < trainer.pop_size
            assert 0 <= b < trainer.pop_size


# ===========================================================================
# Initialization
# ===========================================================================

class TestInitialization:
    def test_default_config_values(self, tmp_path):
        t = NeatCPUTrainer({}, tmp_path / "m.jsonl")
        assert t.pop_size == 50
        assert t.hof_max_size == t.pop_size

    def test_hof_max_size_matches_pop_size(self, trainer):
        assert trainer.hof_max_size == trainer.pop_size

    def test_neat_config_built(self, trainer):
        nc = trainer.neat_config
        assert nc["input_count"] == 4
        assert nc["output_count"] == 3
        assert nc["population_size"] == 5

    def test_benchmark_genomes_created(self, trainer):
        assert len(trainer.benchmark_genomes) == 20


# ===========================================================================
# _compute_outcome_rates – edge cases
# ===========================================================================

class TestComputeOutcomeRatesEdgeCases:
    def test_all_wins(self, trainer):
        results = [_make_game_result(result=1) for _ in range(5)]
        w, d, l = trainer._compute_outcome_rates(results, color=0)
        assert w == 1.0
        assert d == 0.0
        assert l == 0.0

    def test_all_draws(self, trainer):
        results = [_make_game_result(result=2) for _ in range(5)]
        w, d, l = trainer._compute_outcome_rates(results, color=0)
        assert w == 0.0
        assert d == 1.0
        assert l == 0.0

    def test_all_losses(self, trainer):
        results = [_make_game_result(result=-1) for _ in range(5)]
        w, d, l = trainer._compute_outcome_rates(results, color=0)
        assert w == 0.0
        assert d == 0.0
        assert l == 1.0

    def test_black_perspective(self, trainer):
        """result=-1 is black win; result=1 is black loss."""
        results = [_make_game_result(result=-1)]
        w, d, l = trainer._compute_outcome_rates(results, color=1)
        assert w == 1.0
        assert l == 0.0

    def test_rates_sum_to_one(self, trainer):
        results = [_make_game_result(result=r) for r in [1, 2, -1, 2, 1]]
        w, d, l = trainer._compute_outcome_rates(results, color=0)
        assert abs(w + d + l - 1.0) < 1e-6


# ===========================================================================
# _benchmark_vs_random
# ===========================================================================

class TestBenchmarkVsRandom:
    def test_returns_four_floats(self, trainer):
        mock_results = [
            {"result": 1, "white_material": 15.0, "black_material": 10.0}
            for _ in range(10)
        ]
        _chess_cpu_mock.simulate_neat_games_batch = MagicMock(return_value=mock_results)

        white_pop = [_make_genome() for _ in range(trainer.pop_size)]
        black_pop = [_make_genome() for _ in range(trainer.pop_size)]
        white_fitness = [float(i) for i in range(trainer.pop_size)]
        black_fitness = [float(i) for i in range(trainer.pop_size)]
        w_wr, w_mat, b_wr, b_mat = trainer._benchmark_vs_random(
            white_pop, black_pop, white_fitness, black_fitness,
        )
        assert isinstance(w_wr, float)
        assert isinstance(w_mat, float)
        assert isinstance(b_wr, float)
        assert isinstance(b_mat, float)

    def test_all_white_wins(self, trainer):
        mock_results = [
            {"result": 1, "white_material": 20.0, "black_material": 5.0}
            for _ in range(10)
        ]
        _chess_cpu_mock.simulate_neat_games_batch = MagicMock(return_value=mock_results)

        white_pop = [_make_genome() for _ in range(trainer.pop_size)]
        black_pop = [_make_genome() for _ in range(trainer.pop_size)]
        fitness = [float(i) for i in range(trainer.pop_size)]
        w_wr, w_mat, b_wr, b_mat = trainer._benchmark_vs_random(
            white_pop, black_pop, fitness, fitness,
        )
        assert w_wr == 1.0
        assert w_mat > 0

    def test_empty_results(self, trainer):
        _chess_cpu_mock.simulate_neat_games_batch = MagicMock(return_value=[])

        white_pop = [_make_genome() for _ in range(trainer.pop_size)]
        black_pop = [_make_genome() for _ in range(trainer.pop_size)]
        fitness = [1.0] * trainer.pop_size
        w_wr, w_mat, b_wr, b_mat = trainer._benchmark_vs_random(
            white_pop, black_pop, fitness, fitness,
        )
        assert w_wr == 0.0
        assert b_wr == 0.0


# ===========================================================================
# _benchmark_fitness_all
# ===========================================================================

class TestBenchmarkFitnessAll:
    def test_returns_fitness_list(self, trainer):
        mock_results = [
            _make_game_result(white_idx=i % trainer.pop_size, result=1)
            for i in range(20)
        ]
        _chess_cpu_mock.simulate_neat_games_batch = MagicMock(return_value=mock_results)

        pop = [_make_genome() for _ in range(trainer.pop_size)]
        fitness = trainer._benchmark_fitness_all(pop, color=0)
        assert isinstance(fitness, list)
        assert len(fitness) == trainer.pop_size

    def test_includes_hof_opponents(self, trainer):
        """When HoF has entries, opponent pool should grow."""
        trainer.black_hof = [(10.0, _make_genome()), (5.0, _make_genome())]
        mock_results = [_make_game_result(result=2) for _ in range(50)]
        _chess_cpu_mock.simulate_neat_games_batch = MagicMock(return_value=mock_results)

        pop = [_make_genome() for _ in range(trainer.pop_size)]
        trainer._benchmark_fitness_all(pop, color=0)

        # Verify simulate was called with more opponents (benchmark + HoF)
        call_args = _chess_cpu_mock.simulate_neat_games_batch.call_args
        opponents = call_args[0][1]  # second positional arg
        assert len(opponents) == trainer.benchmark_size + 2

    def test_fallback_on_rust_panic(self, trainer):
        """On Rust exception, should return draw-based fallback fitness."""
        _chess_cpu_mock.simulate_neat_games_batch = MagicMock(
            side_effect=RuntimeError("Rust panic"),
        )
        pop = [_make_genome() for _ in range(trainer.pop_size)]
        fitness = trainer._benchmark_fitness_all(pop, color=0)
        assert isinstance(fitness, list)
        assert len(fitness) == trainer.pop_size


# ===========================================================================
# _compute_fitness – loss result
# ===========================================================================

class TestComputeFitnessLoss:
    def test_loss_gives_lower_fitness(self, trainer):
        win = [_make_game_result(result=1)]
        loss = [_make_game_result(result=-1)]
        f_win = trainer._compute_fitness(win, pop_size=2, color=0)
        f_loss = trainer._compute_fitness(loss, pop_size=2, color=0)
        assert f_win[0] > f_loss[0]

    def test_black_loss_from_result_one(self, trainer):
        """result=1 (white wins) should be a loss for black."""
        results = [_make_game_result(result=1)]
        fitness = trainer._compute_fitness(results, pop_size=2, color=1)
        # Black should get loss penalty + negative material diff
        assert fitness[1] < 0 or fitness[1] < trainer._compute_fitness(
            [_make_game_result(result=-1)], pop_size=2, color=1,
        )[1]
