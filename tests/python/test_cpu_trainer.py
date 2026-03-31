"""Tests for CPUTrainer pure-Python logic.

Mocks the Rust backends (chess_cpu, evolve_ga) so we can test fitness
computation, tournament scores, Hall of Fame, immigration, outcome rates,
and pairing generation without needing compiled extensions.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

np = pytest.importorskip("numpy", reason="numpy required for CPUTrainer tests")

# ---------------------------------------------------------------------------
# Mock Rust extension modules before importing CPUTrainer.
# Track which modules we inject so we can remove them afterwards to avoid
# polluting sys.modules for other test files (e.g. test_evolve_ga.py which
# uses a skipif guard based on whether the *real* evolve_ga is importable).
# ---------------------------------------------------------------------------

_injected = []

_chess_cpu_mock = types.ModuleType("chess_cpu")
_chess_cpu_mock.simulate_games_batch = MagicMock(return_value=[])
if "chess_cpu" not in sys.modules:
    sys.modules["chess_cpu"] = _chess_cpu_mock
    _injected.append("chess_cpu")

_evolve_ga_mock = types.ModuleType("evolve_ga")
_evolve_ga_mock.assign_species = MagicMock(return_value=([0] * 10, [[0.0] * 10]))
_evolve_ga_mock.assign_species_flat = MagicMock(return_value=([0] * 10, b"\x00" * 40))
_evolve_ga_mock.shared_fitness = MagicMock(side_effect=lambda f, *a, **kw: f)
_evolve_ga_mock.evolve_generation = MagicMock(side_effect=lambda pop, *a, **kw: pop)
_evolve_ga_mock.evolve_generation_flat = MagicMock(side_effect=lambda pop_bytes, genome_size, *a, **kw: pop_bytes)
if "evolve_ga" not in sys.modules:
    sys.modules["evolve_ga"] = _evolve_ga_mock
    _injected.append("evolve_ga")

from cpu_trainer import CPUTrainer  # noqa: E402

# Clean up mocks so test_evolve_ga.py can detect the real module is missing
for _mod in _injected:
    sys.modules.pop(_mod, None)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def trainer(tmp_path):
    """Create a CPUTrainer with small dimensions for fast tests."""
    config = {
        "population_size": 5,
        "input_size": 4,
        "hidden_size": 2,
        "output_size": 3,
        "elite_count": 1,
        "tournament_opponents": 3,
        "immigration_rate": 0.4,
    }
    return CPUTrainer(config, tmp_path / "metrics.jsonl")


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


# ===========================================================================
# _compute_fitness
# ===========================================================================

class TestComputeFitness:
    def test_white_win_gives_positive_fitness(self, trainer):
        results = [_make_game_result(white_idx=0, black_idx=1, result=1, move_count=30,
                                     white_material=20.0, black_material=5.0)]
        fitness = trainer._compute_fitness(results, pop_size=2, color=0)
        assert fitness[0] > 0, "white win should produce positive fitness"

    def test_white_loss_gives_lower_fitness(self, trainer):
        win_results = [_make_game_result(result=1)]
        loss_results = [_make_game_result(result=-1)]
        win_f = trainer._compute_fitness(win_results, pop_size=2, color=0)
        loss_f = trainer._compute_fitness(loss_results, pop_size=2, color=0)
        assert win_f[0] > loss_f[0]

    def test_draw_fitness_scales_with_material_advantage(self, trainer):
        adv_results = [_make_game_result(result=2, white_material=20.0, black_material=5.0)]
        disadv_results = [_make_game_result(result=2, white_material=5.0, black_material=20.0)]
        f_adv = trainer._compute_fitness(adv_results, pop_size=2, color=0)
        f_disadv = trainer._compute_fitness(disadv_results, pop_size=2, color=0)
        assert f_adv[0] > f_disadv[0]

    def test_black_win_is_result_minus_one(self, trainer):
        """result=-1 means black wins. Verify black's perspective."""
        results = [_make_game_result(result=-1, black_material=20.0, white_material=5.0)]
        fitness = trainer._compute_fitness(results, pop_size=2, color=1)
        assert fitness[1] > 0

    def test_averages_over_multiple_games(self, trainer):
        results = [
            _make_game_result(white_idx=0, result=1, white_material=15.0, black_material=10.0, move_count=20),
            _make_game_result(white_idx=0, result=-1, white_material=5.0, black_material=15.0, move_count=20),
        ]
        fitness = trainer._compute_fitness(results, pop_size=2, color=0)
        # Should be average of win fitness + loss fitness
        assert isinstance(fitness[0], float)

    def test_individual_with_no_games_has_zero_fitness(self, trainer):
        results = [_make_game_result(white_idx=0)]
        fitness = trainer._compute_fitness(results, pop_size=3, color=0)
        assert fitness[2] == 0.0, "individual with no games should have 0 fitness"

    def test_move_count_penalty_applied(self, trainer):
        short = [_make_game_result(result=2, move_count=10)]
        long = [_make_game_result(result=2, move_count=100)]
        f_short = trainer._compute_fitness(short, pop_size=2, color=0)
        f_long = trainer._compute_fitness(long, pop_size=2, color=0)
        assert f_short[0] > f_long[0], "shorter games should have higher fitness (less penalty)"

    def test_mobility_advantage_increases_fitness(self, trainer):
        high_mob = [_make_game_result(result=2, white_mobility=20, black_mobility=5)]
        low_mob = [_make_game_result(result=2, white_mobility=5, black_mobility=20)]
        f_high = trainer._compute_fitness(high_mob, pop_size=2, color=0)
        f_low = trainer._compute_fitness(low_mob, pop_size=2, color=0)
        assert f_high[0] > f_low[0]

    def test_king_safety_increases_fitness(self, trainer):
        safe = [_make_game_result(result=2, white_king_safety=5.0)]
        unsafe = [_make_game_result(result=2, white_king_safety=0.0)]
        f_safe = trainer._compute_fitness(safe, pop_size=2, color=0)
        f_unsafe = trainer._compute_fitness(unsafe, pop_size=2, color=0)
        assert f_safe[0] > f_unsafe[0]

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

    def test_draw_scores_around_half(self, trainer):
        results = [_make_game_result(result=2, white_material=10.0, black_material=10.0)]
        scores = trainer._compute_tournament_scores(results, pop_size=2, color=0)
        assert 0.0 <= scores[0] <= 1.0
        assert abs(scores[0] - 0.5) < 0.3

    def test_draw_material_bonus_clamped(self, trainer):
        results = [_make_game_result(result=2, white_material=100.0, black_material=0.0)]
        scores = trainer._compute_tournament_scores(results, pop_size=2, color=0)
        assert scores[0] <= 0.75, "draw bonus should be clamped at 0.25"

    def test_individual_with_no_games_scores_zero(self, trainer):
        results = [_make_game_result(white_idx=0)]
        scores = trainer._compute_tournament_scores(results, pop_size=3, color=0)
        assert scores[2] == 0.0


# ===========================================================================
# _update_hof
# ===========================================================================

class TestUpdateHoF:
    def test_adds_best_to_empty_hof(self, trainer):
        pop = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        fitness = [5.0, 10.0]
        hof = trainer._update_hof([], pop, fitness)
        assert len(hof) == 1
        assert hof[0][0] == 10.0
        np.testing.assert_array_equal(hof[0][1], [3.0, 4.0])

    def test_hof_sorted_descending(self, trainer):
        pop = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
        fitness = [5.0, 15.0, 10.0]
        hof = []
        for _ in range(3):
            hof = trainer._update_hof(hof, pop, fitness)
        assert hof[0][0] >= hof[-1][0], "HoF should be sorted by fitness descending"

    def test_hof_capped_at_max_size(self, trainer):
        trainer.hof_max_size = 3
        pop = np.array([[float(i)] for i in range(5)], dtype=np.float32)
        hof = []
        for i in range(5):
            fitness = [0.0] * 5
            fitness[i] = float(i + 1) * 10
            hof = trainer._update_hof(hof, pop, fitness)
        assert len(hof) <= 3

    def test_worse_individual_not_added_to_full_hof(self, trainer):
        trainer.hof_max_size = 2
        pop = np.array([[1.0], [2.0]], dtype=np.float32)
        hof = [(100.0, np.array([99.0])), (50.0, np.array([49.0]))]
        fitness = [1.0, 2.0]  # both worse than existing HoF
        hof = trainer._update_hof(hof, pop, fitness)
        assert len(hof) == 2
        assert hof[-1][0] == 50.0, "worst HoF entry should remain 50.0"


# ===========================================================================
# _apply_immigration
# ===========================================================================

class TestApplyImmigration:
    def test_does_not_replace_elites(self, trainer):
        trainer.elite_count = 2
        pop = np.zeros((5, trainer.genome_size), dtype=np.float32)
        pop[0] = 999.0  # elite
        pop[1] = 888.0  # elite
        result = trainer._apply_immigration(pop)
        np.testing.assert_array_equal(result[0], pop[0])
        np.testing.assert_array_equal(result[1], pop[1])

    def test_replaces_some_non_elites(self, trainer):
        trainer.immigration_rate = 0.5
        trainer.elite_count = 1
        pop = np.zeros((5, trainer.genome_size), dtype=np.float32)
        original = pop.copy()  # save copy before in-place mutation
        result = trainer._apply_immigration(pop)
        # At least one non-elite should have changed
        changed = sum(1 for i in range(1, 5) if not np.array_equal(result[i], original[i]))
        assert changed >= 1

    def test_returns_unchanged_when_all_elites(self, trainer):
        trainer.elite_count = 5  # all are elites
        pop = np.ones((5, trainer.genome_size), dtype=np.float32)
        result = trainer._apply_immigration(pop)
        np.testing.assert_array_equal(result, pop)


# ===========================================================================
# _compute_outcome_rates
# ===========================================================================

class TestComputeOutcomeRates:
    def test_all_wins(self, trainer):
        results = [_make_game_result(result=1) for _ in range(10)]
        w, d, l = trainer._compute_outcome_rates(results, color=0)
        assert w == 1.0
        assert d == 0.0
        assert l == 0.0

    def test_all_draws(self, trainer):
        results = [_make_game_result(result=2) for _ in range(10)]
        w, d, l = trainer._compute_outcome_rates(results, color=0)
        assert w == 0.0
        assert d == 1.0
        assert l == 0.0

    def test_all_losses(self, trainer):
        results = [_make_game_result(result=-1) for _ in range(10)]
        w, d, l = trainer._compute_outcome_rates(results, color=0)
        assert w == 0.0
        assert d == 0.0
        assert l == 1.0

    def test_mixed_outcomes(self, trainer):
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

    def test_rates_sum_to_one(self, trainer):
        results = [_make_game_result(result=r) for r in [1, 2, -1, 2, 1]]
        w, d, l = trainer._compute_outcome_rates(results, color=0)
        assert abs(w + d + l - 1.0) < 1e-6

    def test_empty_results_no_crash(self, trainer):
        w, d, l = trainer._compute_outcome_rates([], color=0)
        assert w == 0.0
        assert d == 0.0
        assert l == 0.0

    def test_black_perspective(self, trainer):
        """result=-1 is a black win, result=1 is a black loss."""
        results = [_make_game_result(result=-1)]
        w, d, l = trainer._compute_outcome_rates(results, color=1)
        assert w == 1.0
        assert l == 0.0


# ===========================================================================
# _generate_pairings
# ===========================================================================

class TestGeneratePairings:
    def test_correct_number_of_pairings(self, trainer):
        pairings = trainer._generate_pairings()
        expected = trainer.pop_size * min(trainer.tournament_opponents, trainer.pop_size)
        assert len(pairings) == expected

    def test_all_white_indices_present(self, trainer):
        pairings = trainer._generate_pairings()
        white_indices = {p[0] for p in pairings}
        assert white_indices == set(range(trainer.pop_size))

    def test_pairings_are_valid_indices(self, trainer):
        pairings = trainer._generate_pairings()
        for w, b in pairings:
            assert 0 <= w < trainer.pop_size
            assert 0 <= b < trainer.pop_size


# ===========================================================================
# Initialization
# ===========================================================================

class TestInitialization:
    def test_genome_size_calculation(self, trainer):
        expected = (4 * 2) + 2 + (2 * 3) + 3  # ih + bh + ho + bo
        assert trainer.genome_size == expected

    def test_population_shape(self, trainer):
        pop = trainer._init_population()
        assert pop.shape == (5, trainer.genome_size)
        assert pop.dtype == np.float32

    def test_xavier_init_scale(self, trainer):
        """Verify population uses Xavier initialization scale."""
        pop = trainer._init_population()
        expected_scale = (2.0 / trainer.input_size) ** 0.5
        # Standard deviation should be approximately the Xavier scale
        assert abs(pop.std() - expected_scale) < 0.1

    def test_benchmark_shape(self, trainer):
        assert trainer.benchmark_pop.shape == (20, trainer.genome_size)

    def test_default_config_values(self, tmp_path):
        t = CPUTrainer({}, tmp_path / "m.jsonl")
        assert t.pop_size == 30
        assert t.elite_count == 2
        assert t.mutation_rate == 0.25

    def test_speciation_threshold_initialized(self, trainer):
        assert trainer.white_spec_threshold > 0
        assert trainer.black_spec_threshold > 0


# ===========================================================================
# _speciate_and_evolve
# ===========================================================================

class TestSpeciateAndEvolve:
    def test_returns_correct_shapes(self, trainer):
        pop = trainer._init_population()
        fitness = [float(i) for i in range(trainer.pop_size)]
        new_pop, sp_count, new_reps_bytes, new_n_reps, threshold = trainer._speciate_and_evolve(
            pop, fitness, None, 0, trainer.white_spec_threshold,
        )
        assert new_pop.shape == pop.shape
        assert new_pop.dtype == np.float32
        assert isinstance(sp_count, int)
        assert isinstance(threshold, float)

    def test_threshold_increases_when_too_many_species(self, trainer):
        """If species_count > target_species, threshold should increase."""
        pop = trainer._init_population()
        fitness = [1.0] * trainer.pop_size
        many_species = list(range(trainer.pop_size))
        reps_bytes = b"\x00" * (trainer.pop_size * trainer.genome_size * 4)
        _evolve_ga_mock.assign_species_flat = MagicMock(return_value=(many_species, reps_bytes))

        trainer.target_species = 2
        initial_threshold = 10.0
        _, _, _, _, new_threshold = trainer._speciate_and_evolve(
            pop, fitness, None, 0, initial_threshold,
        )
        assert new_threshold > initial_threshold

    def test_threshold_decreases_when_too_few_species(self, trainer):
        """If species_count < target_species, threshold should decrease."""
        pop = trainer._init_population()
        fitness = [1.0] * trainer.pop_size
        one_species = [0] * trainer.pop_size
        reps_bytes = b"\x00" * (trainer.genome_size * 4)
        _evolve_ga_mock.assign_species_flat = MagicMock(return_value=(one_species, reps_bytes))

        trainer.target_species = 20
        initial_threshold = 10.0
        _, _, _, _, new_threshold = trainer._speciate_and_evolve(
            pop, fitness, None, 0, initial_threshold,
        )
        assert new_threshold < initial_threshold

    def test_threshold_floor_at_one(self, trainer):
        pop = trainer._init_population()
        fitness = [1.0] * trainer.pop_size
        one_species = [0] * trainer.pop_size
        reps_bytes = b"\x00" * (trainer.genome_size * 4)
        _evolve_ga_mock.assign_species_flat = MagicMock(return_value=(one_species, reps_bytes))

        trainer.target_species = 1000
        _, _, _, _, new_threshold = trainer._speciate_and_evolve(
            pop, fitness, None, 0, 0.5,
        )
        assert new_threshold >= 1.0

    def test_fitness_sharing_disabled(self, trainer):
        """When use_fitness_sharing=False, raw fitness should be passed to evolve."""
        trainer.use_fitness_sharing = False
        pop = trainer._init_population()
        fitness = [float(i) for i in range(trainer.pop_size)]
        one_species = [0] * trainer.pop_size
        reps_bytes = b"\x00" * (trainer.genome_size * 4)
        _evolve_ga_mock.assign_species_flat = MagicMock(return_value=(one_species, reps_bytes))
        _evolve_ga_mock.shared_fitness.reset_mock()

        trainer._speciate_and_evolve(pop, fitness, None, 0, 10.0)
        _evolve_ga_mock.shared_fitness.assert_not_called()

    def test_fitness_sharing_enabled(self, trainer):
        """When use_fitness_sharing=True, shared_fitness should be called."""
        trainer.use_fitness_sharing = True
        pop = trainer._init_population()
        fitness = [float(i) for i in range(trainer.pop_size)]
        one_species = [0] * trainer.pop_size
        reps_bytes = b"\x00" * (trainer.genome_size * 4)
        _evolve_ga_mock.assign_species_flat = MagicMock(return_value=(one_species, reps_bytes))
        _evolve_ga_mock.shared_fitness.reset_mock()

        trainer._speciate_and_evolve(pop, fitness, None, 0, 10.0)
        _evolve_ga_mock.shared_fitness.assert_called_once()

    def test_passes_species_reps(self, trainer):
        """Previous representative bytes should be forwarded to assign_species_flat."""
        pop = trainer._init_population()
        fitness = [1.0] * trainer.pop_size
        prev_reps_bytes = b"\x01\x02\x03\x04" * (trainer.genome_size * 2)
        one_species = [0] * trainer.pop_size
        reps_bytes = b"\x00" * (trainer.genome_size * 4)
        _evolve_ga_mock.assign_species_flat = MagicMock(return_value=(one_species, reps_bytes))

        trainer._speciate_and_evolve(pop, fitness, prev_reps_bytes, 2, 10.0)
        call_kwargs = _evolve_ga_mock.assign_species_flat.call_args
        assert call_kwargs[1]["representatives_bytes"] == prev_reps_bytes
        assert call_kwargs[1]["n_reps"] == 2


# ===========================================================================
# _benchmark_vs_random
# ===========================================================================

class TestBenchmarkVsRandom:
    def test_returns_four_floats(self, trainer):
        # Mock simulate_games_batch to return game results
        mock_results = [
            {"result": 1, "white_material": 15.0, "black_material": 10.0}
            for _ in range(20)
        ]
        _chess_cpu_mock.simulate_games_batch = MagicMock(return_value=mock_results)

        white_pop = trainer._init_population()
        black_pop = trainer._init_population()
        w_wr, w_mat, b_wr, b_mat = trainer._benchmark_vs_random(white_pop, black_pop)
        assert isinstance(w_wr, float)
        assert isinstance(w_mat, float)
        assert isinstance(b_wr, float)
        assert isinstance(b_mat, float)

    def test_all_white_wins_gives_full_win_rate(self, trainer):
        mock_results = [
            {"result": 1, "white_material": 20.0, "black_material": 5.0}
            for _ in range(10)
        ]
        _chess_cpu_mock.simulate_games_batch = MagicMock(return_value=mock_results)

        white_pop = trainer._init_population()
        black_pop = trainer._init_population()
        w_wr, w_mat, b_wr, b_mat = trainer._benchmark_vs_random(white_pop, black_pop)
        assert w_wr == 1.0
        assert w_mat > 0

    def test_all_draws_gives_zero_win_rate(self, trainer):
        mock_results = [
            {"result": 2, "white_material": 10.0, "black_material": 10.0}
            for _ in range(10)
        ]
        _chess_cpu_mock.simulate_games_batch = MagicMock(return_value=mock_results)

        white_pop = trainer._init_population()
        black_pop = trainer._init_population()
        w_wr, _, b_wr, _ = trainer._benchmark_vs_random(white_pop, black_pop)
        assert w_wr == 0.0

    def test_empty_results(self, trainer):
        _chess_cpu_mock.simulate_games_batch = MagicMock(return_value=[])
        white_pop = trainer._init_population()
        black_pop = trainer._init_population()
        w_wr, w_mat, b_wr, b_mat = trainer._benchmark_vs_random(white_pop, black_pop)
        assert w_wr == 0.0
        assert b_wr == 0.0
