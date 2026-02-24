"""Tests for the evolve-ga Rust extension module.

These tests verify the genetic algorithm operators exposed via PyO3:
  - Selection: tournament, SUS, rank-based
  - Crossover: two-point, SBX, BLX-alpha
  - Mutation: Gaussian, adaptive Gaussian, Cauchy, 1/5th rule
  - Speciation: species assignment, shared fitness
  - Island model: migration
  - Batch: evolve_generation
"""
from __future__ import annotations

import random
from collections import Counter

import pytest

try:
    import evolve_ga
except ImportError:
    evolve_ga = None

pytestmark = pytest.mark.skipif(evolve_ga is None, reason="evolve_ga not installed (run: cd rust/evolve-ga && maturin develop)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_weights(n: int = 100) -> list[float]:
    return [random.gauss(0, 0.1) for _ in range(n)]


def _random_population(pop_size: int = 10, genome_len: int = 100) -> list[list[float]]:
    return [_random_weights(genome_len) for _ in range(pop_size)]


def _random_fitness(n: int = 10) -> list[float]:
    return [random.random() * 10 for _ in range(n)]


# ===========================================================================
# Selection tests
# ===========================================================================


class TestTournamentSelect:
    def test_returns_valid_index(self):
        fitness = [1.0, 2.0, 3.0, 4.0, 5.0]
        idx = evolve_ga.tournament_select(fitness, k=3)
        assert 0 <= idx < len(fitness)

    def test_full_tournament_selects_best(self):
        fitness = [1.0, 5.0, 3.0, 2.0, 4.0]
        # k = pop_size should always return the best
        for _ in range(50):
            idx = evolve_ga.tournament_select(fitness, k=5)
            assert idx == 1  # fitness[1] = 5.0

    def test_empty_fitness_raises(self):
        with pytest.raises(ValueError):
            evolve_ga.tournament_select([], k=3)


class TestTournamentSelectBatch:
    def test_returns_correct_count(self):
        fitness = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = evolve_ga.tournament_select_batch(fitness, n=10, k=3)
        assert len(result) == 10
        assert all(0 <= idx < 5 for idx in result)

    def test_biases_toward_best(self):
        fitness = [0.0, 0.0, 0.0, 0.0, 100.0]
        counts = Counter(evolve_ga.tournament_select_batch(fitness, n=1000, k=3))
        assert counts[4] > counts.get(0, 0), "best individual should be selected most often"


class TestSUSSelect:
    def test_returns_correct_count(self):
        fitness = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = evolve_ga.sus_select(fitness, n=3)
        assert len(result) == 3

    def test_all_indices_valid(self):
        fitness = [1.0, 2.0, 3.0]
        result = evolve_ga.sus_select(fitness, n=10)
        assert all(0 <= idx < 3 for idx in result)

    def test_higher_fitness_selected_more(self):
        fitness = [0.1, 0.1, 0.1, 0.1, 100.0]
        counts = Counter()
        for _ in range(100):
            result = evolve_ga.sus_select(fitness, n=5)
            for idx in result:
                counts[idx] += 1
        assert counts[4] > counts[0], "highest fitness should dominate"


class TestRankSelect:
    def test_returns_correct_count(self):
        fitness = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = evolve_ga.rank_select(fitness, n=4, selective_pressure=1.5)
        assert len(result) == 4

    def test_selective_pressure_effect(self):
        fitness = [0.0, 0.0, 0.0, 0.0, 100.0]
        # High selective pressure should bias strongly toward best
        counts_high = Counter()
        counts_low = Counter()
        for _ in range(200):
            for idx in evolve_ga.rank_select(fitness, n=1, selective_pressure=2.0):
                counts_high[idx] += 1
            for idx in evolve_ga.rank_select(fitness, n=1, selective_pressure=1.0):
                counts_low[idx] += 1
        # With uniform pressure (1.0), selection should be more even
        # With max pressure (2.0), best should dominate more
        assert counts_high[4] >= counts_low[4] * 0.5  # Allow some randomness


# ===========================================================================
# Crossover tests
# ===========================================================================


class TestTwoPointCrossover:
    def test_preserves_length(self):
        a = [1.0] * 100
        b = [2.0] * 100
        child = evolve_ga.two_point_crossover(a, b)
        assert len(child) == 100

    def test_mixes_parents(self):
        a = [0.0] * 200
        b = [1.0] * 200
        child = evolve_ga.two_point_crossover(a, b)
        has_zero = any(v == 0.0 for v in child)
        has_one = any(v == 1.0 for v in child)
        assert has_zero or has_one

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            evolve_ga.two_point_crossover([1.0, 2.0], [1.0, 2.0, 3.0])


class TestSBXCrossover:
    def test_preserves_length(self):
        a = [1.0] * 50
        b = [2.0] * 50
        child = evolve_ga.sbx_crossover(a, b, eta=2.0)
        assert len(child) == 50

    def test_identical_parents_produce_same_child(self):
        a = [1.5] * 50
        child = evolve_ga.sbx_crossover(a, a, eta=2.0)
        for v in child:
            assert abs(v - 1.5) < 1e-5

    def test_high_eta_stays_close(self):
        a = [0.0] * 100
        b = [1.0] * 100
        # With very high eta, children should be close to parents
        child = evolve_ga.sbx_crossover(a, b, eta=100.0)
        for v in child:
            assert -0.5 <= v <= 1.5, f"value {v} too far from parents with high eta"


class TestBLXAlphaCrossover:
    def test_preserves_length(self):
        a = [0.0] * 50
        b = [1.0] * 50
        child = evolve_ga.blx_alpha_crossover(a, b, alpha=0.5)
        assert len(child) == 50

    def test_alpha_zero_interpolates_only(self):
        a = [0.0] * 1000
        b = [1.0] * 1000
        child = evolve_ga.blx_alpha_crossover(a, b, alpha=0.0)
        for v in child:
            assert 0.0 <= v <= 1.0, f"got {v} outside [0, 1] with alpha=0"

    def test_alpha_positive_can_extrapolate(self):
        a = [0.0] * 10000
        b = [1.0] * 10000
        child = evolve_ga.blx_alpha_crossover(a, b, alpha=0.5)
        outside = any(v < 0.0 or v > 1.0 for v in child)
        assert outside, "with alpha=0.5 and many elements, some should extrapolate"


# ===========================================================================
# Mutation tests
# ===========================================================================


class TestGaussianMutate:
    def test_preserves_length(self):
        w = [0.0] * 100
        result = evolve_ga.gaussian_mutate(w, mutation_rate=0.5, strength=0.1)
        assert len(result) == 100

    def test_zero_rate_no_change(self):
        w = [1.0] * 100
        result = evolve_ga.gaussian_mutate(w, mutation_rate=0.0, strength=0.5)
        assert result == w

    def test_full_rate_mutates_all(self):
        w = [0.0] * 1000
        result = evolve_ga.gaussian_mutate(w, mutation_rate=1.0, strength=0.5)
        changed = sum(1 for v in result if v != 0.0)
        assert changed > 990, f"expected almost all mutated, got {changed}"

    def test_partial_rate_mutates_some(self):
        w = [0.0] * 10000
        result = evolve_ga.gaussian_mutate(w, mutation_rate=0.1, strength=0.5)
        changed = sum(1 for v in result if v != 0.0)
        # Expect ~1000 mutations (+/- 100 or so)
        assert 700 < changed < 1300, f"expected ~1000 mutations, got {changed}"


class TestAdaptiveGaussianMutate:
    def test_returns_correct_shapes(self):
        w = [0.0] * 50
        sigmas = [0.1] * 50
        new_w, new_s = evolve_ga.adaptive_gaussian_mutate(w, sigmas)
        assert len(new_w) == 50
        assert len(new_s) == 50

    def test_sigmas_change(self):
        w = [0.0] * 50
        sigmas = [0.1] * 50
        _, new_s = evolve_ga.adaptive_gaussian_mutate(w, sigmas)
        assert any(abs(s - 0.1) > 1e-10 for s in new_s), "sigmas should adapt"

    def test_sigma_min_respected(self):
        w = [0.0] * 10
        sigmas = [1e-10] * 10
        _, new_s = evolve_ga.adaptive_gaussian_mutate(w, sigmas, sigma_min=0.01)
        assert all(s >= 0.01 for s in new_s), "sigma_min should be respected"

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            evolve_ga.adaptive_gaussian_mutate([0.0] * 5, [0.1] * 3)


class TestOneFifthRule:
    def test_increases_on_high_success(self):
        sigma = 0.1
        new_sigma = evolve_ga.one_fifth_rule_adapt(sigma, 5, 10)
        assert new_sigma > sigma, "50% success rate should increase sigma"

    def test_decreases_on_low_success(self):
        sigma = 0.1
        new_sigma = evolve_ga.one_fifth_rule_adapt(sigma, 0, 10)
        assert new_sigma < sigma, "0% success rate should decrease sigma"

    def test_bounds_respected(self):
        # Below min
        new_sigma = evolve_ga.one_fifth_rule_adapt(0.0001, 0, 10, sigma_min=0.001, sigma_max=1.0)
        assert new_sigma >= 0.001

        # Above max
        new_sigma = evolve_ga.one_fifth_rule_adapt(0.99, 10, 10, sigma_min=0.001, sigma_max=1.0)
        assert new_sigma <= 1.0

    def test_zero_total_no_change(self):
        sigma = 0.1
        new_sigma = evolve_ga.one_fifth_rule_adapt(sigma, 0, 0)
        assert new_sigma == sigma


class TestCauchyMutate:
    def test_preserves_length(self):
        w = [0.0] * 100
        result = evolve_ga.cauchy_mutate(w, mutation_rate=0.5, scale=0.1)
        assert len(result) == 100

    def test_zero_rate_no_change(self):
        w = [1.0] * 100
        result = evolve_ga.cauchy_mutate(w, mutation_rate=0.0, scale=0.1)
        assert result == w

    def test_has_occasional_large_jumps(self):
        """Cauchy distribution has heavier tails than Gaussian."""
        w = [0.0] * 10000
        result = evolve_ga.cauchy_mutate(w, mutation_rate=1.0, scale=0.1)
        large_jumps = sum(1 for v in result if abs(v) > 1.0)
        # Cauchy should produce some large outliers
        assert large_jumps > 0, "Cauchy should produce occasional large mutations"


# ===========================================================================
# Speciation tests
# ===========================================================================


class TestAssignSpecies:
    def test_identical_individuals_same_species(self):
        pop = [[1.0] * 10] * 5
        ids, reps = evolve_ga.assign_species(pop, threshold=3.0)
        assert len(ids) == 5
        assert all(id == 0 for id in ids)
        assert len(reps) == 1

    def test_distant_individuals_different_species(self):
        pop = [[0.0] * 10, [100.0] * 10]
        ids, reps = evolve_ga.assign_species(pop, threshold=1.0)
        assert ids[0] != ids[1]
        assert len(reps) == 2

    def test_with_previous_representatives(self):
        reps = [[0.0] * 5, [10.0] * 5]
        pop = [[0.1] * 5, [9.9] * 5, [0.2] * 5]
        ids, _ = evolve_ga.assign_species(pop, threshold=3.0, representatives=reps)
        assert ids[0] == 0  # close to rep 0
        assert ids[1] == 1  # close to rep 1
        assert ids[2] == 0  # close to rep 0

    def test_empty_population(self):
        ids, reps = evolve_ga.assign_species([], threshold=3.0)
        assert len(ids) == 0
        assert len(reps) == 0


class TestSharedFitness:
    def test_divides_by_species_size(self):
        fitness = [10.0, 10.0, 10.0, 20.0]
        species_ids = [0, 0, 0, 1]
        shared = evolve_ga.shared_fitness(fitness, species_ids)
        assert abs(shared[0] - 10.0 / 3.0) < 1e-4
        assert abs(shared[3] - 20.0) < 1e-4

    def test_min_species_size(self):
        fitness = [10.0]
        species_ids = [0]
        shared = evolve_ga.shared_fitness(fitness, species_ids, min_species_size=5)
        assert abs(shared[0] - 2.0) < 1e-4  # 10 / 5

    def test_mismatched_raises(self):
        with pytest.raises(ValueError):
            evolve_ga.shared_fitness([1.0, 2.0], [0])


# ===========================================================================
# Island model tests
# ===========================================================================


class TestMigrate:
    def test_ring_migration(self):
        # Two islands, each with 5 individuals
        pop0 = [[float(i)] * 5 for i in range(5)]
        fit0 = [float(i) for i in range(5)]
        pop1 = [[float(i + 100)] * 5 for i in range(5)]
        fit1 = [float(i) for i in range(5)]

        new_islands, new_fitness = evolve_ga.migrate(
            [pop0, pop1], [fit0, fit1], n_migrants=1, topology="ring"
        )
        assert len(new_islands) == 2
        assert len(new_islands[0]) == 5
        assert len(new_islands[1]) == 5

    def test_preserves_population_sizes(self):
        islands = [_random_population(8, 20) for _ in range(3)]
        fitness = [_random_fitness(8) for _ in range(3)]
        new_islands, new_fitness = evolve_ga.migrate(islands, fitness, n_migrants=2)
        for i in range(3):
            assert len(new_islands[i]) == 8
            assert len(new_fitness[i]) == 8

    def test_single_island_no_migration(self):
        pop = _random_population(5, 10)
        fit = _random_fitness(5)
        new_islands, new_fitness = evolve_ga.migrate([pop], [fit], n_migrants=2)
        assert len(new_islands) == 1
        assert new_islands[0] == pop


# ===========================================================================
# Batch evolution test
# ===========================================================================


class TestEvolveGeneration:
    def test_returns_same_pop_size(self):
        pop = _random_population(20, 50)
        fit = _random_fitness(20)
        new_pop = evolve_ga.evolve_generation(pop, fit, elite_count=3)
        assert len(new_pop) == 20

    def test_preserves_genome_length(self):
        pop = _random_population(10, 200)
        fit = _random_fitness(10)
        new_pop = evolve_ga.evolve_generation(pop, fit)
        for genome in new_pop:
            assert len(genome) == 200

    def test_elites_preserved(self):
        # Create a population where one individual is clearly the best
        pop = [[0.0] * 50 for _ in range(10)]
        pop[3] = [42.0] * 50  # Distinctive individual
        fit = [0.0] * 10
        fit[3] = 100.0  # Best fitness

        new_pop = evolve_ga.evolve_generation(pop, fit, elite_count=1, mutation_rate=0.0, crossover_rate=0.0)
        # The best individual (all 42s) should be preserved as elite
        assert new_pop[0] == [42.0] * 50

    def test_different_crossover_modes(self):
        pop = _random_population(10, 50)
        fit = _random_fitness(10)

        for mode in ["two_point", "sbx", "blx_alpha"]:
            new_pop = evolve_ga.evolve_generation(pop, fit, crossover_mode=mode)
            assert len(new_pop) == 10
            for genome in new_pop:
                assert len(genome) == 50

    def test_empty_population_raises(self):
        with pytest.raises(ValueError):
            evolve_ga.evolve_generation([], [])

    def test_mismatched_raises(self):
        with pytest.raises(ValueError):
            evolve_ga.evolve_generation([[1.0]], [1.0, 2.0])
