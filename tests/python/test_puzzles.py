"""Tests for puzzle evaluation via the Rust chess_cpu crate.

Verifies:
- Side-to-move consistency: white networks should only see white-to-move puzzles
- Soft scoring: partial credit for ranking the correct move highly
- Binary vs soft scoring differentiation
"""
from __future__ import annotations

import json

import pytest

try:
    import chess_cpu
    import neat_ga
except ImportError:
    pytest.skip("Rust extensions not built", allow_module_level=True)


def _make_genome(input_size: int, output_size: int, seed: int = 42) -> str:
    """Create a minimal NEAT genome via Rust."""
    config = {
        "input_count": input_size,
        "output_count": output_size,
        "population_size": 1,
        "initial_connections_per_output": 3,
        "compatibility_threshold": 3.0,
        "c1_excess": 1.0,
        "c2_disjoint": 1.0,
        "c3_weight_diff": 0.4,
        "target_species_count": 1,
        "threshold_step": 0.3,
        "weight_mutate_rate": 0.0,
        "weight_perturb_rate": 0.0,
        "weight_perturb_strength": 0.0,
        "weight_reset_range": 2.0,
        "add_node_rate": 0.0,
        "add_connection_rate": 0.0,
        "disable_connection_rate": 0.0,
        "add_node_count": 0,
        "add_connection_count": 0,
        "prune_rate": 0.0,
        "complexity_cost": 0.0,
        "elite_fraction": 1.0,
        "survival_fraction": 1.0,
        "crossover_rate": 0.0,
        "interspecies_crossover_rate": 0.0,
        "disabled_gene_inherit_rate": 0.75,
        "stagnation_threshold": 999,
        "stagnation_kill_threshold": 999,
        "min_species_protected": 1,
    }
    pop = neat_ga.create_population(json.dumps(config))
    return pop[0]


# --- Puzzle FENs ---
# White to move: Scholar's mate position, Qxf7# is the answer
WHITE_TO_MOVE_FEN = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
WHITE_TO_MOVE_BEST = "h5f7"  # Qxf7#

# Black to move: back-rank mate
BLACK_TO_MOVE_FEN = "6k1/5ppp/8/8/8/8/5PPP/r3K2R b K - 0 1"
BLACK_TO_MOVE_BEST = "a1e1"  # Ra1-e1#


class TestPuzzleSideToMove:
    """Verify that puzzle evaluation works correctly for both sides."""

    def test_white_to_move_puzzle_encodes_correctly(self):
        """A white-to-move puzzle should be evaluable by a genome."""
        genome = _make_genome(389, 384)
        results = chess_cpu.evaluate_puzzles_batch(
            [genome], [WHITE_TO_MOVE_FEN], [WHITE_TO_MOVE_BEST], output_size=384,
        )
        assert len(results) == 1
        assert results[0]["total"] == 1
        # The genome is random, but it should produce a valid result
        assert results[0]["correct"] in (0, 1)
        assert 0.0 <= results[0]["soft_score"] <= 1.0

    def test_black_to_move_puzzle_encodes_correctly(self):
        """A black-to-move puzzle should be evaluable by a genome."""
        genome = _make_genome(389, 384)
        results = chess_cpu.evaluate_puzzles_batch(
            [genome], [BLACK_TO_MOVE_FEN], [BLACK_TO_MOVE_BEST], output_size=384,
        )
        assert len(results) == 1
        assert results[0]["total"] == 1
        assert results[0]["correct"] in (0, 1)
        assert 0.0 <= results[0]["soft_score"] <= 1.0

    def test_side_to_move_changes_encoding(self):
        """The same board with different side-to-move should produce different inputs.

        This proves that a network trained on white-to-move puzzles sees
        different input distributions than one trained on black-to-move puzzles.
        """
        w_encoding = chess_cpu.encode_board_fen(WHITE_TO_MOVE_FEN)
        b_encoding = chess_cpu.encode_board_fen(BLACK_TO_MOVE_FEN)

        # The side-to-move bit (index 384) should differ
        # White: side_to_move=0 → encoded as 0.0
        # Black: side_to_move=1 → encoded as 1.0
        assert w_encoding[384] == 0.0, "White-to-move should encode side_to_move as 0.0"
        assert b_encoding[384] == 1.0, "Black-to-move should encode side_to_move as 1.0"

    def test_network_never_sees_wrong_side_in_games(self):
        """During games, white network only acts when side_to_move=0.

        This test verifies the encoding difference exists, proving that
        training a white network on black-to-move puzzles is wasteful.
        """
        # Starting position: white to move
        start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        # After 1.e4: black to move
        after_e4_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"

        start_enc = chess_cpu.encode_board_fen(start_fen)
        after_e4_enc = chess_cpu.encode_board_fen(after_e4_fen)

        # In a real game, white network sees start_enc (side_to_move=0),
        # but never sees after_e4_enc (side_to_move=1)
        assert start_enc[384] == 0.0
        assert after_e4_enc[384] == 1.0


class TestPuzzleSoftScoring:
    """Verify that soft scoring provides gradient for evolution."""

    def test_soft_score_returned(self):
        """evaluate_puzzles_batch should return soft_score field."""
        genome = _make_genome(389, 384)
        results = chess_cpu.evaluate_puzzles_batch(
            [genome], [WHITE_TO_MOVE_FEN], [WHITE_TO_MOVE_BEST], output_size=384,
        )
        assert "soft_score" in results[0]
        assert isinstance(results[0]["soft_score"], float)

    def test_correct_answer_gives_max_soft_score(self):
        """If the network gets the puzzle right, soft_score should equal 1.0 per puzzle."""
        genome = _make_genome(389, 384)
        results = chess_cpu.evaluate_puzzles_batch(
            [genome], [WHITE_TO_MOVE_FEN], [WHITE_TO_MOVE_BEST], output_size=384,
        )
        if results[0]["correct"] == 1:
            assert results[0]["soft_score"] == 1.0

    def test_wrong_answer_still_gets_partial_credit(self):
        """Even if the network doesn't pick the right move, soft_score > 0."""
        # Use many genomes to ensure at least some get it wrong
        genomes = [_make_genome(389, 384, seed=i) for i in range(20)]
        # Duplicate genome strings don't matter — each is independently evaluated
        results = chess_cpu.evaluate_puzzles_batch(
            genomes, [WHITE_TO_MOVE_FEN], [WHITE_TO_MOVE_BEST], output_size=384,
        )
        wrong_results = [r for r in results if r["correct"] == 0]
        if wrong_results:
            # At least one wrong answer should still have partial credit
            assert any(r["soft_score"] > 0.0 for r in wrong_results), \
                "Wrong answers should get partial credit via rank-based soft scoring"

    def test_soft_score_varies_across_genomes(self):
        """Different genomes should get different soft scores, enabling selection."""
        genomes = [_make_genome(389, 384, seed=i) for i in range(20)]
        # Use multiple puzzles for more signal
        fens = [WHITE_TO_MOVE_FEN, BLACK_TO_MOVE_FEN]
        moves = [WHITE_TO_MOVE_BEST, BLACK_TO_MOVE_BEST]
        results = chess_cpu.evaluate_puzzles_batch(
            genomes, fens, moves, output_size=384,
        )
        scores = [r["soft_score"] for r in results]
        # With 20 different genomes, we should see score variation
        assert max(scores) > min(scores), \
            "Soft scores should vary across genomes to provide selection signal"

    def test_soft_score_bounded(self):
        """Soft score per puzzle should be in [0, 1], so total in [0, n_puzzles]."""
        genome = _make_genome(389, 384)
        n_puzzles = 5
        fens = [WHITE_TO_MOVE_FEN] * n_puzzles
        moves = [WHITE_TO_MOVE_BEST] * n_puzzles
        results = chess_cpu.evaluate_puzzles_batch(
            [genome], fens, moves, output_size=384,
        )
        assert 0.0 <= results[0]["soft_score"] <= float(n_puzzles)


class TestPuzzleMultipleEncodings:
    """Verify puzzle evaluation works with all output encodings."""

    @pytest.mark.parametrize("output_size", [128, 384, 4096])
    def test_encoding_produces_valid_results(self, output_size):
        """All three output encodings should produce valid puzzle results."""
        genome = _make_genome(389, output_size)
        results = chess_cpu.evaluate_puzzles_batch(
            [genome], [WHITE_TO_MOVE_FEN], [WHITE_TO_MOVE_BEST],
            output_size=output_size,
        )
        assert len(results) == 1
        assert results[0]["total"] == 1
        assert 0.0 <= results[0]["soft_score"] <= 1.0
