"""Tests for puzzle curriculum integration."""
import json
import os
import sys
import tempfile

import pytest

# Add python/ to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))


@pytest.fixture
def sample_puzzles_file():
    """Create a temp file with sample puzzles."""
    puzzles = [
        {"fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", "solution": "e7e5"},
        {"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "solution": "e2e4"},
        {"fen": "k7/4P3/8/8/8/8/8/K7 w - - 0 1", "solution": "e7e8q"},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(puzzles, f)
        path = f.name
    yield path
    os.unlink(path)


class TestPrepPuzzles:
    """Test scripts/prepare_puzzles.py logic."""

    def test_extract_first_solution_move(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
        from prepare_puzzles import extract_first_solution_move

        # Second move is the solution
        assert extract_first_solution_move("e2e4 e7e5") == "e7e5"
        assert extract_first_solution_move("d2d4 d7d5 c2c4") == "d7d5"
        assert extract_first_solution_move("e7e8q") is None  # Only one move


class TestPuzzleCurriculumConfig:
    """Test that NeatCPUTrainer accepts puzzle config."""

    def test_puzzle_config_defaults(self):
        """Verify default puzzle config keys are set."""
        try:
            from neat_cpu_trainer import NeatCPUTrainer
        except ImportError:
            pytest.skip("neat_cpu_trainer not importable (missing neat_ga)")

        config = {"population_size": 5, "output_size": 128, "curriculum_stage": 0}
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as f:
            trainer = NeatCPUTrainer(config, f.name)

        assert trainer.curriculum_stage == 0
        assert trainer.puzzle_count == 500
        assert trainer.puzzle_max_rating == 800
        assert trainer.puzzle_advance_threshold == 0.85

    def test_puzzle_config_override(self):
        """Verify puzzle config can be overridden."""
        try:
            from neat_cpu_trainer import NeatCPUTrainer
        except ImportError:
            pytest.skip("neat_cpu_trainer not importable (missing neat_ga)")

        config = {
            "population_size": 5,
            "output_size": 128,
            "curriculum_stage": 0,
            "puzzle_count": 100,
            "puzzle_max_rating": 1200,
            "puzzle_advance_threshold": 0.7,
        }
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as f:
            trainer = NeatCPUTrainer(config, f.name)

        assert trainer.puzzle_count == 100
        assert trainer.puzzle_max_rating == 1200
        assert trainer.puzzle_advance_threshold == 0.7


class TestRustPuzzleEvaluator:
    """Test the Rust puzzle evaluator (JSON-based) directly."""

    def test_evaluate_empty_puzzles(self):
        try:
            import chess_cpu
        except ImportError:
            pytest.skip("chess_cpu not available")

        result = chess_cpu.evaluate_puzzles_json_batch(["{}", "{}"], "[]", output_size=384, temperature=0.1)
        assert len(result) == 2
        assert all(s == 0.0 for s in result)

    def test_evaluate_invalid_json(self):
        try:
            import chess_cpu
        except ImportError:
            pytest.skip("chess_cpu not available")

        result = chess_cpu.evaluate_puzzles_json_batch(["{}"], "not json", output_size=384, temperature=0.1)
        assert len(result) == 1
        assert result[0] == 0.0

    def test_evaluate_with_puzzles(self, sample_puzzles_file):
        """Test evaluation with real puzzles and a minimal NEAT genome."""
        try:
            import chess_cpu
            import neat_ga
        except ImportError:
            pytest.skip("chess_cpu or neat_ga not available")

        # Create a minimal NEAT genome
        config = json.dumps({
            "input_count": 389,
            "output_count": 384,
            "population_size": 2,
            "initial_connections_per_output": 1,
        })
        genomes = neat_ga.create_population(config)

        with open(sample_puzzles_file) as f:
            puzzles_json = f.read()

        scores = chess_cpu.evaluate_puzzles_json_batch(
            genomes, puzzles_json, output_size=384, temperature=0.1,
        )
        assert len(scores) == 2
        # Scores should be between 0 and 1
        for s in scores:
            assert 0.0 <= s <= 1.0


class TestRustPuzzleEvaluatorFEN:
    """Test the Rust FEN-based puzzle evaluator (with soft scoring)."""

    def test_evaluate_with_fen_puzzles(self):
        """Test FEN-based evaluate_puzzles_batch returns dict results."""
        try:
            import chess_cpu
            import neat_ga
        except ImportError:
            pytest.skip("chess_cpu or neat_ga not available")

        config = json.dumps({
            "input_count": 389,
            "output_count": 384,
            "population_size": 2,
            "initial_connections_per_output": 1,
        })
        genomes = neat_ga.create_population(config)

        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        best_move = "e2e4"

        results = chess_cpu.evaluate_puzzles_batch(
            genomes, [fen], [best_move], output_size=384,
        )
        assert len(results) == 2
        for r in results:
            assert "correct" in r
            assert "total" in r
            assert "soft_score" in r
            assert 0.0 <= r["soft_score"] <= 1.0
