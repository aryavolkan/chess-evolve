"""Integration test: verify curriculum stage transitions fire correctly."""
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Mock Rust backends
for name in ("chess_cpu", "neat_ga", "evolve_ga"):
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)

sys.modules["chess_cpu"].simulate_neat_games_batch = MagicMock(return_value=[])
sys.modules["chess_cpu"].evaluate_puzzles_batch = MagicMock(return_value=[])
sys.modules["neat_ga"].create_population_with_tracker = MagicMock(
    return_value={"population": ["{}"] * 10, "tracker": "{}"}
)
sys.modules["neat_ga"].evolve_neat_generation = MagicMock(
    return_value={
        "population": ["{}"] * 10, "species": "[]", "tracker": "{}",
        "config": "{}", "stats": {"avg_connections": 5, "avg_nodes": 400,
                                    "species_count": 2, "avg_depth": 1, "avg_width": 3},
    }
)

from curriculum import CurriculumManager


def test_curriculum_manager_full_lifecycle(tmp_path):
    """Test creating, advancing, saving, and loading curriculum state."""
    cm = CurriculumManager({"curriculum_stage": 0})
    assert cm.stage == 0
    assert cm.stage_name() == "puzzles"

    # Simulate stage 0 exit
    metrics = {"puzzle_bench_accuracy": 0.65, "puzzle_max_rating": 1500}
    assert cm.check_exit(metrics) is True
    cm.advance_stage()
    assert cm.stage == 1
    assert cm.stage_name() == "guided_play"
    assert cm.is_transitioning() is True

    # Save and reload
    state_path = tmp_path / "run_state.json"
    cm.save_run_state(state_path, elo_estimate=500)
    data = json.loads(state_path.read_text())
    assert data["highest_stage"] == 1

    # Reload starts one stage back
    cm2 = CurriculumManager.from_run_state(state_path, {})
    assert cm2.stage == 0
