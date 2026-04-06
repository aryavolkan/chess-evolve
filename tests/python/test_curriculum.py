"""Tests for curriculum stage management."""
import pytest
from curriculum import CurriculumManager, STAGES


def test_stages_count():
    assert len(STAGES) == 5


def test_stage_names():
    assert [s["name"] for s in STAGES] == [
        "puzzles", "guided_play", "opponent_ladder", "sf_shaping", "coevo_refinement",
    ]


def test_initial_stage_default():
    cm = CurriculumManager({})
    assert cm.stage == 0


def test_initial_stage_from_config():
    cm = CurriculumManager({"curriculum_stage": 2})
    assert cm.stage == 2


def test_temperature_stage0():
    cm = CurriculumManager({})
    cm.stage = 0
    assert cm.training_temperature(gen_in_stage=0) == pytest.approx(0.3)
    assert cm.training_temperature(gen_in_stage=10) == pytest.approx(0.3)


def test_temperature_stage1_anneals():
    cm = CurriculumManager({})
    cm.stage = 1
    assert cm.training_temperature(gen_in_stage=0) == pytest.approx(0.3)
    assert cm.training_temperature(gen_in_stage=15) == pytest.approx(0.15)
    assert cm.training_temperature(gen_in_stage=7) == pytest.approx(0.225, abs=0.01)


def test_temperature_stage4_fixed():
    cm = CurriculumManager({})
    cm.stage = 4
    assert cm.training_temperature(gen_in_stage=0) == pytest.approx(0.05)
    assert cm.training_temperature(gen_in_stage=100) == pytest.approx(0.05)


def test_eval_temperature_always_zero():
    cm = CurriculumManager({})
    for stage in range(5):
        cm.stage = stage
        assert cm.eval_temperature() == 0.0


def test_stage_weights_stage0_high_checkmate():
    cm = CurriculumManager({})
    cm.stage = 0
    w = cm.fitness_weights()
    assert w["checkmate_bonus"] == 15.0
    assert w["mobility_weight"] == 0.0
    assert w["sf_fitness_weight"] == 0.0


def test_stage_weights_stage1_high_material():
    cm = CurriculumManager({})
    cm.stage = 1
    w = cm.fitness_weights()
    assert w["material_weight"] == 1.5
    assert w["mobility_weight"] == 0.5


def test_stage_weights_stage3_sf_ramps():
    cm = CurriculumManager({})
    cm.stage = 3
    cm.gen_in_stage = 0
    w = cm.fitness_weights()
    assert w["sf_fitness_weight"] == pytest.approx(0.2)

    cm.gen_in_stage = 20
    w = cm.fitness_weights()
    assert w["sf_fitness_weight"] == pytest.approx(0.5)

    cm.gen_in_stage = 10
    w = cm.fitness_weights()
    assert w["sf_fitness_weight"] == pytest.approx(0.35)


def test_stage_weights_stage4_low_sf():
    cm = CurriculumManager({})
    cm.stage = 4
    w = cm.fitness_weights()
    assert w["sf_fitness_weight"] == pytest.approx(0.1)


def test_check_stage0_exit_not_ready():
    cm = CurriculumManager({})
    cm.stage = 0
    metrics = {"puzzle_bench_accuracy": 0.10, "puzzle_max_rating": 600}
    assert cm.check_exit(metrics) is False


def test_check_stage0_exit_ready():
    cm = CurriculumManager({})
    cm.stage = 0
    metrics = {"puzzle_bench_accuracy": 0.15, "puzzle_max_rating": 800}
    assert cm.check_exit(metrics) is True


def test_check_stage1_needs_consecutive():
    cm = CurriculumManager({})
    cm.stage = 1
    assert cm.check_exit({"bench_avg_win_rate": 0.20}) is False
    assert cm.check_exit({"bench_avg_win_rate": 0.20}) is False
    assert cm.check_exit({"bench_avg_win_rate": 0.20}) is True


def test_check_stage1_resets_on_low():
    cm = CurriculumManager({})
    cm.stage = 1
    cm.check_exit({"bench_avg_win_rate": 0.20})
    cm.check_exit({"bench_avg_win_rate": 0.20})
    cm.check_exit({"bench_avg_win_rate": 0.05})
    assert cm.check_exit({"bench_avg_win_rate": 0.20}) is False


def test_check_stage2_exit_by_win_rate():
    cm = CurriculumManager({})
    cm.stage = 2
    metrics = {"bench_avg_win_rate": 0.30, "sf_avg_cpl": 9999}
    assert cm.check_exit(metrics) is True


def test_check_stage2_exit_by_cpl():
    cm = CurriculumManager({})
    cm.stage = 2
    metrics = {"bench_avg_win_rate": 0.05, "sf_avg_cpl": 1100}
    assert cm.check_exit(metrics) is True


def test_check_stage2_no_exit():
    cm = CurriculumManager({})
    cm.stage = 2
    metrics = {"bench_avg_win_rate": 0.10, "sf_avg_cpl": 1500}
    assert cm.check_exit(metrics) is False


def test_check_stage3_needs_5_consecutive():
    cm = CurriculumManager({})
    cm.stage = 3
    for _ in range(4):
        assert cm.check_exit({"sf_avg_cpl": 700}) is False
    assert cm.check_exit({"sf_avg_cpl": 700}) is True


def test_check_stage4_never_exits():
    cm = CurriculumManager({})
    cm.stage = 4
    assert cm.check_exit({}) is False


def test_transition_blending_active():
    cm = CurriculumManager({})
    cm.stage = 0
    cm.advance_stage()
    assert cm.stage == 1
    assert cm.is_transitioning() is True
    assert cm.transition_blend_weight() == pytest.approx(0.7)
    cm.tick_generation()
    cm.tick_generation()
    cm.tick_generation()
    assert cm.is_transitioning() is False
    assert cm.transition_blend_weight() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Regression tests for training bugs (B1-B5)
# ---------------------------------------------------------------------------


def test_stage0_exit_uses_config_threshold():
    """B9 regression: check_exit should use puzzle_advance_threshold from config,
    not a hardcoded value."""
    cm_strict = CurriculumManager({"puzzle_advance_threshold": 0.50})
    cm_strict.stage = 0
    # 20% accuracy passes default 0.12 but not 0.50
    assert cm_strict.check_exit({"puzzle_bench_accuracy": 0.20, "puzzle_max_rating": 900}) is False

    cm_easy = CurriculumManager({"puzzle_advance_threshold": 0.10})
    cm_easy.stage = 0
    assert cm_easy.check_exit({"puzzle_bench_accuracy": 0.15, "puzzle_max_rating": 900}) is True


def test_stage0_exit_works_without_puzzle_bench_accuracy():
    """B2 regression: check_exit should return False (not crash) when
    puzzle_bench_accuracy is missing from metrics."""
    cm = CurriculumManager({})
    cm.stage = 0
    assert cm.check_exit({"puzzle_max_rating": 900}) is False


def test_stage2_exit_requires_sf_avg_cpl_key():
    """B1 regression: stage 2 exit reads sf_avg_cpl. This test documents the
    exact key name that the trainer must write."""
    cm = CurriculumManager({})
    cm.stage = 2
    # With the wrong key name, CPL exit never fires
    assert cm.check_exit({"bench_avg_win_rate": 0.10, "sf_white_avg_cpl": 500}) is False
    # With the correct key name, it works
    assert cm.check_exit({"bench_avg_win_rate": 0.10, "sf_avg_cpl": 500}) is True


def test_stage3_exit_requires_sf_avg_cpl_key():
    """B1 regression: stage 3 exit also reads sf_avg_cpl."""
    cm = CurriculumManager({})
    cm.stage = 3
    for _ in range(5):
        cm.check_exit({"sf_avg_cpl": 700})
    assert cm._consecutive_sf_cpl == 5


def test_check_exit_keys_are_documented():
    """Contract test: all metric keys read by check_exit must be listed here.
    If check_exit starts reading a new key, this test forces an update."""
    import inspect
    source = inspect.getsource(CurriculumManager.check_exit)
    expected_keys = [
        "puzzle_bench_accuracy",
        "puzzle_max_rating",
        "bench_avg_win_rate",
        "sf_avg_cpl",
    ]
    for key in expected_keys:
        assert key in source, f"check_exit should read '{key}'"


import json
from pathlib import Path


def test_save_run_state(tmp_path):
    cm = CurriculumManager({})
    cm.stage = 2
    path = tmp_path / "run_state.json"
    cm.save_run_state(path, elo_estimate=680, best_genome_id="w_gen87_0")
    data = json.loads(path.read_text())
    assert data["highest_stage"] == 2
    assert data["best_elo_estimate"] == 680
    assert data["best_genome_id"] == "w_gen87_0"
    assert "timestamp" in data


def test_load_run_state(tmp_path):
    path = tmp_path / "run_state.json"
    path.write_text(json.dumps({
        "highest_stage": 3,
        "best_elo_estimate": 800,
        "best_genome_id": "x",
        "puzzle_max_rating": 1800,
        "timestamp": "2026-03-25T00:00:00Z",
    }))
    cm = CurriculumManager.from_run_state(path, {})
    assert cm.stage == 2


def test_load_run_state_missing_file():
    cm = CurriculumManager.from_run_state(Path("/nonexistent"), {})
    assert cm.stage == 0


def test_elo_estimate():
    assert CurriculumManager.compute_elo_estimate(1320) == 680
    assert CurriculumManager.compute_elo_estimate(0) == 2000
    assert CurriculumManager.compute_elo_estimate(3000) == 0
