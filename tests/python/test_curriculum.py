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
