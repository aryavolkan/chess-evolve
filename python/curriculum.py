"""Curriculum stage management for NEAT chess training.

Defines 5 training stages, temperature schedules, and stage transition logic.
"""
from __future__ import annotations

STAGES = [
    {
        "name": "puzzles",
        "temp_start": 0.3, "temp_end": 0.3,
        "expected_length": 15,
    },
    {
        "name": "guided_play",
        "temp_start": 0.3, "temp_end": 0.15,
        "expected_length": 15,
    },
    {
        "name": "opponent_ladder",
        "temp_start": 0.15, "temp_end": 0.05,
        "expected_length": 15,
    },
    {
        "name": "sf_shaping",
        "temp_start": 0.05, "temp_end": 0.05,
        "expected_length": 20,
    },
    {
        "name": "coevo_refinement",
        "temp_start": 0.05, "temp_end": 0.05,
        "expected_length": 50,
    },
]


class CurriculumManager:
    """Manages curriculum stage, temperature schedule, and stage transitions."""

    def __init__(self, config: dict):
        self.stage: int = config.get("curriculum_stage", 0)
        self.gen_in_stage: int = 0
        self._eval_temp: float = config.get("eval_temperature", 0.0)
        self.transition_gens: int = config.get("transition_gens", 3)
        self._transition_remaining: int = 0
        self._prev_stage: int | None = None
        self._consecutive_bench_wins: int = 0
        self._consecutive_sf_cpl: int = 0
        self._ladder_tier: int = 0

    def training_temperature(self, gen_in_stage: int | None = None) -> float:
        """Compute training temperature for current stage and generation."""
        if gen_in_stage is None:
            gen_in_stage = self.gen_in_stage
        s = STAGES[self.stage]
        t_start, t_end = s["temp_start"], s["temp_end"]
        if t_start == t_end:
            return t_start
        progress = min(1.0, gen_in_stage / max(1, s["expected_length"]))
        return t_start - (t_start - t_end) * progress

    def eval_temperature(self) -> float:
        """Evaluation temperature — always deterministic."""
        return self._eval_temp

    def stage_name(self) -> str:
        return STAGES[self.stage]["name"]

    def is_transitioning(self) -> bool:
        return self._transition_remaining > 0

    def transition_blend_weight(self) -> float:
        """Returns weight for new stage fitness (0.7) vs old stage (0.3)."""
        if self._transition_remaining <= 0:
            return 1.0
        return 0.7

    def advance_stage(self) -> bool:
        """Advance to next stage. Returns True if advanced, False if already at max."""
        if self.stage >= len(STAGES) - 1:
            return False
        self._prev_stage = self.stage
        self.stage += 1
        self.gen_in_stage = 0
        self._transition_remaining = self.transition_gens
        self._consecutive_bench_wins = 0
        self._consecutive_sf_cpl = 0
        self._ladder_tier = 0
        return True

    def tick_generation(self):
        """Call at end of each generation to update internal counters."""
        self.gen_in_stage += 1
        if self._transition_remaining > 0:
            self._transition_remaining -= 1
