"""Curriculum stage management for NEAT chess training.

Defines 5 training stages, temperature schedules, and stage transition logic.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

from fitness import FITNESS_DEFAULTS

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


STAGE_WEIGHTS = [
    # Stage 0: Puzzles — tactics focus
    {"checkmate_bonus": 15.0, "material_weight": 0.5, "mobility_weight": 0.0, "king_danger_weight": 0.0, "draw_bonus": 0.0, "sf_fitness_weight": 0.0},
    # Stage 1: Guided Play — don't blunder pieces
    {"material_weight": 1.5, "mobility_weight": 0.5, "king_danger_weight": 0.5, "sf_fitness_weight": 0.0},
    # Stage 2: Opponent Ladder — balanced
    {"material_weight": 1.0, "mobility_weight": 0.3, "king_danger_weight": 1.0, "sf_fitness_weight": 0.0},
    # Stage 3: SF Shaping — CPL ramps 0.2→0.5 over 20 gens
    {"material_weight": 0.5, "mobility_weight": 0.1, "king_danger_weight": 0.5},
    # Stage 4: Coevo Refinement — balanced with light SF validation
    {"material_weight": 1.0, "mobility_weight": 0.3, "king_danger_weight": 1.0, "sf_fitness_weight": 0.1},
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

    def _stage_weights(self, stage: int) -> dict:
        """Get fitness weights for a specific stage, merged with defaults."""
        weights = dict(FITNESS_DEFAULTS)
        weights.update(STAGE_WEIGHTS[stage])
        if stage == 3:
            progress = min(1.0, self.gen_in_stage / 20.0)
            weights["sf_fitness_weight"] = 0.2 + 0.3 * progress
        return weights

    def fitness_weights(self) -> dict:
        """Get fitness weights for current stage, merged with defaults."""
        return self._stage_weights(self.stage)

    def fitness_weights_for_stage(self, stage: int) -> dict:
        """Get fitness weights for a specific stage (used for transition blending)."""
        return self._stage_weights(stage)

    def check_exit(self, metrics: dict) -> bool:
        """Check if current stage exit criteria are met."""
        if self.stage == 0:
            acc = metrics.get("puzzle_bench_accuracy", 0.0)
            rating = metrics.get("puzzle_max_rating", 0)
            return acc >= 0.12 and rating >= 800

        elif self.stage == 1:
            wr = metrics.get("bench_avg_win_rate", 0.0)
            if wr >= 0.15:
                self._consecutive_bench_wins += 1
            else:
                self._consecutive_bench_wins = 0
            return self._consecutive_bench_wins >= 3

        elif self.stage == 2:
            wr = metrics.get("bench_avg_win_rate", 0.0)
            cpl = metrics.get("sf_avg_cpl", 9999)
            return wr >= 0.25 or cpl <= 1200

        elif self.stage == 3:
            cpl = metrics.get("sf_avg_cpl", 9999)
            if cpl <= 800:
                self._consecutive_sf_cpl += 1
            else:
                self._consecutive_sf_cpl = 0
            return self._consecutive_sf_cpl >= 5

        else:
            return False

    def tick_generation(self):
        """Call at end of each generation to update internal counters."""
        self.gen_in_stage += 1
        if self._transition_remaining > 0:
            self._transition_remaining -= 1

    def save_run_state(self, path: Path, elo_estimate: float = 0, best_genome_id: str = ""):
        """Persist run state for cross-run seeding."""
        data = {
            "highest_stage": self.stage,
            "best_elo_estimate": elo_estimate,
            "best_genome_id": best_genome_id,
            "puzzle_max_rating": getattr(self, "_puzzle_max_rating", 0),
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def from_run_state(cls, path: Path, config: dict) -> CurriculumManager:
        """Load from previous run state, starting one stage back."""
        if not path.exists():
            return cls(config)
        try:
            data = json.loads(path.read_text())
            start_stage = max(0, data.get("highest_stage", 0) - 1)
            config_with_stage = {**config, "curriculum_stage": start_stage}
            return cls(config_with_stage)
        except (json.JSONDecodeError, OSError):
            return cls(config)

    @staticmethod
    def compute_elo_estimate(sf_avg_cpl: float) -> int:
        """Rough Elo estimate from Stockfish average centipawn loss."""
        return max(0, int(2000 - sf_avg_cpl))
