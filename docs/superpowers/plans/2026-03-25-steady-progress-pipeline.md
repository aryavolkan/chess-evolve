# Steady Progress Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 5-stage NEAT curriculum pipeline where every overnight run produces measurable Elo improvement toward 1200+.

**Architecture:** Extract curriculum logic (stage definitions, temperature schedules, fitness weight tables, transition blending) into a new `python/curriculum.py` module. Modify `neat_cpu_trainer.py` to delegate stage management to it. Update `fitness.py` to accept stage-dependent weights. Add cross-run seeding via `run_state.json`.

**Tech Stack:** Python 3.10+, Rust (chess-cpu crate via PyO3), pytest, W&B

**Spec:** `docs/superpowers/specs/2026-03-25-steady-progress-pipeline-design.md`

**Note — Rust crate changes deferred:** Stages 1 (guided play heuristic opponent) and 2 (HoF opponent ladder) require changes to the `rust/chess-cpu` crate. This plan builds all Python-side infrastructure so that stages 1 and 2 work with the existing random opponent logic. A follow-up plan will add the Rust-side opponent modes. The curriculum, temperature, fitness weights, benchmarking, and cross-run seeding are all functional without the Rust changes.

---

### Task 1: Create `python/curriculum.py` — Stage Definitions & Temperature Schedule

**Files:**
- Create: `python/curriculum.py`
- Test: `tests/python/test_curriculum.py`

- [ ] **Step 1: Write failing tests for stage definitions and temperature**

```python
# tests/python/test_curriculum.py
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
    # At expected_stage_length (15), should reach end temp
    assert cm.training_temperature(gen_in_stage=15) == pytest.approx(0.15)
    # Midpoint
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/aryasen/projects/chess-evolve && python -m pytest tests/python/test_curriculum.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'curriculum'`

- [ ] **Step 3: Implement `python/curriculum.py`**

```python
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
        # Transition blending
        self.transition_gens: int = config.get("transition_gens", 3)
        self._transition_remaining: int = 0
        self._prev_stage: int | None = None
        # Exit criteria tracking
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/aryasen/projects/chess-evolve && python -m pytest tests/python/test_curriculum.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/aryasen/projects/chess-evolve
git add python/curriculum.py tests/python/test_curriculum.py
git commit -m "feat: add curriculum module with stage definitions and temperature schedule"
```

---

### Task 2: Add Stage-Specific Fitness Weights to `python/curriculum.py`

**Files:**
- Modify: `python/curriculum.py`
- Test: `tests/python/test_curriculum.py`

- [ ] **Step 1: Write failing tests for fitness weight lookup**

Append to `tests/python/test_curriculum.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/aryasen/projects/chess-evolve && python -m pytest tests/python/test_curriculum.py::test_stage_weights_stage0_high_checkmate -v`
Expected: FAIL — `AttributeError: 'CurriculumManager' object has no attribute 'fitness_weights'`

- [ ] **Step 3: Add fitness weight tables to `python/curriculum.py`**

Add after the `STAGES` list:

```python
# Per-stage fitness weight overrides (keys match fitness.FITNESS_DEFAULTS)
STAGE_WEIGHTS = [
    # Stage 0: Puzzles — tactics focus
    {
        "checkmate_bonus": 15.0, "material_weight": 0.5,
        "mobility_weight": 0.0, "king_danger_weight": 0.0,
        "draw_bonus": 0.0, "sf_fitness_weight": 0.0,
    },
    # Stage 1: Guided Play — don't blunder pieces
    {
        "material_weight": 1.5, "mobility_weight": 0.5,
        "king_danger_weight": 0.5, "sf_fitness_weight": 0.0,
    },
    # Stage 2: Opponent Ladder — balanced
    {
        "material_weight": 1.0, "mobility_weight": 0.3,
        "king_danger_weight": 1.0, "sf_fitness_weight": 0.0,
    },
    # Stage 3: SF Shaping — CPL ramps 0.2→0.5 over 20 gens
    {
        "material_weight": 0.5, "mobility_weight": 0.1,
        "king_danger_weight": 0.5,
        # sf_fitness_weight handled dynamically
    },
    # Stage 4: Coevo Refinement — balanced with light SF validation
    {
        "material_weight": 1.0, "mobility_weight": 0.3,
        "king_danger_weight": 1.0, "sf_fitness_weight": 0.1,
    },
]
```

Add method to `CurriculumManager`:

```python
    def fitness_weights(self) -> dict:
        """Get fitness weights for current stage, merged with defaults."""
        from fitness import FITNESS_DEFAULTS
        weights = dict(FITNESS_DEFAULTS)
        weights.update(STAGE_WEIGHTS[self.stage])
        # Stage 3: SF weight ramps linearly 0.2→0.5 over 20 gens
        if self.stage == 3:
            progress = min(1.0, self.gen_in_stage / 20.0)
            weights["sf_fitness_weight"] = 0.2 + 0.3 * progress
        return weights
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/aryasen/projects/chess-evolve && python -m pytest tests/python/test_curriculum.py -v`
Expected: All 12 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/aryasen/projects/chess-evolve
git add python/curriculum.py tests/python/test_curriculum.py
git commit -m "feat: add stage-specific fitness weight tables to curriculum"
```

---

### Task 3: Add Stage Transition Logic to `CurriculumManager`

**Files:**
- Modify: `python/curriculum.py`
- Test: `tests/python/test_curriculum.py`

- [ ] **Step 1: Write failing tests for exit criteria checking**

Append to `tests/python/test_curriculum.py`:

```python
def test_check_stage0_exit_not_ready():
    cm = CurriculumManager({})
    cm.stage = 0
    metrics = {"puzzle_bench_accuracy": 0.50, "puzzle_max_rating": 1200}
    assert cm.check_exit(metrics) is False


def test_check_stage0_exit_ready():
    cm = CurriculumManager({})
    cm.stage = 0
    metrics = {"puzzle_bench_accuracy": 0.65, "puzzle_max_rating": 1500}
    assert cm.check_exit(metrics) is True


def test_check_stage1_needs_consecutive():
    cm = CurriculumManager({})
    cm.stage = 1
    # First high bench — not enough
    assert cm.check_exit({"bench_avg_win_rate": 0.75}) is False
    assert cm.check_exit({"bench_avg_win_rate": 0.75}) is False
    # Third consecutive — exit
    assert cm.check_exit({"bench_avg_win_rate": 0.75}) is True


def test_check_stage1_resets_on_low():
    cm = CurriculumManager({})
    cm.stage = 1
    cm.check_exit({"bench_avg_win_rate": 0.75})
    cm.check_exit({"bench_avg_win_rate": 0.75})
    # Dip resets counter
    cm.check_exit({"bench_avg_win_rate": 0.50})
    assert cm.check_exit({"bench_avg_win_rate": 0.75}) is False


def test_check_stage2_exit():
    cm = CurriculumManager({})
    cm.stage = 2
    metrics = {"bench_avg_win_rate": 0.90, "sf_avg_cpl": 700}
    assert cm.check_exit(metrics) is True


def test_check_stage2_no_exit_high_cpl():
    cm = CurriculumManager({})
    cm.stage = 2
    metrics = {"bench_avg_win_rate": 0.90, "sf_avg_cpl": 900}
    assert cm.check_exit(metrics) is False


def test_check_stage3_needs_5_consecutive():
    cm = CurriculumManager({})
    cm.stage = 3
    for _ in range(4):
        assert cm.check_exit({"sf_avg_cpl": 350}) is False
    assert cm.check_exit({"sf_avg_cpl": 350}) is True


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/aryasen/projects/chess-evolve && python -m pytest tests/python/test_curriculum.py::test_check_stage0_exit_not_ready -v`
Expected: FAIL — `AttributeError: 'CurriculumManager' object has no attribute 'check_exit'`

- [ ] **Step 3: Add `check_exit` to `CurriculumManager`**

Add to `CurriculumManager` in `python/curriculum.py`:

```python
    def check_exit(self, metrics: dict) -> bool:
        """Check if current stage exit criteria are met.

        Args:
            metrics: Dict of current generation metrics.

        Returns:
            True if stage should advance.
        """
        if self.stage == 0:
            acc = metrics.get("puzzle_bench_accuracy", 0.0)
            rating = metrics.get("puzzle_max_rating", 0)
            return acc >= 0.60 and rating >= 1400

        elif self.stage == 1:
            wr = metrics.get("bench_avg_win_rate", 0.0)
            if wr >= 0.70:
                self._consecutive_bench_wins += 1
            else:
                self._consecutive_bench_wins = 0
            return self._consecutive_bench_wins >= 3

        elif self.stage == 2:
            wr = metrics.get("bench_avg_win_rate", 0.0)
            cpl = metrics.get("sf_avg_cpl", 9999)
            return wr >= 0.85 and cpl <= 800

        elif self.stage == 3:
            cpl = metrics.get("sf_avg_cpl", 9999)
            if cpl <= 400:
                self._consecutive_sf_cpl += 1
            else:
                self._consecutive_sf_cpl = 0
            return self._consecutive_sf_cpl >= 5

        else:  # stage 4 — never exits
            return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/aryasen/projects/chess-evolve && python -m pytest tests/python/test_curriculum.py -v`
Expected: All 21 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/aryasen/projects/chess-evolve
git add python/curriculum.py tests/python/test_curriculum.py
git commit -m "feat: add stage exit criteria and transition blending to curriculum"
```

---

### Task 4: Add Cross-Run State Persistence

**Files:**
- Modify: `python/curriculum.py`
- Test: `tests/python/test_curriculum.py`

- [ ] **Step 1: Write failing tests for save/load**

Append to `tests/python/test_curriculum.py`:

```python
import json
import tempfile
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
    # Starts one stage back for robustness
    assert cm.stage == 2


def test_load_run_state_missing_file():
    cm = CurriculumManager.from_run_state(Path("/nonexistent"), {})
    assert cm.stage == 0


def test_elo_estimate():
    assert CurriculumManager.compute_elo_estimate(1320) == 680
    assert CurriculumManager.compute_elo_estimate(0) == 2000
    assert CurriculumManager.compute_elo_estimate(3000) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/aryasen/projects/chess-evolve && python -m pytest tests/python/test_curriculum.py::test_save_run_state -v`
Expected: FAIL — `AttributeError`

- [ ] **Step 3: Implement save/load and elo_estimate**

Add to `CurriculumManager` in `python/curriculum.py`:

```python
    def save_run_state(self, path: Path, elo_estimate: float = 0, best_genome_id: str = ""):
        """Persist run state for cross-run seeding."""
        import datetime
        data = {
            "highest_stage": self.stage,
            "best_elo_estimate": elo_estimate,
            "best_genome_id": best_genome_id,
            "puzzle_max_rating": getattr(self, "_puzzle_max_rating", 0),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def from_run_state(cls, path: Path, config: dict) -> "CurriculumManager":
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
```

Also add `import json` at the top of `curriculum.py` and `from pathlib import Path` to the imports.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/aryasen/projects/chess-evolve && python -m pytest tests/python/test_curriculum.py -v`
Expected: All 25 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/aryasen/projects/chess-evolve
git add python/curriculum.py tests/python/test_curriculum.py
git commit -m "feat: add cross-run state persistence and elo estimate to curriculum"
```

---

### Task 5: Integrate CurriculumManager into NeatCPUTrainer

**Files:**
- Modify: `python/neat_cpu_trainer.py`
- Test: `tests/python/test_neat_cpu_trainer.py`

- [ ] **Step 1: Write failing test for curriculum integration**

Append to `tests/python/test_neat_cpu_trainer.py`:

```python
def test_trainer_creates_curriculum_manager():
    config = {"population_size": 10, "curriculum_stage": 0}
    trainer = NeatCPUTrainer(config, Path("/tmp/test_metrics.jsonl"))
    assert hasattr(trainer, "curriculum")
    assert trainer.curriculum.stage == 0


def test_trainer_temperature_from_curriculum():
    config = {"population_size": 10, "curriculum_stage": 1}
    trainer = NeatCPUTrainer(config, Path("/tmp/test_metrics.jsonl"))
    # Stage 1 starts at 0.3
    assert trainer.curriculum.training_temperature(gen_in_stage=0) == pytest.approx(0.3)


def test_trainer_eval_temperature_zero():
    config = {"population_size": 10}
    trainer = NeatCPUTrainer(config, Path("/tmp/test_metrics.jsonl"))
    assert trainer.curriculum.eval_temperature() == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/aryasen/projects/chess-evolve && python -m pytest tests/python/test_neat_cpu_trainer.py::test_trainer_creates_curriculum_manager -v`
Expected: FAIL — `AssertionError: 'curriculum' not found`

- [ ] **Step 3: Wire CurriculumManager into NeatCPUTrainer.__init__**

In `python/neat_cpu_trainer.py`, add import at top:

```python
from curriculum import CurriculumManager
```

In `__init__`, after the line `self.fitness_weights = merge_fitness_weights(config)`, add:

```python
        # Curriculum manager — replaces old curriculum_stage tracking
        self.curriculum = CurriculumManager(config)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/aryasen/projects/chess-evolve && python -m pytest tests/python/test_neat_cpu_trainer.py::test_trainer_creates_curriculum_manager tests/python/test_neat_cpu_trainer.py::test_trainer_temperature_from_curriculum tests/python/test_neat_cpu_trainer.py::test_trainer_eval_temperature_zero -v`
Expected: All 3 PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/aryasen/projects/chess-evolve
git add python/neat_cpu_trainer.py tests/python/test_neat_cpu_trainer.py
git commit -m "feat: wire CurriculumManager into NeatCPUTrainer"
```

---

### Task 6: Replace Old Curriculum Logic in Training Loop

**Files:**
- Modify: `python/neat_cpu_trainer.py`

This is the largest task — replacing the inline `curriculum_stage` variable and promotion logic in the `train()` method with calls to `self.curriculum`. This is a refactor of existing logic, not new behavior.

- [ ] **Step 1: Replace temperature usage in train()**

In the `train()` method, find where `self.temperature` is used in `chess_cpu.simulate_neat_games_batch()` calls. Replace with:

```python
temperature = self.curriculum.training_temperature()
```

And for benchmark evaluations, use:

```python
eval_temp = self.curriculum.eval_temperature()
```

- [ ] **Step 2: Replace fitness weight lookup**

Replace the static `self.fitness_weights` in the train loop with:

```python
weights = self.curriculum.fitness_weights()
```

This needs to be called each generation since weights change per stage.

- [ ] **Step 3: Replace curriculum promotion logic**

Replace the block at lines ~1190-1221 that checks `curriculum_stage == 0` and `use_curriculum` with:

```python
# Check stage exit criteria
if self.curriculum.check_exit(metrics):
    old_name = self.curriculum.stage_name()
    self.curriculum.advance_stage()
    new_name = self.curriculum.stage_name()
    print(f"  Curriculum promotion: {old_name} -> {new_name}")
self.curriculum.tick_generation()
```

- [ ] **Step 4: Replace `curriculum_stage` variable references**

Replace all remaining `curriculum_stage` local variable reads with `self.curriculum.stage`. Remove the `curriculum_stage = self.curriculum_stage` line and the `use_curriculum` variable.

- [ ] **Step 5: Run existing tests to verify no regression**

Run: `cd /Users/aryasen/projects/chess-evolve && python -m pytest tests/python/test_neat_cpu_trainer.py -v`
Expected: All existing tests PASS

- [ ] **Step 6: Commit**

```bash
cd /Users/aryasen/projects/chess-evolve
git add python/neat_cpu_trainer.py
git commit -m "refactor: replace inline curriculum logic with CurriculumManager in train loop"
```

---

### Task 7: Add Transition Blending to Fitness Computation

**Files:**
- Modify: `python/neat_cpu_trainer.py`

- [ ] **Step 1: Add blending logic after fitness computation**

In the training loop, after computing fitness for the current stage, add blending during transition periods:

```python
if self.curriculum.is_transitioning():
    blend_w = self.curriculum.transition_blend_weight()
    # Compute fitness with previous stage weights too
    prev_weights = self.curriculum.fitness_weights_for_stage(self.curriculum._prev_stage)
    prev_w_fit = compute_fitness(results, self.pop_size, 0, prev_weights)
    prev_b_fit = compute_fitness(results, self.pop_size, 1, prev_weights)
    white_fitness = blend_fitness(prev_w_fit, white_fitness, blend_w)
    black_fitness = blend_fitness(prev_b_fit, black_fitness, blend_w)
```

- [ ] **Step 2: Add `fitness_weights_for_stage` to CurriculumManager**

In `python/curriculum.py`, add:

```python
    def fitness_weights_for_stage(self, stage: int) -> dict:
        """Get fitness weights for a specific stage (used for transition blending)."""
        from fitness import FITNESS_DEFAULTS
        weights = dict(FITNESS_DEFAULTS)
        weights.update(STAGE_WEIGHTS[stage])
        if stage == 3:
            progress = min(1.0, self.gen_in_stage / 20.0)
            weights["sf_fitness_weight"] = 0.2 + 0.3 * progress
        return weights
```

- [ ] **Step 3: Run tests**

Run: `cd /Users/aryasen/projects/chess-evolve && python -m pytest tests/python/ -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
cd /Users/aryasen/projects/chess-evolve
git add python/curriculum.py python/neat_cpu_trainer.py
git commit -m "feat: add transition blending between curriculum stages"
```

---

### Task 8: Expand Benchmark Population

**Files:**
- Modify: `python/neat_cpu_trainer.py`
- Modify: `train_wandb.py`

- [ ] **Step 1: Increase benchmark size**

In `python/neat_cpu_trainer.py`, change line 139:

```python
# Before:
self.benchmark_size = 20
# After:
self.benchmark_size = config.get("benchmark_size", 50)
```

- [ ] **Step 2: Add `elo_estimate` to metrics logging**

In the metrics dict construction in `train()` (~line 1100-1160), add:

```python
# Elo estimate from SF CPL
if ran_sf:
    avg_cpl = (sf_w_cpl + sf_b_cpl) / 2
    metrics["elo_estimate"] = CurriculumManager.compute_elo_estimate(avg_cpl)
```

- [ ] **Step 3: Add `elo_estimate` to CHESS_LOG_KEYS in `train_wandb.py`**

In `train_wandb.py`, add to the `CHESS_LOG_KEYS` list:

```python
    "elo_estimate",
```

- [ ] **Step 4: Add `benchmark_size` to DEFAULT_CONFIG in `train_wandb.py`**

```python
    "benchmark_size": 50,
    "eval_temperature": 0.0,
```

- [ ] **Step 5: Run tests**

Run: `cd /Users/aryasen/projects/chess-evolve && python -m pytest tests/python/ -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
cd /Users/aryasen/projects/chess-evolve
git add python/neat_cpu_trainer.py train_wandb.py
git commit -m "feat: expand benchmark to 50 genomes, add elo_estimate metric"
```

---

### Task 9: Add Run Summary and State Persistence

**Files:**
- Modify: `python/neat_cpu_trainer.py`

- [ ] **Step 1: Add run summary at end of train()**

Before the `return last_metrics` line at the end of `train()`, add:

```python
        # Save cross-run state
        run_state_path = self.save_genome_path.parent / "run_state.json"
        elo = last_metrics.get("elo_estimate", 0)
        self.curriculum.save_run_state(
            run_state_path,
            elo_estimate=elo,
            best_genome_id=f"gen{max_generations}",
        )

        # Print run summary
        print(f"\n  Run summary:")
        print(f"    End stage: {self.curriculum.stage} ({self.curriculum.stage_name()})")
        print(f"    Elo estimate: {elo}")
        print(f"    Benchmark win rate: {last_metrics.get('bench_avg_win_rate', 0):.1%}")
        print(f"    Generations: {max_generations}")
```

- [ ] **Step 2: Load run state at start of train()**

In `__init__`, after creating the CurriculumManager, add cross-run state loading:

```python
        # Load previous run state for seeding
        run_state_path = self.save_genome_path.parent / "run_state.json"
        if run_state_path.exists() and self.curriculum.stage == 0:
            self.curriculum = CurriculumManager.from_run_state(run_state_path, config)
            print(f"  Loaded run state: starting at stage {self.curriculum.stage} "
                  f"({self.curriculum.stage_name()})")
```

- [ ] **Step 3: Run tests**

Run: `cd /Users/aryasen/projects/chess-evolve && python -m pytest tests/python/ -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
cd /Users/aryasen/projects/chess-evolve
git add python/neat_cpu_trainer.py
git commit -m "feat: add run summary and cross-run state persistence"
```

---

### Task 10: Create Steady Progress Config

**Files:**
- Create: `configs/steady_progress_config.json`

- [ ] **Step 1: Create the config file**

```json
{
    "population_size": 100,
    "max_generations": 100,
    "max_moves_per_game": 100,
    "input_size": 389,
    "output_size": 384,
    "use_neat": true,
    "curriculum_stage": 0,
    "move_temperature": 0.3,
    "eval_temperature": 0.0,
    "benchmark_size": 50,
    "tournament_opponents": 5,
    "tournament_k": 2,
    "neat_add_node_rate": 0.15,
    "neat_add_connection_rate": 0.25,
    "neat_initial_connections_per_output": 1,
    "neat_target_species_count": 5,
    "sf_fitness_weight": 0.0,
    "sf_fitness_interval": 1,
    "sf_fitness_top_n": 50,
    "sf_bench_interval": 5,
    "puzzle_advance_threshold": 0.75,
    "puzzle_accuracy_weight": 0.4,
    "transition_gens": 3
}
```

- [ ] **Step 2: Commit**

```bash
cd /Users/aryasen/projects/chess-evolve
git add configs/steady_progress_config.json
git commit -m "feat: add steady progress training config"
```

---

### Task 11: Integration Test

**Files:**
- Create: `tests/integration/test_curriculum_integration.py`

- [ ] **Step 1: Write integration test**

```python
"""Integration test: verify curriculum stage transitions fire correctly
with a micro training run (mocked Rust backends)."""
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
```

- [ ] **Step 2: Run integration test**

Run: `cd /Users/aryasen/projects/chess-evolve && python -m pytest tests/integration/test_curriculum_integration.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
cd /Users/aryasen/projects/chess-evolve
git add tests/integration/test_curriculum_integration.py
git commit -m "test: add curriculum integration test"
```

---

### Task 12: Update CLAUDE.md and Documentation

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/superpowers/specs/2026-03-25-steady-progress-pipeline-design.md` (mark as implemented)

- [ ] **Step 1: Add curriculum module to CLAUDE.md architecture section**

In the Python layer description in CLAUDE.md, add:

```
   - `python/curriculum.py`: 5-stage curriculum manager (stage definitions, temperature annealing, fitness weight tables, cross-run state persistence). Stages: puzzles → guided play → opponent ladder → SF shaping → coevo refinement.
```

- [ ] **Step 2: Add `configs/steady_progress_config.json` mention**

In the CLAUDE.md training section, add:

```
# Steady progress pipeline (5-stage curriculum)
python train_wandb.py --config configs/steady_progress_config.json
```

- [ ] **Step 3: Commit**

```bash
cd /Users/aryasen/projects/chess-evolve
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with curriculum module and steady progress config"
```
