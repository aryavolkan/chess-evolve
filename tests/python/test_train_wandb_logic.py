"""Tests for train_wandb.py pure-Python logic.

Tests backend detection, config merging, metric transforms, and
chained training logic without requiring actual W&B or Rust backends.
"""
from __future__ import annotations

import json
import runpy
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Load train_wandb module safely via runpy to avoid heavy imports.
# Track injected mocks so we can clean them up afterwards.
# ---------------------------------------------------------------------------

_injected = []

for mod_name in ("chess_cpu", "evolve_ga", "neat_ga", "torch", "chess"):
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)
        _injected.append(mod_name)

if "wandb" not in sys.modules:
    _wandb_mock = types.SimpleNamespace(
        init=MagicMock(), agent=MagicMock(), define_metric=MagicMock(),
        sweep=MagicMock(return_value="test_sweep_id"),
        Artifact=MagicMock(), log=MagicMock(),
    )
    sys.modules["wandb"] = _wandb_mock
    _injected.append("wandb")

# Mock godot_wandb with required exports
if "godot_wandb" not in sys.modules:
    _godot_wandb_mock = types.ModuleType("godot_wandb")
    _godot_wandb_mock.SweepWorker = MagicMock
    _godot_wandb_mock.define_step_metric = MagicMock()
    _godot_wandb_mock.godot_user_dir = MagicMock(return_value=Path("/tmp/test"))
    _godot_wandb_mock.launch_godot = MagicMock()
    _godot_wandb_mock.log_final_summary = MagicMock()
    _godot_wandb_mock.poll_metrics = MagicMock(return_value={})
    _godot_wandb_mock.run_training = MagicMock()
    _godot_wandb_mock.wait_for_metrics = MagicMock(return_value=True)
    sys.modules["godot_wandb"] = _godot_wandb_mock
    _injected.append("godot_wandb")

# Mock global_elite
if "global_elite" not in sys.modules:
    _global_elite_mock = types.ModuleType("global_elite")
    _global_elite_mock.GlobalElitePool = MagicMock
    sys.modules["global_elite"] = _global_elite_mock
    _injected.append("global_elite")

# Now load the module
_MODULE = runpy.run_path("train_wandb.py", run_name="__train_wandb_test__")

# Clean up mocks so other test files can detect real modules as missing
for _mod in _injected:
    sys.modules.pop(_mod, None)


# ===========================================================================
# _chess_metric_transform
# ===========================================================================

class TestChessMetricTransform:
    def test_computes_combined_best(self):
        transform = _MODULE["_chess_metric_transform"]
        data = {"white_best": 10.0, "black_best": 7.0}
        result = transform(data)
        assert result["combined_best"] == 7.0

    def test_handles_missing_keys(self):
        transform = _MODULE["_chess_metric_transform"]
        data = {}
        result = transform(data)
        assert result["combined_best"] == 0

    def test_symmetric_when_equal(self):
        transform = _MODULE["_chess_metric_transform"]
        data = {"white_best": 5.0, "black_best": 5.0}
        result = transform(data)
        assert result["combined_best"] == 5.0


# ===========================================================================
# DEFAULT_CONFIG
# ===========================================================================

class TestDefaultConfig:
    def test_has_required_keys(self):
        config = _MODULE["DEFAULT_CONFIG"]
        required = [
            "population_size", "hidden_size", "elite_count",
            "crossover_rate", "mutation_rate", "mutation_strength",
            "max_generations", "max_moves_per_game",
        ]
        for key in required:
            assert key in config, f"missing key: {key}"

    def test_population_size_positive(self):
        config = _MODULE["DEFAULT_CONFIG"]
        assert config["population_size"] > 0

    def test_rates_in_valid_range(self):
        config = _MODULE["DEFAULT_CONFIG"]
        for key in ("crossover_rate", "mutation_rate", "mutation_strength", "immigration_rate"):
            assert 0.0 <= config[key] <= 1.0, f"{key} out of range"

    def test_elite_count_less_than_population(self):
        config = _MODULE["DEFAULT_CONFIG"]
        assert config["elite_count"] < config["population_size"]


# ===========================================================================
# CHESS_LOG_KEYS
# ===========================================================================

class TestChessLogKeys:
    def test_is_list(self):
        keys = _MODULE["CHESS_LOG_KEYS"]
        assert isinstance(keys, list)

    def test_contains_core_metrics(self):
        keys = _MODULE["CHESS_LOG_KEYS"]
        core = ["generation", "white_best", "black_best",
                "white_win_rate", "black_win_rate", "combined_best"]
        for k in core:
            assert k in keys, f"missing core key: {k}"

    def test_contains_benchmark_keys(self):
        keys = _MODULE["CHESS_LOG_KEYS"]
        bench = ["bench_white_win_rate", "bench_black_win_rate"]
        for k in bench:
            assert k in keys

    def test_no_duplicates(self):
        keys = _MODULE["CHESS_LOG_KEYS"]
        assert len(keys) == len(set(keys)), "duplicate keys found"


# ===========================================================================
# Backend detection functions
# ===========================================================================

class TestBackendDetection:
    def test_detect_rust_returns_bool(self):
        fn = _MODULE["_detect_rust"]
        assert isinstance(fn(), bool)

    def test_detect_neat_rust_returns_bool(self):
        fn = _MODULE["_detect_neat_rust"]
        assert isinstance(fn(), bool)

    def test_detect_pytorch_returns_bool(self):
        fn = _MODULE["_detect_pytorch"]
        assert isinstance(fn(), bool)


# ===========================================================================
# do_chained_training
# ===========================================================================

class TestDoChainedTraining:
    def test_stops_on_no_improvement(self, tmp_path):
        """Chained training should stop when benchmark doesn't improve."""
        # do_chained_training calls do_training by closure, so we must
        # patch the actual function reference in the module globals dict.
        do_chained = _MODULE["do_chained_training"]
        call_count = 0
        _globals = do_chained.__globals__

        def fake_do_training(config=None, visible=False):
            nonlocal call_count
            call_count += 1
            save_path = Path(config.get("save_genome_path", ""))
            save_path.write_text(json.dumps({
                "white": "w", "black": "b",
                "bench_avg_win_rate": 0.05,
            }))

        original = _globals["do_training"]
        _globals["do_training"] = fake_do_training
        try:
            do_chained(
                config={"save_genome_path": str(tmp_path / "best.json")},
                max_chains=5,
            )
        finally:
            _globals["do_training"] = original
        # First run: 0.05 > 0.0 + 0.01 → improvement, continues
        # Second run: 0.05 <= 0.05 + 0.01 → no improvement, stops
        assert call_count == 2

    def test_stops_when_no_save_file_produced(self, tmp_path):
        do_chained = _MODULE["do_chained_training"]
        call_count = 0
        _globals = do_chained.__globals__

        def fake_do_training(config=None, visible=False):
            nonlocal call_count
            call_count += 1

        original = _globals["do_training"]
        _globals["do_training"] = fake_do_training
        try:
            do_chained(
                config={"save_genome_path": str(tmp_path / "best.json")},
                max_chains=5,
            )
        finally:
            _globals["do_training"] = original
        assert call_count == 1


# ===========================================================================
# _harvest_godot_elites (Godot-only path)
# ===========================================================================

class TestHarvestGodotElites:
    def test_no_contrib_file_noop(self, tmp_path):
        """When no elite_contrib file exists, nothing should happen."""
        harvest = _MODULE["_harvest_godot_elites"]
        _globals = harvest.__globals__
        orig_worker = _globals.get("_worker")
        orig_user_dir = _globals.get("USER_DIR")
        mock_worker = type("W", (), {"worker_id": "test_w1"})()
        _globals["_worker"] = mock_worker
        _globals["USER_DIR"] = str(tmp_path)

        mock_run = MagicMock()
        mock_pool = MagicMock()
        try:
            harvest(mock_run, mock_pool)
        finally:
            _globals["_worker"] = orig_worker
            _globals["USER_DIR"] = orig_user_dir

        mock_pool.update_contrib.assert_not_called()

    def test_harvests_valid_elites(self, tmp_path):
        """When elite_contrib file exists with valid genomes, update pool."""
        harvest = _MODULE["_harvest_godot_elites"]
        _globals = harvest.__globals__
        orig_worker = _globals.get("_worker")
        orig_user_dir = _globals.get("USER_DIR")
        orig_wandb = _globals.get("wandb")
        mock_worker = type("W", (), {"worker_id": "test_w2"})()
        _globals["_worker"] = mock_worker
        _globals["USER_DIR"] = str(tmp_path)
        mock_wandb = types.SimpleNamespace(Artifact=MagicMock())
        _globals["wandb"] = mock_wandb

        contrib = tmp_path / "elite_contrib_test_w2.json"
        contrib.write_text(json.dumps({
            "elites": [{"fitness": 5.0}, {"fitness": 3.0}],
        }))

        mock_run = MagicMock()
        mock_pool = MagicMock()
        mock_pool.update_contrib = MagicMock(return_value=2)

        try:
            harvest(mock_run, mock_pool)
        finally:
            _globals["_worker"] = orig_worker
            _globals["USER_DIR"] = orig_user_dir
            _globals["wandb"] = orig_wandb

        mock_pool.update_contrib.assert_called_once_with(
            "test_w2", [{"fitness": 5.0}, {"fitness": 3.0}],
        )

    def test_handles_corrupt_json(self, tmp_path):
        """Corrupt contrib file should not crash."""
        harvest = _MODULE["_harvest_godot_elites"]
        _globals = harvest.__globals__
        orig_worker = _globals.get("_worker")
        orig_user_dir = _globals.get("USER_DIR")
        mock_worker = type("W", (), {"worker_id": "test_w3"})()
        _globals["_worker"] = mock_worker
        _globals["USER_DIR"] = str(tmp_path)

        contrib = tmp_path / "elite_contrib_test_w3.json"
        contrib.write_text("{bad json")

        mock_run = MagicMock()
        mock_pool = MagicMock()

        try:
            harvest(mock_run, mock_pool)  # should not raise
        finally:
            _globals["_worker"] = orig_worker
            _globals["USER_DIR"] = orig_user_dir

        mock_pool.update_contrib.assert_not_called()
