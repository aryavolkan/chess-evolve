"""Tests for godot_wandb.py shared utilities.

Tests platform-specific path resolution, config/metrics I/O, SweepWorker
lifecycle, and derived metric computation.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

# Mock wandb before import to avoid ImportError
sys.modules.setdefault("wandb", types.SimpleNamespace(
    init=lambda **kw: None, define_metric=lambda *a, **kw: None,
))

from godot_wandb import (  # noqa: E402
    SweepWorker,
    calc_training_timeout,
    compute_derived_metrics,
    godot_user_dir,
    launch_godot,
    log_final_summary,
    read_metrics,
    wait_for_metrics,
    write_config,
)


# ===========================================================================
# godot_user_dir
# ===========================================================================

class TestGodotUserDir:
    def test_linux_path(self):
        with patch("godot_wandb.platform.system", return_value="Linux"), \
             patch.dict("os.environ", {}, clear=False):
            # Remove override env var if present
            import os
            env = os.environ.copy()
            env.pop("GODOT_USER_DIR", None)
            with patch.dict("os.environ", env, clear=True):
                path = godot_user_dir("Chess Evolve")
                assert ".local/share/godot/app_userdata/Chess Evolve" in str(path)

    def test_darwin_path(self):
        with patch("godot_wandb.platform.system", return_value="Darwin"), \
             patch.dict("os.environ", {}, clear=False):
            import os
            env = os.environ.copy()
            env.pop("GODOT_USER_DIR", None)
            with patch.dict("os.environ", env, clear=True):
                path = godot_user_dir("Chess Evolve")
                assert "Library/Application Support/Godot/app_userdata/Chess Evolve" in str(path)

    def test_override_env_var(self):
        with patch.dict("os.environ", {"GODOT_USER_DIR": "/custom/path"}):
            path = godot_user_dir("Chess Evolve")
            assert str(path) == "/custom/path"


# ===========================================================================
# write_config
# ===========================================================================

class TestWriteConfig:
    def test_creates_file(self, tmp_path):
        config = {"population_size": 30, "mutation_rate": 0.1}
        config_path = tmp_path / "subdir" / "config.json"
        write_config(config, config_path)
        assert config_path.exists()
        assert json.loads(config_path.read_text()) == config

    def test_creates_parent_dirs(self, tmp_path):
        config_path = tmp_path / "a" / "b" / "c" / "config.json"
        write_config({"key": "value"}, config_path)
        assert config_path.exists()


# ===========================================================================
# read_metrics
# ===========================================================================

class TestReadMetrics:
    def test_returns_none_for_missing(self, tmp_path):
        assert read_metrics(tmp_path / "nope.json") is None

    def test_returns_none_for_invalid_json(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{bad json")
        assert read_metrics(p) is None

    def test_reads_valid_json(self, tmp_path):
        p = tmp_path / "metrics.json"
        data = {"generation": 5, "best_fitness": 3.14}
        p.write_text(json.dumps(data))
        assert read_metrics(p) == data


# ===========================================================================
# SweepWorker
# ===========================================================================

class TestSweepWorker:
    def test_worker_id_generated(self, tmp_path):
        w = SweepWorker(tmp_path)
        assert len(w.worker_id) == 8

    def test_custom_worker_id(self, tmp_path):
        w = SweepWorker(tmp_path, worker_id="custom42")
        assert w.worker_id == "custom42"

    def test_paths_use_worker_id(self, tmp_path):
        w = SweepWorker(tmp_path, worker_id="abc")
        assert "abc" in str(w.config_path)
        assert "abc" in str(w.metrics_path)

    def test_write_config(self, tmp_path):
        w = SweepWorker(tmp_path, worker_id="w1")
        w.write_config({"key": "value"})
        assert w.config_path.exists()
        assert json.loads(w.config_path.read_text()) == {"key": "value"}

    def test_clear_metrics(self, tmp_path):
        w = SweepWorker(tmp_path, worker_id="w1")
        w.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        w.metrics_path.write_text("{}")
        jsonl = Path(str(w.metrics_path) + "l")
        jsonl.write_text("")
        w.clear_metrics()
        assert not w.metrics_path.exists()
        assert not jsonl.exists()

    def test_cleanup(self, tmp_path):
        w = SweepWorker(tmp_path, worker_id="w1")
        w.write_config({"key": "value"})
        w.metrics_path.write_text("{}")
        w.cleanup()
        assert not w.config_path.exists()
        assert not w.metrics_path.exists()

    def test_cleanup_noop_when_no_files(self, tmp_path):
        w = SweepWorker(tmp_path, worker_id="w1")
        w.cleanup()  # should not raise


# ===========================================================================
# compute_derived_metrics
# ===========================================================================

class TestComputeDerivedMetrics:
    def test_empty_history(self):
        assert compute_derived_metrics([], []) == {}

    def test_single_generation(self):
        result = compute_derived_metrics([5.0], [3.0])
        assert result["mean_best_fitness"] == 5.0
        assert result["max_best_fitness"] == 5.0
        assert result["improvement_rate"] == 0.0

    def test_improvement_rate(self):
        best = [1.0, 2.0, 3.0, 4.0, 5.0]
        avg = [0.5, 1.0, 1.5, 2.0, 2.5]
        result = compute_derived_metrics(best, avg)
        assert result["improvement_rate"] == pytest.approx(0.8, abs=0.01)
        assert result["max_best_fitness"] == 5.0

    def test_std_dev_computed(self):
        best = [10.0, 10.0]
        avg = [5.0, 15.0]
        result = compute_derived_metrics(best, avg)
        assert result["fitness_std_dev"] == pytest.approx(5.0, abs=0.01)


# ===========================================================================
# calc_training_timeout
# ===========================================================================

class TestCalcTrainingTimeout:
    def test_basic_calculation(self):
        # 10 gens * (30 * 5 / 4) * (5/60) = 10 * 37.5 * 0.0833 = ~31.25 → 32
        result = calc_training_timeout(
            population_size=30, evals_per_individual=5,
            parallel_count=4, max_generations=10,
        )
        assert result > 0
        assert isinstance(result, int)

    def test_zero_parallel_count(self):
        result = calc_training_timeout(
            population_size=10, evals_per_individual=2,
            parallel_count=0, max_generations=5,
        )
        assert result > 0  # should use max(1, 0)


# ===========================================================================
# log_final_summary
# ===========================================================================

class TestLogFinalSummary:
    def test_empty_metrics_noop(self):
        run = type("FakeRun", (), {"summary": {}})()
        log_final_summary(run, {})
        assert run.summary == {}

    def test_none_metrics_noop(self):
        run = type("FakeRun", (), {"summary": {}})()
        log_final_summary(run, None)
        assert run.summary == {}

    def test_writes_final_prefix(self):
        run = type("FakeRun", (), {"summary": {}})()
        metrics = {"generation": 10, "best_fitness": 5.5, "name": "test"}
        log_final_summary(run, metrics)
        assert run.summary["final_generation"] == 10
        assert run.summary["final_best_fitness"] == 5.5
        assert "final_name" not in run.summary  # strings excluded

    def test_custom_key_map(self):
        run = type("FakeRun", (), {"summary": {}})()
        metrics = {"best_fitness": 5.5, "avg_fitness": 3.0}
        log_final_summary(run, metrics, key_map={"best_fitness": "top_score"})
        assert run.summary["top_score"] == 5.5
        assert "final_avg_fitness" not in run.summary  # key_map overrides


# ===========================================================================
# wait_for_metrics
# ===========================================================================

class TestWaitForMetrics:
    def test_returns_true_when_file_exists(self, tmp_path):
        metrics = tmp_path / "metrics.json"
        metrics.write_text("{}")
        assert wait_for_metrics(metrics, timeout=1.0) is True

    def test_returns_false_on_timeout(self, tmp_path):
        metrics = tmp_path / "metrics.json"
        assert wait_for_metrics(metrics, timeout=0.1) is False


# ===========================================================================
# launch_godot
# ===========================================================================

class TestLaunchGodot:
    def test_builds_headless_command(self):
        with patch("godot_wandb.subprocess.Popen") as mock_popen:
            mock_popen.return_value = type("P", (), {"pid": 123})()
            proc = launch_godot("/path/to/project", godot_path="/usr/bin/godot")
            cmd = mock_popen.call_args[0][0]
            assert "/usr/bin/godot" in cmd
            assert "--headless" in cmd
            assert "--path" in cmd
            assert "--auto-train" in cmd

    def test_visible_mode_no_headless(self):
        with patch("godot_wandb.subprocess.Popen") as mock_popen:
            mock_popen.return_value = type("P", (), {"pid": 123})()
            launch_godot("/path/to/project", godot_path="/usr/bin/godot", visible=True)
            cmd = mock_popen.call_args[0][0]
            assert "--headless" not in cmd

    def test_worker_id_passed(self):
        with patch("godot_wandb.subprocess.Popen") as mock_popen:
            mock_popen.return_value = type("P", (), {"pid": 123})()
            launch_godot("/path", godot_path="/usr/bin/godot", worker_id="w123")
            cmd = mock_popen.call_args[0][0]
            assert "--worker-id=w123" in cmd

    def test_clears_old_metrics(self, tmp_path):
        metrics = tmp_path / "metrics.json"
        metrics.write_text("{}")
        with patch("godot_wandb.subprocess.Popen") as mock_popen:
            mock_popen.return_value = type("P", (), {"pid": 123})()
            launch_godot("/path", godot_path="/usr/bin/godot", metrics_path=metrics)
            assert not metrics.exists()
