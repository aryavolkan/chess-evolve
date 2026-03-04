"""Tests for check_workers.py process detection and health classification.

Mocks subprocess and /proc reads to test parsing logic without
requiring actual running processes.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "overnight-agent"))

from check_workers import (  # noqa: E402
    _metrics_age_minutes,
    _read_metrics,
    get_godot_workers,
    get_sweep_workers,
)


# ===========================================================================
# get_sweep_workers
# ===========================================================================

class TestGetSweepWorkers:
    def test_finds_worker(self):
        ps_output = [
            "USER  PID %CPU %MEM  VSZ  RSS TTY STAT START   TIME COMMAND",
            "user 1234  5.0  2.0  1000  500 ?   S    10:00  0:30 python3 chess_sweep_worker.py --sweep-id abc123",
        ]
        with patch("check_workers._ps_lines", return_value=ps_output):
            workers = get_sweep_workers()
        assert len(workers) == 1
        assert workers[0] == (1234, "abc123")

    def test_ignores_grep(self):
        ps_output = [
            "USER  PID %CPU %MEM  VSZ  RSS TTY STAT START   TIME COMMAND",
            "user 5678  0.0  0.0  100   50 ?   S    10:00  0:00 grep chess_sweep_worker.py",
        ]
        with patch("check_workers._ps_lines", return_value=ps_output):
            workers = get_sweep_workers()
        assert len(workers) == 0

    def test_no_sweep_id(self):
        ps_output = [
            "USER  PID %CPU %MEM  VSZ  RSS TTY STAT START   TIME COMMAND",
            "user 1234  5.0  2.0  1000  500 ?   S    10:00  0:30 python3 chess_sweep_worker.py",
        ]
        with patch("check_workers._ps_lines", return_value=ps_output):
            workers = get_sweep_workers()
        assert len(workers) == 1
        assert workers[0] == (1234, None)

    def test_empty_ps(self):
        with patch("check_workers._ps_lines", return_value=[]):
            workers = get_sweep_workers()
        assert len(workers) == 0


# ===========================================================================
# get_godot_workers
# ===========================================================================

class TestGetGodotWorkers:
    def test_finds_godot_auto_train(self):
        ps_output = [
            "USER  PID %CPU %MEM  VSZ  RSS TTY STAT START   TIME COMMAND",
            "user 2345  50.0  5.0  2000 1000 ?   S    10:00  1:00 /usr/bin/godot --headless -- --auto-train --worker-id=abc12345",
        ]
        with patch("check_workers._ps_lines", return_value=ps_output):
            workers = get_godot_workers()
        assert len(workers) == 1
        assert workers[0] == (2345, "abc12345")

    def test_ignores_non_auto_train_godot(self):
        ps_output = [
            "USER  PID %CPU %MEM  VSZ  RSS TTY STAT START   TIME COMMAND",
            "user 2345  50.0  5.0  2000 1000 ?   S    10:00  1:00 /usr/bin/godot --headless",
        ]
        with patch("check_workers._ps_lines", return_value=ps_output):
            workers = get_godot_workers()
        assert len(workers) == 0

    def test_no_worker_id(self):
        ps_output = [
            "USER  PID %CPU %MEM  VSZ  RSS TTY STAT START   TIME COMMAND",
            "user 2345  50.0  5.0  2000 1000 ?   S    10:00  1:00 /usr/bin/godot --headless -- --auto-train",
        ]
        with patch("check_workers._ps_lines", return_value=ps_output):
            workers = get_godot_workers()
        assert len(workers) == 1
        assert workers[0] == (2345, None)


# ===========================================================================
# _metrics_age_minutes
# ===========================================================================

class TestMetricsAgeMinutes:
    def test_missing_file_returns_inf(self, tmp_path):
        age = _metrics_age_minutes(tmp_path, "nonexistent")
        assert age == float("inf")

    def test_recent_file_returns_small_age(self, tmp_path):
        metrics_file = tmp_path / "metrics_w1.json"
        metrics_file.write_text("{}")
        age = _metrics_age_minutes(tmp_path, "w1")
        assert age < 1.0  # should be nearly zero


# ===========================================================================
# _read_metrics
# ===========================================================================

class TestReadMetrics:
    def test_missing_file_returns_none(self, tmp_path):
        assert _read_metrics(tmp_path, "nonexistent") is None

    def test_invalid_json_returns_none(self, tmp_path):
        p = tmp_path / "metrics_w1.json"
        p.write_text("{invalid")
        assert _read_metrics(tmp_path, "w1") is None

    def test_valid_json(self, tmp_path):
        payload = {"generation": 10, "best_fitness": 5.0}
        p = tmp_path / "metrics_w1.json"
        p.write_text(json.dumps(payload))
        result = _read_metrics(tmp_path, "w1")
        assert result == payload
