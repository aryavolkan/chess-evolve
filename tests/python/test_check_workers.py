"""Tests for check_workers.py process detection and health classification.

Mocks subprocess and /proc reads to test parsing logic without
requiring actual running processes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "overnight-agent"))

from check_workers import (  # noqa: E402
    _cpu_percent,
    _metrics_age_minutes,
    _read_metrics,
    get_godot_workers,
    get_sweep_workers,
    main,
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


# ===========================================================================
# _cpu_percent
# ===========================================================================

class TestCpuPercent:
    def test_returns_none_for_missing_pid(self):
        result = _cpu_percent(999999999)
        assert result is None

    def test_returns_float_or_none_for_self(self):
        import os
        result = _cpu_percent(os.getpid())
        # In sandboxed environments /proc may not exist
        assert result is None or isinstance(result, float)


# ===========================================================================
# main
# ===========================================================================

class TestMain:
    def test_no_workers_returns_zero(self, tmp_path):
        with patch("check_workers.get_sweep_workers", return_value=[]), \
             patch("check_workers.get_godot_workers", return_value=[]), \
             patch("sys.argv", ["check_workers.py", "--metrics-dir", str(tmp_path)]):
            assert main() == 0

    def test_healthy_worker_returns_zero(self, tmp_path):
        metrics = {"generation": 10, "best_fitness": 5.0}
        (tmp_path / "metrics_abc12345.json").write_text(json.dumps(metrics))
        with patch("check_workers.get_sweep_workers", return_value=[]), \
             patch("check_workers.get_godot_workers", return_value=[(1234, "abc12345")]), \
             patch("check_workers._cpu_percent", return_value=50.0), \
             patch("sys.argv", ["check_workers.py", "--metrics-dir", str(tmp_path)]):
            assert main() == 0

    def test_stuck_worker_returns_one(self, tmp_path):
        # Create a stale metrics file (modify time is old)
        p = tmp_path / "metrics_stuck123.json"
        p.write_text(json.dumps({"generation": 1}))
        with patch("check_workers.get_sweep_workers", return_value=[]), \
             patch("check_workers.get_godot_workers", return_value=[(5678, "stuck123")]), \
             patch("check_workers._cpu_percent", return_value=0.0), \
             patch("check_workers._metrics_age_minutes", return_value=30.0), \
             patch("sys.argv", ["check_workers.py", "--metrics-dir", str(tmp_path)]):
            assert main() == 1

    def test_no_metrics_low_cpu_is_stuck(self, tmp_path):
        with patch("check_workers.get_sweep_workers", return_value=[]), \
             patch("check_workers.get_godot_workers", return_value=[(9999, "nometrics")]), \
             patch("check_workers._cpu_percent", return_value=0.0), \
             patch("sys.argv", ["check_workers.py", "--metrics-dir", str(tmp_path)]):
            assert main() == 1

    def test_json_output(self, tmp_path, capsys):
        with patch("check_workers.get_sweep_workers", return_value=[]), \
             patch("check_workers.get_godot_workers", return_value=[]), \
             patch("sys.argv", ["check_workers.py", "--metrics-dir", str(tmp_path), "--json"]):
            main()
        output = capsys.readouterr().out
        data = json.loads(output)
        assert "sweep_workers" in data
        assert "godot_workers" in data
        assert data["healthy"] == []
        assert data["stuck"] == []
