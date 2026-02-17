import json
import runpy
import sys
import types
from pathlib import Path

sys.modules.setdefault("wandb", types.SimpleNamespace())
MODULE = runpy.run_path("scripts/wandb_bridge.py")
read_metrics = MODULE["read_metrics"]


def test_read_metrics_returns_none_for_missing_file(tmp_path: Path) -> None:
    assert read_metrics(tmp_path / "does-not-exist.json") is None


def test_read_metrics_returns_none_for_invalid_json(tmp_path: Path) -> None:
    bad_file = tmp_path / "metrics.json"
    bad_file.write_text("{oops}", encoding="utf-8")

    assert read_metrics(bad_file) is None


def test_read_metrics_loads_valid_metrics(tmp_path: Path) -> None:
    metrics_file = tmp_path / "metrics.json"
    payload = {"generation": 12, "white_best": 4.5}
    metrics_file.write_text(json.dumps(payload), encoding="utf-8")

    assert read_metrics(metrics_file) == payload
