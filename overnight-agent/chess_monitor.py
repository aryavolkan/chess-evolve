#!/usr/bin/env python3
"""
Monitor chess-evolve sweep workers and optionally auto-spawn new ones.

Usage:
    python chess_monitor.py                          # Monitor only
    python chess_monitor.py --notify                 # Monitor + WhatsApp report
    python chess_monitor.py --auto-spawn             # Spawn if idle/none running
    python chess_monitor.py --fill --max-workers 4   # Fill all slots
    python chess_monitor.py --auto-spawn --sweep-id abc123
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_SHARED = next(
    p for p in [
        os.path.expanduser("~/projects/shared-evolve-utils"),
        os.path.expanduser("~/Projects/shared-evolve-utils"),
        os.path.expanduser("~/shared-evolve-utils"),
    ]
    if os.path.isdir(p)
)
sys.path.insert(0, _SHARED)
from worker_monitor import WorkerConfig, add_monitor_args, monitor_once  # noqa: E402

from godot_wandb import godot_user_dir  # noqa: E402

PROJECT_PATH = os.environ.get(
    "CHESS_EVOLVE_PROJECT_PATH",
    os.path.expanduser("~/projects/chess-evolve"),
)
APP_NAME = os.environ.get("CHESS_EVOLVE_APP_NAME", "Chess Evolve")
WANDB_PROJECT = os.environ.get("CHESS_EVOLVE_WANDB_PROJECT", "chess-evolve")

_HERE = Path(__file__).parent


PYTHON_BIN = os.environ.get(
    "CHESS_EVOLVE_PYTHON",
    next(
        (p for p in [
            os.path.expanduser("~/projects/evolve/.venv/bin/python"),
            os.path.expanduser("~/evolve/.venv/bin/python"),
            sys.executable,
        ] if os.path.isfile(p)),
        sys.executable,
    ),
)


def _make_config() -> WorkerConfig:
    return WorkerConfig(
        godot_data_dir=godot_user_dir(APP_NAME),
        worker_script=_HERE / "chess_sweep_worker.py",
        worker_script_names=["chess_sweep_worker.py"],
        project_dir=Path(PROJECT_PATH),
        wandb_project=WANDB_PROJECT,
        sweep_flag="--sweep-id",
        display_name="Chess-Evolve",
        python_bin=PYTHON_BIN,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Monitor and auto-spawn chess-evolve sweep workers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_monitor_args(parser)
    parser.add_argument("--project", type=str, default=WANDB_PROJECT,
                        help=f"W&B project name (default: {WANDB_PROJECT})")
    parser.add_argument("--count", type=int, default=5,
                        help="Runs per spawned worker (default: 5)")
    args = parser.parse_args()

    cfg = _make_config()
    cfg.wandb_project = args.project

    monitor_once(
        cfg,
        auto_spawn=args.auto_spawn,
        fill=args.fill,
        max_workers=args.max_workers,
        sweep_id=args.sweep_id,
        cpu_threshold=args.cpu_threshold,
        notify=args.notify,
        notify_host=args.notify_host,
        as_json=args.as_json,
        count_per_worker=args.count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
