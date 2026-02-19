#!/usr/bin/env python3
"""Wrapper to run the project-level train_wandb.py from scripts/."""
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "train_wandb.py"), run_name="__main__")
