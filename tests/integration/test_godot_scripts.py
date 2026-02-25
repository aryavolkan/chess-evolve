"""
GDScript sanity and parse-error tests for chess-evolve.

Two layers:
 1. Static checks (fast, no Godot needed) — catch structural issues like
    wrong class names, missing methods, and type-inference pitfalls.
 2. Godot subprocess check — runs `godot --headless --import` to fully parse
    every .gd file, the same way Godot itself would on startup.

These tests exist because GDScript errors were repeatedly caught only after
launching sweep workers and waiting minutes for Godot to boot:

  - "Cannot infer the type of 'white_best_idx'" (untyped var + :=)
  - "Could not find type 'ScoreManager'" (tile-empire code leaked in)
  - "Invalid renderer/driver combination" (--rendering-driver dummy flag)

Run with:
    pytest tests/integration/test_godot_scripts.py -v
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Same probe order as chess_sweep_worker.py
GODOT_PATH: str | None = os.environ.get("GODOT_PATH") or next(
    (
        p
        for p in [
            "/usr/local/bin/godot",
            os.path.expanduser("~/.local/bin/godot"),
            "/opt/homebrew/bin/godot",
        ]
        if Path(p).is_file()
    ),
    None,
)

# GDScript dirs to skip (vendored/generated content)
_SKIP_DIRS = {"addons", "assets", ".godot"}

# Class names that must NOT appear in chess-evolve (tile-empire and other projects)
_FORBIDDEN_CLASS_NAMES = {
    "ScoreManager",
    "GameStateManager",
    "TrainingModeBase",
}


def _gd_files() -> list[Path]:
    return [
        p
        for p in PROJECT_ROOT.rglob("*.gd")
        if not (set(p.relative_to(PROJECT_ROOT).parts) & _SKIP_DIRS)
    ]


# ---------------------------------------------------------------------------
# Static checks (no Godot binary required)
# ---------------------------------------------------------------------------


def test_no_forbidden_class_names():
    """Foreign class names (e.g. from tile-empire) must not appear in the project.

    If code from another project is accidentally included, Godot will fail to
    parse scripts that reference those classes and emit obscure errors at runtime.
    """
    violations: list[str] = []
    for path in _gd_files():
        for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            m = re.match(r"^\s*class_name\s+(\w+)", line)
            if m and m.group(1) in _FORBIDDEN_CLASS_NAMES:
                violations.append(f"{path.relative_to(PROJECT_ROOT)}:{lineno}: {line.strip()}")
    assert not violations, "Forbidden class names found:\n" + "\n".join(violations)


def test_tile_empire_files_absent():
    """Specific tile-empire files that leaked in previously must not exist."""
    forbidden = [
        "score_manager.gd",
        "game_state_manager.gd",
        "modes/training_mode_base.gd",
    ]
    present = [f for f in forbidden if (PROJECT_ROOT / f).exists()]
    assert not present, "Tile-empire files found in chess-evolve:\n" + "\n".join(present)


def test_evolution_gd_exposes_required_methods():
    """training_manager.gd calls get_best_index() and get_fitness() on evolution.

    These methods must exist in evolution.gd with explicit return types so that
    callers can type their local variables correctly.
    """
    evolution_gd = PROJECT_ROOT / "ai" / "evolution.gd"
    assert evolution_gd.exists(), "ai/evolution.gd is missing"
    text = evolution_gd.read_text()
    assert re.search(r"func get_best_index\s*\(", text), "evolution.gd: missing get_best_index()"
    assert re.search(r"func get_fitness\s*\(", text), "evolution.gd: missing get_fitness()"


def test_training_manager_no_walrus_on_untyped_evolution():
    """Godot 4.5 cannot infer types when the base object is untyped.

    'var evolution' (no type annotation) means Godot can't resolve the return
    type of evolution.some_method(), so 'var x := evolution.method()' raises:
      Parse Error: Cannot infer the type of "x"

    All local variables assigned from evolution method calls must use explicit
    types, e.g.  'var x: int = evolution.get_best_index(0)'.
    """
    tm = PROJECT_ROOT / "ai" / "training_manager.gd"
    if not tm.exists():
        pytest.skip("ai/training_manager.gd not found")

    bad: list[str] = []
    for lineno, line in enumerate(tm.read_text().splitlines(), 1):
        if re.search(r"\bvar\s+\w+\s*:=\s*evolution\.", line):
            bad.append(f"  line {lineno}: {line.strip()}")
    assert not bad, (
        "training_manager.gd uses ':=' on evolution.method() — "
        "Godot 4.5 cannot infer types from calls on untyped vars.\n"
        "Replace ':=' with explicit types, e.g.  var x: int = evolution.get_best_index(0)\n"
        + "\n".join(bad)
    )


def test_godot_wandb_no_rendering_driver_dummy():
    """launch_godot() must not pass '--rendering-driver dummy'.

    That flag is incompatible with the Forward Plus renderer used by chess-evolve
    and causes: 'Invalid renderer/driver combination forward_plus and dummy'.
    """
    shared_roots = [
        os.path.expanduser("~/projects/shared-evolve-utils"),
        os.path.expanduser("~/Projects/shared-evolve-utils"),
        os.path.expanduser("~/shared-evolve-utils"),
    ]
    gw = next((Path(d) / "godot_wandb.py" for d in shared_roots if (Path(d) / "godot_wandb.py").exists()), None)
    if gw is None:
        pytest.skip("shared-evolve-utils/godot_wandb.py not found")

    for lineno, line in enumerate(gw.read_text().splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if "--rendering-driver" in line and "dummy" in line:
            pytest.fail(
                f"godot_wandb.py:{lineno}: Found '--rendering-driver dummy' — "
                "remove this flag; it breaks Forward Plus renderer.\n"
                f"  {stripped}"
            )


# ---------------------------------------------------------------------------
# Godot subprocess tests (require Godot binary on PATH or GODOT_PATH env)
# ---------------------------------------------------------------------------


def _run_godot(*extra_args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    cmd = [GODOT_PATH, "--headless", "--path", str(PROJECT_ROOT), *extra_args]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


@pytest.mark.skipif(not GODOT_PATH, reason="Godot binary not found (set GODOT_PATH env)")
def test_godot_import_no_script_parse_errors():
    """Run 'godot --headless --import' and assert no GDScript parse errors.

    This is the definitive test: it runs the actual Godot parser on every
    .gd file in the project.  Any SCRIPT ERROR: Parse Error line in the
    output means Godot would refuse to load the script at runtime.

    Catches:
    - Type inference failures (Cannot infer the type of "x" variable)
    - Missing class_name references (Could not find type "Foo")
    - Syntax errors
    - Any other parse-time issues
    """
    result = _run_godot("--import", timeout=120)
    output = result.stdout + result.stderr

    parse_errors = [line for line in output.splitlines() if "SCRIPT ERROR" in line and "Parse Error" in line]

    assert not parse_errors, (
        "Godot reported GDScript parse errors:\n" + "\n".join(parse_errors) + "\n\nFull output:\n" + output[-3000:]
    )
