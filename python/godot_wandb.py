#!/usr/bin/env python3
"""
Shared Godot + W&B training utilities.

Common helpers for launching Godot training, polling metrics, and logging to
Weights & Biases. Used by both Evolve and Chess-Evolve projects.
"""

import json
import logging
import math
import os
import platform
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

# Force unbuffered output for nohup/sweep compatibility
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Default Godot binary
DEFAULT_GODOT_PATH = os.environ.get("GODOT_PATH", "/opt/homebrew/bin/godot")


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def godot_user_dir(app_name: str) -> Path:
    """Return the Godot user data directory for a project on the current platform.

    Args:
        app_name: The Godot project name (e.g. "evolve", "Chess Evolve").
    """
    override = os.environ.get("GODOT_USER_DIR")
    if override:
        return Path(override)

    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library/Application Support/Godot/app_userdata" / app_name
    elif system == "Windows":
        appdata = os.environ.get("APPDATA", "")
        return Path(appdata) / "Godot/app_userdata" / app_name
    else:  # Linux
        return Path.home() / ".local/share/godot/app_userdata" / app_name


# ---------------------------------------------------------------------------
# Config / metrics I/O
# ---------------------------------------------------------------------------


def write_config(config: dict, config_path: Path) -> None:
    """Write a JSON config file for Godot to read at startup."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"✓ Config written to {config_path}")


def read_metrics(metrics_path: Path) -> dict | None:
    """Read current metrics from Godot's JSON file. Returns None if unavailable."""
    try:
        if not metrics_path.exists():
            return None
        with open(metrics_path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Per-worker state (sweep workers need isolated paths)
# ---------------------------------------------------------------------------


class SweepWorker:
    """Manages per-worker state for parallel sweep runs.

    Each worker gets a unique ID so multiple workers can share the same Godot
    user-data directory without clobbering each other's config/metrics files.

    Usage::

        worker = SweepWorker(godot_user_dir("evolve"))
        worker.write_config(config)
        proc = launch_godot(..., metrics_path=worker.metrics_path)
        # ... training ...
        worker.cleanup()
    """

    def __init__(self, user_dir: Path, worker_id: str = None):
        self.worker_id = worker_id or uuid.uuid4().hex[:8]
        self.user_dir = user_dir
        self.config_path = user_dir / f"sweep_config_{self.worker_id}.json"
        self.metrics_path = user_dir / f"metrics_{self.worker_id}.json"

    def write_config(self, config: dict) -> None:
        """Write per-worker config JSON for Godot."""
        self.user_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"  Config written: {self.config_path}")

    def clear_metrics(self) -> None:
        """Delete stale metrics files before a new run."""
        if self.metrics_path.exists():
            self.metrics_path.unlink()
        jsonl_path = Path(str(self.metrics_path) + "l")
        if jsonl_path.exists():
            jsonl_path.unlink()

    def cleanup(self) -> None:
        """Remove per-worker config and metrics files."""
        for path in [self.config_path, self.metrics_path]:
            try:
                if path.exists():
                    path.unlink()
            except OSError as e:
                logger.warning("Failed to remove worker file %s: %s", path, e)


# ---------------------------------------------------------------------------
# Godot process management
# ---------------------------------------------------------------------------


def launch_godot(
    project_path: str,
    godot_path: str = DEFAULT_GODOT_PATH,
    visible: bool = False,
    extra_args: list[str] = None,
    metrics_path: Path | None = None,
    worker_id: str | None = None,
) -> subprocess.Popen:
    """Launch a Godot training process.

    Args:
        project_path: Path to the Godot project directory.
        godot_path: Path to the Godot binary.
        visible: If False, run headless.
        extra_args: Extra Godot user args (after --).
        metrics_path: If provided, clear old metrics before launch.
        worker_id: If provided, pass ``--worker-id=<id>`` to Godot.

    Returns:
        The subprocess.Popen instance.
    """
    if metrics_path and metrics_path.exists():
        metrics_path.unlink()
        print("✓ Cleared old metrics")

    cmd = [godot_path, "--path", project_path]
    if not visible:
        cmd.extend(["--headless", "--rendering-driver", "dummy"])

    user_args = extra_args or ["--auto-train"]
    if worker_id:
        user_args = list(user_args) + [f"--worker-id={worker_id}"]
    cmd.extend(["--"] + user_args)

    print(f"🚀 Launching: {' '.join(cmd)}")
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def wait_for_metrics(
    metrics_path: Path,
    timeout: float = 30.0,
) -> bool:
    """Wait for metrics.json to appear (training started).

    Returns True if training started within timeout.
    """
    print(f"⏳ Waiting for training to start (timeout: {timeout}s)...")
    start = time.time()
    while time.time() - start < timeout:
        if metrics_path.exists():
            print("✓ Training started!")
            return True
        time.sleep(1)
    print("❌ Timeout waiting for training start")
    return False


# ---------------------------------------------------------------------------
# Metrics polling + W&B logging
# ---------------------------------------------------------------------------


def define_step_metric(step: str = "generation") -> None:
    """Configure W&B so all metrics use *step* as their x-axis.

    Call once after ``wandb.init()``.
    """
    import wandb

    wandb.define_metric(step)
    wandb.define_metric("*", step_metric=step)


def compute_derived_metrics(
    best_history: list[float],
    avg_history: list[float],
) -> dict:
    """Compute aggregate metrics across generations.

    Returns a dict with:
    - ``mean_best_fitness``
    - ``mean_avg_fitness``
    - ``max_best_fitness``
    - ``improvement_rate`` (points per generation)
    - ``fitness_std_dev``
    """
    if not best_history:
        return {}
    n = len(best_history)
    mean_best = sum(best_history) / n
    mean_avg = sum(avg_history) / n if avg_history else 0.0
    max_best = max(best_history)
    improvement_rate = (best_history[-1] - best_history[0]) / max(n, 1) if n > 1 else 0.0
    fitness_std = (sum((x - mean_avg) ** 2 for x in avg_history) / n) ** 0.5 if avg_history else 0.0
    return {
        "mean_best_fitness": mean_best,
        "mean_avg_fitness": mean_avg,
        "max_best_fitness": max_best,
        "improvement_rate": improvement_rate,
        "fitness_std_dev": fitness_std,
    }


def calc_training_timeout(
    population_size: int,
    evals_per_individual: int,
    parallel_count: int,
    max_generations: int,
    min_per_eval: float = 5.0,
) -> int:
    """Estimate training timeout in minutes.

    Formula: ceil(max_gens * (pop * evals / parallel) * min_per_eval / 60)

    Args:
        min_per_eval: Minutes per individual evaluation (default 5s → 5/60).
    """
    evals_per_gen = population_size * evals_per_individual / max(parallel_count, 1)
    return int(math.ceil(max_generations * evals_per_gen * min_per_eval / 60))


def poll_metrics(
    wandb_run,
    metrics_path: Path,
    max_generations: int = 100,
    poll_interval: float = 1.0,
    max_stale: int = 60,
    log_keys: list[str] | None = None,
) -> dict | None:
    """Tail metrics.jsonl and log every generation to W&B incrementally.

    Reads new lines appended to metrics_path + 'l' (e.g. metrics.jsonl) so
    that fast training runs don't get skipped by a slow poll interval.
    Falls back to polling metrics.json if the JSONL file doesn't exist.

    Args:
        wandb_run: An active wandb.Run instance.
        metrics_path: Path to the metrics.json file (JSONL at same path + 'l').
        max_generations: Stop after this many generations.
        poll_interval: Seconds between checks when no new data.
        max_stale: Max consecutive stale polls before aborting.
        log_keys: If provided, only log these keys. Otherwise log all numeric keys.

    Returns:
        The final metrics dict, or None.
    """
    jsonl_path = Path(str(metrics_path) + "l")  # metrics.json → metrics.jsonl
    file_pos = 0
    last_gen = -1
    stale_count = 0
    final_metrics = None
    print(f"📊 Tailing {jsonl_path.name} for incremental gen logging...")

    while True:
        # --- Try JSONL incremental read first ---
        new_records = []
        if jsonl_path.exists():
            try:
                with open(jsonl_path) as f:
                    f.seek(file_pos)
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                new_records.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
                    file_pos = f.tell()
            except OSError as e:
                logger.warning("Failed to read JSONL metrics from %s: %s", jsonl_path, e)

        # --- Fall back to snapshot if no JSONL yet ---
        if not new_records and not jsonl_path.exists():
            metrics = read_metrics(metrics_path)
            if metrics and "generation" in metrics and metrics["generation"] > last_gen:
                new_records = [metrics]

        # --- Log each new record ---
        for metrics in new_records:
            gen = metrics.get("generation", -1)
            if gen <= last_gen:
                continue
            last_gen = gen
            stale_count = 0
            final_metrics = metrics

            if log_keys:
                log_data = {k: metrics.get(k, 0) for k in log_keys}
            else:
                log_data = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}

            wandb_run.log(log_data)

            parts = [f"Gen {gen}"]
            for k in sorted(log_data):
                if k != "generation":
                    v = log_data[k]
                    parts.append(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}")
            print(", ".join(parts[:6]))

        if last_gen >= max_generations - 1:
            print(f"\n✓ Training complete! Reached generation {last_gen}")
            break

        if not new_records:
            stale_count += 1
            if stale_count >= max_stale:
                print("❌ Training appears stuck")
                return final_metrics

        time.sleep(poll_interval)

    return final_metrics


def log_final_summary(wandb_run, metrics: dict, key_map: dict[str, str] | None = None) -> None:
    """Write final metric values to wandb.summary with a ``final_`` prefix.

    Args:
        wandb_run: Active wandb run (or ``wandb`` module for the active run).
        metrics: Raw metrics dict from ``read_metrics()``.
        key_map: Optional ``{metric_key: summary_key}`` mapping. If omitted,
            all numeric keys are written as ``final_<key>``.
    """
    if not metrics:
        return
    if key_map:
        for src, dst in key_map.items():
            v = metrics.get(src)
            if v is not None:
                wandb_run.summary[dst] = v
    else:
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                wandb_run.summary[f"final_{k}"] = v


# ---------------------------------------------------------------------------
# Sweep management
# ---------------------------------------------------------------------------


def create_or_join_sweep(
    sweep_config: dict,
    project: str,
    sweep_id: str = None,
) -> str:
    """Create a new W&B sweep or join an existing one.

    Args:
        sweep_config: Sweep config dict (method, metric, parameters).
        project: W&B project name (``"entity/project"`` or just ``"project"``).
        sweep_id: If provided, join this sweep instead of creating a new one.

    Returns:
        The sweep ID string.
    """
    import wandb

    if sweep_id:
        print(f"\n🔄 Joining existing sweep: {sweep_id}")
    else:
        sweep_id = wandb.sweep(sweep_config, project=project)
        print(f"\n✨ Created new sweep: {sweep_id}")

    # Extract just the project slug for the URL
    project_slug = project.split("/")[-1]
    print(f"   Sweep URL: https://wandb.ai/{project_slug}/sweeps/{sweep_id}")
    return sweep_id


def run_sweep_agent(
    sweep_id: str,
    project: str,
    train_fn: Callable[[], None],
    count: int | None = None,
    cleanup_fn: Callable[[], None] | None = None,
) -> None:
    """Run a W&B sweep agent with graceful SIGINT/SIGTERM shutdown.

    Args:
        sweep_id: The W&B sweep ID to join.
        project: W&B project name (``"entity/project"`` or just ``"project"``).
        train_fn: Callable that runs one sweep trial (called by wandb.agent).
        count: Max number of runs (None = unlimited).
        cleanup_fn: Optional callable invoked on shutdown before exit.
    """
    import wandb

    def _handle_signal(sig, frame):
        print("\n\n🛑 Sweep interrupted. Cleaning up...")
        if cleanup_fn:
            cleanup_fn()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # Parse entity/project
    if "/" in project:
        entity, proj = project.split("/", 1)
    else:
        entity, proj = None, project

    wandb.agent(sweep_id, function=train_fn, count=count, entity=entity, project=proj)


# ---------------------------------------------------------------------------
# High-level: full training session
# ---------------------------------------------------------------------------


def run_training(
    config: dict,
    project_path: str,
    app_name: str,
    wandb_project: str,
    wandb_tags: list[str] | None = None,
    visible: bool = False,
    godot_path: str = DEFAULT_GODOT_PATH,
    log_keys: list[str] | None = None,
) -> None:
    """Run a complete training session: config → launch → poll → log → cleanup.

    Args:
        config: Training hyperparameter config dict.
        project_path: Path to the Godot project.
        app_name: Godot app name for user data dir.
        wandb_project: W&B project name.
        wandb_tags: Tags for the W&B run.
        visible: Show Godot window.
        godot_path: Path to Godot binary.
        log_keys: Specific metric keys to log (or None for all).
    """
    import wandb

    user_dir = godot_user_dir(app_name)
    metrics_path = user_dir / "metrics.json"
    config_path = user_dir / "sweep_config.json"

    run = wandb.init(
        project=wandb_project,
        config=config,
        tags=wandb_tags or [],
    )
    define_step_metric()

    max_gens = config.get("max_generations", 100)
    print(f"\n🎮 Starting training (pop={config.get('population_size', '?')}, gens={max_gens})\n")

    write_config(config, config_path)
    process = launch_godot(
        project_path,
        godot_path,
        visible,
        metrics_path=metrics_path,
    )

    try:
        if not wait_for_metrics(metrics_path, timeout=120.0):
            process.kill()
            run.finish(exit_code=1)
            return

        final = poll_metrics(run, metrics_path, max_gens, log_keys=log_keys)

        time.sleep(5)
        process.terminate()
        process.wait(timeout=10)

        log_final_summary(run, final)
        print("\n✅ Training complete!")
        run.finish(exit_code=0)

    except KeyboardInterrupt:
        print("\n🛑 Interrupted")
        process.kill()
        run.finish(exit_code=130)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        process.kill()
        run.finish(exit_code=1)
