#!/usr/bin/env python3
"""
Check chess-evolve sweep worker health.

Detects stuck workers by tracking metrics staleness and CPU activity:
- metrics stale >N min AND low CPU  → stuck (likely hung)
- metrics stale >N min AND high CPU → slow (large population, not a problem)
- no metrics file after startup     → init hang

Usage:
    python3 check_workers.py
    python3 check_workers.py --stale-minutes 20 --json
    python3 check_workers.py --metrics-dir /custom/path

Exits 0 = all healthy/slow, 1 = stuck workers found.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_METRICS_DIR = Path.home() / ".local/share/godot/app_userdata/Chess Evolve"
STALE_MINUTES = 15


def _ps_lines() -> list[str]:
    try:
        return subprocess.check_output(["ps", "aux"], text=True, stderr=subprocess.DEVNULL).splitlines()
    except Exception:
        return []


def get_sweep_workers() -> list[tuple[int, str | None]]:
    """Return (pid, sweep_id) for running chess_sweep_worker.py processes."""
    result = []
    for line in _ps_lines():
        if "chess_sweep_worker.py" in line and "grep" not in line:
            parts = line.split()
            pid = int(parts[1])
            sweep_id = None
            for i, t in enumerate(parts):
                if t == "--sweep-id" and i + 1 < len(parts):
                    sweep_id = parts[i + 1]
            result.append((pid, sweep_id))
    return result


def get_godot_workers() -> list[tuple[int, str | None]]:
    """Return (pid, worker_id) for running Godot auto-train processes."""
    result = []
    for line in _ps_lines():
        if "godot" in line and "auto-train" in line and "grep" not in line:
            parts = line.split()
            pid = int(parts[1])
            worker_id = None
            for t in parts:
                if t.startswith("--worker-id="):
                    worker_id = t.split("=", 1)[1]
            result.append((pid, worker_id))
    return result


def _cpu_percent(pid: int) -> float | None:
    """Sample CPU% for a PID over 0.5s using /proc/stat."""
    try:
        def _ticks(p):
            with open(f"/proc/{p}/stat") as f:
                s = f.read().split()
            return int(s[13]) + int(s[14]), time.time()
        t1, ts1 = _ticks(pid)
        time.sleep(0.5)
        t2, ts2 = _ticks(pid)
        clk = os.sysconf("SC_CLK_TCK")
        return round((t2 - t1) / clk / (ts2 - ts1) * 100, 1)
    except Exception:
        return None


def _metrics_age_minutes(metrics_dir: Path, worker_id: str) -> float:
    p = metrics_dir / f"metrics_{worker_id}.json"
    if not p.exists():
        return float("inf")
    return (time.time() - p.stat().st_mtime) / 60


def _read_metrics(metrics_dir: Path, worker_id: str) -> dict | None:
    p = metrics_dir / f"metrics_{worker_id}.json"
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Check chess-evolve sweep worker health")
    parser.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR)
    parser.add_argument("--stale-minutes", type=float, default=STALE_MINUTES)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    sweep_workers = get_sweep_workers()
    godot_workers = get_godot_workers()

    healthy, slow, stuck = [], [], []

    for g_pid, worker_id in godot_workers:
        if not worker_id:
            continue

        stale = _metrics_age_minutes(args.metrics_dir, worker_id)
        has_metrics = stale < float("inf")
        metrics = _read_metrics(args.metrics_dir, worker_id)
        last_gen = metrics.get("generation") if metrics else None
        best_fitness = metrics.get("best_fitness") if metrics else None

        cpu = _cpu_percent(g_pid)

        entry = {
            "godot_pid": g_pid,
            "worker_id": worker_id,
            "stale_minutes": round(stale, 1) if has_metrics else None,
            "last_gen": last_gen,
            "best_fitness": best_fitness,
            "has_metrics": has_metrics,
            "cpu_pct": cpu,
            "status": "ok",
            "problems": [],
        }

        if not has_metrics and (cpu is None or cpu < 10):
            entry["problems"].append(f"no metrics file, cpu={cpu}% — likely init hang")
        elif has_metrics and stale > args.stale_minutes:
            if cpu is None or cpu < 10:
                entry["problems"].append(
                    f"metrics stale {stale:.1f}min, cpu={cpu}% (gen={last_gen}) — likely hung"
                )
            else:
                entry["status"] = "slow"  # high CPU + stale = slow but alive

        if entry["problems"]:
            entry["status"] = "stuck"
            stuck.append(entry)
        elif entry["status"] == "slow":
            slow.append(entry)
        else:
            healthy.append(entry)

    if args.json:
        print(json.dumps({
            "sweep_workers": len(sweep_workers),
            "godot_workers": len(godot_workers),
            "healthy": healthy,
            "slow": slow,
            "stuck": stuck,
        }, indent=2))
    else:
        print(f"Sweep workers: {len(sweep_workers)} | Godot workers: {len(godot_workers)}")
        if healthy:
            print(f"\n✅ Healthy ({len(healthy)}):")
            for e in healthy:
                gen_str = f"gen={e['last_gen']}" if e["last_gen"] is not None else "starting"
                best_str = f" best={e['best_fitness']:.1f}" if e["best_fitness"] else ""
                stale_str = f" stale={e['stale_minutes']}m" if e["stale_minutes"] and e["stale_minutes"] > 1 else ""
                print(f"  [{e['godot_pid']}] {e['worker_id'][:8]} {gen_str}{best_str}{stale_str}")
        if slow:
            print(f"\n🟡 Slow ({len(slow)}) — high CPU, large population or complex generation:")
            for e in slow:
                gen_str = f"gen={e['last_gen']}" if e["last_gen"] is not None else "starting"
                best_str = f" best={e['best_fitness']:.1f}" if e["best_fitness"] else ""
                print(f"  [{e['godot_pid']}] {e['worker_id'][:8]} {gen_str}{best_str} cpu={e['cpu_pct']}% stale={e['stale_minutes']}m")
        if stuck:
            print(f"\n🔴 Stuck ({len(stuck)}):")
            for e in stuck:
                print(f"  [{e['godot_pid']}] {e['worker_id'][:8]} cpu={e['cpu_pct']}%")
                for p in e["problems"]:
                    print(f"    → {p}")
            print("\nTo restart stuck workers:")
            for e in stuck:
                print(f"  kill -9 {e['godot_pid']}  # {e['worker_id'][:8]}")

    return 1 if stuck else 0


if __name__ == "__main__":
    sys.exit(main())
