"""FastAPI backend for chess-evolve worker monitoring dashboard."""
from __future__ import annotations

import os
import platform
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import psutil
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
TRAIN_SCRIPT = PROJECT_ROOT / "train_wandb.py"

app = FastAPI(title="Chess-Evolve Worker Monitor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------

@dataclass
class WorkerInfo:
    pid: int
    sweep_id: str | None
    log_file: str | None
    spawn_time: float = field(default_factory=time.time)


_tracked: dict[int, WorkerInfo] = {}
_sweep_cache: dict = {"data": [], "ts": 0.0}
SWEEP_CACHE_TTL = 30.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_workers() -> list[dict]:
    """Discover running workers via psutil, reconcile with tracked state."""
    workers = []
    alive_pids: set[int] = set()

    for proc in psutil.process_iter(["pid", "cmdline", "cpu_percent", "memory_percent", "create_time"]):
        try:
            cmdline = proc.info["cmdline"] or []
            cmd_str = " ".join(cmdline)
            if "train_wandb.py" not in cmd_str:
                continue
            if "--sweep" not in cmd_str:
                continue
            # Only count actual Python workers — skip bash/sh wrappers
            # and ssh -f processes whose cmdline contains the train script.
            if cmdline and not any("python" in c.lower() for c in cmdline[:2]):
                continue

            pid = proc.info["pid"]
            alive_pids.add(pid)

            # Extract sweep ID from cmdline
            sweep_id = None
            for i, arg in enumerate(cmdline):
                if arg == "--sweep" and i + 1 < len(cmdline):
                    sweep_id = cmdline[i + 1]
                    break

            # Reconcile with tracked state
            if pid not in _tracked:
                log_file = _guess_log_file(pid)
                _tracked[pid] = WorkerInfo(pid=pid, sweep_id=sweep_id, log_file=log_file)
            elif sweep_id:
                _tracked[pid].sweep_id = sweep_id

            info = _tracked[pid]
            cpu = proc.info["cpu_percent"] or 0.0
            mem = proc.info["memory_percent"] or 0.0
            uptime = time.time() - (proc.info["create_time"] or time.time())
            last_line = _read_last_line(info.log_file) if info.log_file else ""
            gen, run_count = _extract_gen_and_run(info.log_file) if info.log_file else (None, 0)

            status = "running" if cpu > 5.0 else "idle"

            workers.append({
                "pid": pid,
                "sweep_id": info.sweep_id,
                "cpu_pct": round(cpu, 1),
                "mem_pct": round(mem, 1),
                "status": status,
                "uptime_sec": int(uptime),
                "last_log_line": last_line,
                "log_file": info.log_file,
                "generation": gen,
                "run_count": run_count,
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # Prune dead tracked workers
    dead = [pid for pid in _tracked if pid not in alive_pids]
    for pid in dead:
        del _tracked[pid]

    return workers


def _guess_log_file(pid: int) -> str | None:
    """Try to find the log file for a worker by checking open files or scanning worker*.log."""
    try:
        proc = psutil.Process(pid)
        for f in proc.open_files():
            if "worker" in f.path and f.path.endswith(".log"):
                return f.path
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

    # Fallback: scan worker*.log in project root for most recently modified
    for log in sorted(PROJECT_ROOT.glob("worker*.log"), key=lambda p: p.stat().st_mtime, reverse=True):
        return str(log)
    return None


def _read_last_lines(path: str | None, n: int = 5) -> list[str]:
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            if size == 0:
                return []
            read_size = min(size, 8192)
            f.seek(-read_size, 2)
            lines = f.read().decode("utf-8", errors="replace").splitlines()
            return lines[-n:]
    except OSError:
        return []


def _read_last_line(path: str | None) -> str:
    lines = _read_last_lines(path, 1)
    return lines[0] if lines else ""


def _extract_gen_and_run(path: str | None) -> tuple[int | None, int]:
    """Extract current generation and run count from worker log."""
    if not path or not os.path.exists(path):
        return None, 0

    # Get generation from tail of file
    lines = _read_last_lines(path, 50)
    gen = None
    for line in reversed(lines):
        if gen is None and "gen " in line and ":" in line:
            try:
                part = line.strip().split("gen ")[1].split(":")[0].strip()
                gen = int(part)
                break
            except (IndexError, ValueError):
                pass

    # Count total runs by scanning full file for "Starting Run" markers.
    # Uses binary read + count for efficiency on large logs.
    run_count = 0
    try:
        with open(path, "rb") as f:
            content = f.read()
            run_count = content.count(b"Starting Run")
    except OSError:
        pass

    return gen, run_count


def _tail_file(path: str, lines: int = 100) -> str:
    if not os.path.exists(path):
        return ""
    try:
        with open(path) as f:
            return "\n".join(deque(f, maxlen=lines))
    except OSError:
        return ""


def _next_log_number() -> int:
    existing = list(PROJECT_ROOT.glob("worker*.log"))
    nums = []
    for p in existing:
        name = p.stem.replace("worker", "")
        if name.isdigit():
            nums.append(int(name))
    return max(nums, default=0) + 1


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/system")
def get_system():
    mem = psutil.virtual_memory()
    return {
        "hostname": platform.node(),
        "cpu_cores": psutil.cpu_count(),
        "cpu_percent": psutil.cpu_percent(interval=0.5),
        "mem_total_gb": round(mem.total / (1024**3), 1),
        "mem_used_gb": round(mem.used / (1024**3), 1),
        "mem_percent": mem.percent,
    }


@app.get("/api/workers")
def list_workers():
    return _find_workers()


@app.post("/api/workers/kill/{pid}")
def kill_worker(pid: int):
    if pid not in _tracked and not any(
        p.pid == pid for p in psutil.process_iter(["pid", "cmdline"])
        if "train_wandb.py" in " ".join(p.info.get("cmdline") or [])
    ):
        raise HTTPException(404, f"PID {pid} is not a known worker")
    try:
        proc = psutil.Process(pid)
        proc.terminate()
        proc.wait(timeout=5)
    except psutil.TimeoutExpired:
        psutil.Process(pid).kill()
    except psutil.NoSuchProcess:
        pass
    _tracked.pop(pid, None)
    return {"killed": pid}


class SpawnRequest(BaseModel):
    sweep_id: str
    count: int = 1
    rayon_threads: int = 4


@app.post("/api/workers/spawn")
def spawn_workers(req: SpawnRequest):
    spawned = []
    for _ in range(req.count):
        log_num = _next_log_number()
        log_file = PROJECT_ROOT / f"worker{log_num}.log"
        env = {**os.environ, "RAYON_NUM_THREADS": str(req.rayon_threads)}
        proc = subprocess.Popen(
            [str(VENV_PYTHON), str(TRAIN_SCRIPT), "--sweep", req.sweep_id],
            stdout=open(log_file, "w"),
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env=env,
            start_new_session=True,
        )
        _tracked[proc.pid] = WorkerInfo(
            pid=proc.pid, sweep_id=req.sweep_id, log_file=str(log_file),
        )
        spawned.append({"pid": proc.pid, "log_file": str(log_file)})
        time.sleep(1)  # stagger startup
    return {"spawned": spawned}


@app.get("/api/workers/{pid}/logs")
def get_worker_logs(pid: int, lines: int = Query(default=100, le=2000)):
    info = _tracked.get(pid)
    if not info or not info.log_file:
        raise HTTPException(404, f"No log file for PID {pid}")
    return {"pid": pid, "log": _tail_file(info.log_file, lines)}


_SWEEP_FETCH_SCRIPT = '''
import json, os, sys
os.environ["WANDB_HTTP_TIMEOUT"] = "10"
os.environ["WANDB_SILENT"] = "true"
import wandb
api = wandb.Api(timeout=10)
runs = list(api.runs("chess-evolve", per_page=50, order="-created_at"))
sweeps = {}
for r in runs:
    sid = r.sweep.id if r.sweep else None
    if not sid:
        continue
    if sid not in sweeps:
        sweeps[sid] = {"id": sid, "state": "unknown", "run_count": 0, "running": 0, "best_metric": None,
                       "url": f"https://wandb.ai/aryavolkan-personal/chess-evolve/sweeps/{sid}"}
    sweeps[sid]["run_count"] += 1
    if r.state == "running":
        sweeps[sid]["running"] += 1
    bwr = r.summary.get("bench_avg_win_rate")
    if bwr is not None:
        cur = sweeps[sid]["best_metric"]
        if cur is None or bwr > cur:
            sweeps[sid]["best_metric"] = round(bwr, 3)
json.dump(list(sweeps.values())[:10], sys.stdout)
'''


@app.get("/api/sweeps")
def list_sweeps():
    now = time.time()
    if now - _sweep_cache["ts"] < SWEEP_CACHE_TTL and _sweep_cache["data"]:
        return _sweep_cache["data"]

    # Run W&B query in a subprocess with a hard kill timeout.
    # The W&B SDK has an infinite retry loop on network errors that cannot
    # be interrupted from a thread — only a process kill stops it.
    import json as _json
    try:
        result = subprocess.run(
            [str(VENV_PYTHON), "-c", _SWEEP_FETCH_SCRIPT],
            capture_output=True, text=True, timeout=20,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode == 0 and result.stdout.strip():
            data = _json.loads(result.stdout)
            _sweep_cache["data"] = data
            _sweep_cache["ts"] = now
            return data
    except (subprocess.TimeoutExpired, Exception):
        pass

    return _sweep_cache.get("data", [])
