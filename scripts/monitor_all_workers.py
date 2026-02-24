#!/usr/bin/env python3
"""
Monitor chess-evolve and evolve workers on local + mint-alien.
Sends a WhatsApp alert via NanoClaw IPC when any worker type is missing.

Usage:
    python monitor_all_workers.py              # Check and notify if broken
    python monitor_all_workers.py --dry-run    # Check only, no notification
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

NANOCLAW_IPC = Path.home() / "projects/nanoclaw/data/ipc/main/messages"
WHATSAPP_JID = "12066088083@s.whatsapp.net"

REMOTE_HOST = "mint-alien"
REMOTE_PASS = "4242"


@dataclass
class WorkerStatus:
    host: str
    project: str
    python_count: int
    godot_count: int

    @property
    def alive(self) -> bool:
        return self.godot_count > 0

    def summary(self) -> str:
        if self.alive:
            return f"  {self.host}/{self.project}: {self.godot_count} godot, {self.python_count} python"
        return f"  {self.host}/{self.project}: DOWN (0 workers)"


def _parse_ps(ps_output: str, project_path_fragment: str) -> tuple[int, int]:
    """Count (python_workers, godot_instances) for a project from ps output.

    Godot processes are reliably identified by ``--path .../project``.
    Python workers can't be distinguished by command line alone (they all
    run ``train_wandb.py --sweep``), so we only count them when the
    project fragment appears in the full command/path.
    """
    py_count = 0
    godot_count = 0
    for line in ps_output.splitlines():
        if "grep" in line:
            continue
        if "godot" in line.lower() and "--auto-train" in line:
            if project_path_fragment in line:
                godot_count += 1
        elif "train_wandb.py" in line and "--sweep" in line:
            if project_path_fragment in line:
                py_count += 1
    return py_count, godot_count


def _local_ps() -> str:
    try:
        return subprocess.check_output(["ps", "aux"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return ""


def _remote_ps() -> str | None:
    """Get ps aux from mint-alien. Returns None on SSH failure."""
    try:
        result = subprocess.run(
            ["sshpass", "-p", REMOTE_PASS, "ssh",
             "-o", "ConnectTimeout=10",
             "-o", "StrictHostKeyChecking=no",
             f"aryasen@{REMOTE_HOST}",
             "ps aux"],
            capture_output=True, text=True, timeout=20,
        )
        return result.stdout if result.returncode == 0 else None
    except Exception:
        return None


def check_all() -> list[WorkerStatus]:
    """Check all worker types on all hosts."""
    results = []

    local_ps = _local_ps()
    for project in ("chess-evolve", "evolve"):
        py, godot = _parse_ps(local_ps, project)
        results.append(WorkerStatus("local", project, py, godot))

    remote_ps = _remote_ps()
    for project in ("chess-evolve", "evolve"):
        if remote_ps is None:
            results.append(WorkerStatus(REMOTE_HOST, project, -1, -1))
        else:
            py, godot = _parse_ps(remote_ps, project)
            results.append(WorkerStatus(REMOTE_HOST, project, py, godot))

    return results


def build_report(statuses: list[WorkerStatus]) -> str:
    """Build a WhatsApp status report. Always sent."""
    down = [s for s in statuses if not s.alive and s.python_count != -1]
    ssh_failed = [s for s in statuses if s.python_count == -1]
    hostname = os.uname().nodename
    now = datetime.now().strftime("%H:%M")

    if down or ssh_failed:
        lines = [f"*Worker Alert* [{hostname}] {now}"]
    else:
        lines = [f"*Worker Status* [{hostname}] {now}"]

    if ssh_failed:
        lines.append(f"SSH to {REMOTE_HOST} unreachable")

    if down:
        for s in down:
            lines.append(f"DOWN: {s.host}/{s.project}")

    lines.append("")
    for s in statuses:
        if s.python_count == -1:
            lines.append(f"  {s.host}/{s.project}: unreachable")
        elif s.alive:
            lines.append(f"  {s.host}/{s.project}: {s.godot_count} godot")
        else:
            lines.append(f"  {s.host}/{s.project}: DOWN")

    total = sum(s.godot_count for s in statuses if s.godot_count > 0)
    lines.append(f"\nTotal: {total} godot workers")

    return "\n".join(lines)


def send_whatsapp(message: str) -> bool:
    """Write IPC message for NanoClaw."""
    if not NANOCLAW_IPC.exists():
        print(f"IPC dir not found: {NANOCLAW_IPC}", file=sys.stderr)
        return False
    ts = int(time.time() * 1000)
    msg_file = NANOCLAW_IPC / f"worker-alert-{ts}.json"
    msg_file.write_text(json.dumps({
        "type": "message",
        "chatJid": WHATSAPP_JID,
        "text": message,
    }))
    print(f"WhatsApp alert queued: {msg_file.name}")
    return True


def main() -> int:
    dry_run = "--dry-run" in sys.argv

    statuses = check_all()

    # Print status
    print(f"Worker check — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    for s in statuses:
        print(s.summary())

    report = build_report(statuses)
    print(f"\n{report}")
    if not dry_run:
        send_whatsapp(report)

    has_down = any(not s.alive for s in statuses)
    return 1 if has_down else 0


if __name__ == "__main__":
    raise SystemExit(main())
