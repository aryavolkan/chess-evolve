"""Global elite genome pool shared across parallel chess-evolve training runs.

Architecture
------------
Each worker owns exactly one contribution file:
    elite_contrib_{worker_id}.json

Before each run a worker:
  1. Reads ALL existing elite_contrib_*.json files and merges them.
  2. Writes a per-run seed file global_elite_{worker_id}.json for Godot to read.
  3. Launches Godot, which seeds its hall-of-fame from that file.

After each run a worker:
  1. Reads elite_contrib_{worker_id}.json that Godot wrote.
  2. Merges new genomes into the worker's own contribution file.
  3. Removes the seed file so stale genomes aren't re-loaded next time.

This avoids file-locking: each worker owns exactly one file it writes to,
and reads (never writes) the files of other workers.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

# Maximum total pool size across all merged contributions.
DEFAULT_POOL_SIZE = 20

# Number of elites injected into each new training run.
DEFAULT_SEED_COUNT = 5


class GlobalElitePool:
    """Manages the shared genome pool for parallel training workers."""

    def __init__(self, user_dir: str | Path, pool_size: int = DEFAULT_POOL_SIZE):
        self.user_dir = Path(user_dir)
        self.pool_size = pool_size

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    def contrib_path(self, worker_id: str) -> Path:
        """Contribution file owned by this worker (persists between runs)."""
        return self.user_dir / f"elite_contrib_{worker_id}.json"

    def seed_path(self, worker_id: str) -> Path:
        """Seed file written before a run, consumed by Godot at startup."""
        return self.user_dir / f"global_elite_{worker_id}.json"

    # ------------------------------------------------------------------
    # Pool management
    # ------------------------------------------------------------------

    def merge_all(self) -> list[dict]:
        """Read every contribution file and return the merged top-N elites."""
        all_elites: list[dict] = []
        for path in self.user_dir.glob("elite_contrib_*.json"):
            try:
                data = json.loads(path.read_text())
                for e in data.get("elites", []):
                    if _valid_genome(e):
                        all_elites.append(e)
            except (json.JSONDecodeError, OSError):
                pass
        all_elites.sort(key=lambda x: float(x.get("fitness", 0.0)), reverse=True)
        return all_elites[: self.pool_size]

    def update_contrib(self, worker_id: str, new_genomes: list[dict]) -> int:
        """Merge new_genomes into this worker's contribution file.

        Returns the total number of genomes now stored.
        """
        path = self.contrib_path(worker_id)
        existing: list[dict] = []
        if path.exists():
            try:
                existing = json.loads(path.read_text()).get("elites", [])
            except (json.JSONDecodeError, OSError):
                pass

        merged = [e for e in (existing + new_genomes) if _valid_genome(e)]
        merged.sort(key=lambda x: float(x.get("fitness", 0.0)), reverse=True)
        # Keep half the pool per worker so the merged total stays bounded.
        keep = max(1, self.pool_size // 2)
        merged = merged[:keep]

        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "worker_id": worker_id,
                    "elites": merged,
                    "updated_at": int(time.time()),
                },
                indent=2,
            )
        )
        tmp.replace(path)  # Atomic rename on POSIX.
        return len(merged)

    # ------------------------------------------------------------------
    # Seed-file lifecycle
    # ------------------------------------------------------------------

    def write_seed_file(
        self, worker_id: str, n: int = DEFAULT_SEED_COUNT
    ) -> Path | None:
        """Merge all contributions and write the seed file Godot will read.

        Returns the file path written, or None if the pool is empty.
        """
        top = self.merge_all()
        if not top:
            return None
        seeds = top[:n]
        out = self.seed_path(worker_id)
        out.write_text(
            json.dumps(
                {
                    "elites": seeds,
                    "generated_at": int(time.time()),
                },
                indent=2,
            )
        )
        return out

    def cleanup_seed_file(self, worker_id: str) -> None:
        """Remove the seed file after the run completes."""
        path = self.seed_path(worker_id)
        if path.exists():
            path.unlink()

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return summary statistics about the current pool."""
        elites = self.merge_all()
        contrib_files = list(self.user_dir.glob("elite_contrib_*.json"))
        return {
            "total_elites": len(elites),
            "contributor_count": len(contrib_files),
            "top_fitness": float(elites[0]["fitness"]) if elites else 0.0,
            "avg_fitness": (
                sum(float(e["fitness"]) for e in elites) / len(elites)
                if elites
                else 0.0
            ),
        }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _valid_genome(e: object) -> bool:
    """Return True if e looks like a serialised NeuralNetwork genome."""
    if not isinstance(e, dict):
        return False
    return all(k in e for k in ("weights_ih", "bias_h", "weights_ho", "bias_o", "fitness"))
