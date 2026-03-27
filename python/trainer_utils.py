"""Shared utilities for CPUTrainer and NeatCPUTrainer.

Extracts duplicated logic: pairings generation, Hall of Fame management,
and metrics file I/O.
"""

import json
import random
from pathlib import Path
from typing import Any


def generate_pairings(
    pop_size: int, tournament_opponents: int,
) -> list[tuple[int, int]]:
    """Generate round-robin pairings: each white plays tournament_opponents random blacks."""
    pairings = []
    for w_idx in range(pop_size):
        opponents = random.sample(range(pop_size), min(tournament_opponents, pop_size))
        for b_idx in opponents:
            pairings.append((w_idx, b_idx))
    return pairings


def update_hof(
    hof: list[tuple[float, Any]],
    population: Any,
    fitness: list[float],
    hof_max_size: int,
) -> list[tuple[float, Any]]:
    """Update Hall of Fame with best individual from current generation.

    Works with both numpy arrays (CPUTrainer) and JSON strings (NeatCPUTrainer).
    """
    best_idx = max(range(len(fitness)), key=lambda i: fitness[i])
    best_fit = fitness[best_idx]
    best_genome = population[best_idx]

    if len(hof) < hof_max_size or best_fit > hof[-1][0]:
        hof.append((best_fit, best_genome))
        hof.sort(key=lambda x: -x[0])
        if len(hof) > hof_max_size:
            hof.pop()

    return hof


class MetricsWriter:
    """Persistent file handle for per-generation metrics output.

    Keeps the file open for the lifetime of the training run instead of
    opening/closing on every generation (avoids N open() syscalls).
    """

    def __init__(self, path: Path):
        self._path = Path(path)
        self._file = None

    def write(self, metrics: dict) -> None:
        """Append a JSON-lines metrics entry."""
        try:
            if self._file is None:
                self._file = open(self._path, "a")  # noqa: SIM115
            self._file.write(json.dumps(metrics) + "\n")
            self._file.flush()
        except OSError:
            pass

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except OSError:
                pass
            self._file = None

    def __del__(self) -> None:
        self.close()
