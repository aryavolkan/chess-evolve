"""Tests for GlobalElitePool (overnight-agent/global_elite.py).

Verifies pool merging, contribution updates, seed file lifecycle, stats,
and genome validation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure overnight-agent is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "overnight-agent"))

from global_elite import GlobalElitePool, _valid_genome  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _genome(fitness: float = 1.0) -> dict:
    """Create a minimal valid genome dict."""
    return {
        "weights_ih": [0.1],
        "bias_h": [0.2],
        "weights_ho": [0.3],
        "bias_o": [0.4],
        "fitness": fitness,
    }


def _write_contrib(user_dir: Path, worker_id: str, elites: list[dict]):
    path = user_dir / f"elite_contrib_{worker_id}.json"
    path.write_text(json.dumps({"worker_id": worker_id, "elites": elites}))


# ===========================================================================
# _valid_genome
# ===========================================================================

class TestValidGenome:
    def test_valid(self):
        assert _valid_genome(_genome()) is True

    def test_missing_key(self):
        g = _genome()
        del g["fitness"]
        assert _valid_genome(g) is False

    def test_not_a_dict(self):
        assert _valid_genome("not a dict") is False
        assert _valid_genome(42) is False
        assert _valid_genome(None) is False

    def test_empty_dict(self):
        assert _valid_genome({}) is False


# ===========================================================================
# merge_all
# ===========================================================================

class TestMergeAll:
    def test_empty_directory(self, tmp_path):
        pool = GlobalElitePool(tmp_path)
        assert pool.merge_all() == []

    def test_merges_multiple_workers(self, tmp_path):
        _write_contrib(tmp_path, "w1", [_genome(5.0), _genome(3.0)])
        _write_contrib(tmp_path, "w2", [_genome(8.0), _genome(1.0)])
        pool = GlobalElitePool(tmp_path, pool_size=10)
        merged = pool.merge_all()
        assert len(merged) == 4
        assert merged[0]["fitness"] == 8.0  # sorted descending

    def test_respects_pool_size(self, tmp_path):
        _write_contrib(tmp_path, "w1", [_genome(float(i)) for i in range(10)])
        pool = GlobalElitePool(tmp_path, pool_size=3)
        merged = pool.merge_all()
        assert len(merged) == 3
        assert merged[0]["fitness"] == 9.0

    def test_ignores_invalid_genomes(self, tmp_path):
        path = tmp_path / "elite_contrib_bad.json"
        path.write_text(json.dumps({
            "elites": [{"not_valid": True}, _genome(5.0)],
        }))
        pool = GlobalElitePool(tmp_path)
        merged = pool.merge_all()
        assert len(merged) == 1
        assert merged[0]["fitness"] == 5.0

    def test_ignores_corrupt_json(self, tmp_path):
        path = tmp_path / "elite_contrib_corrupt.json"
        path.write_text("{invalid json")
        _write_contrib(tmp_path, "good", [_genome(7.0)])
        pool = GlobalElitePool(tmp_path)
        merged = pool.merge_all()
        assert len(merged) == 1


# ===========================================================================
# update_contrib
# ===========================================================================

class TestUpdateContrib:
    def test_creates_new_file(self, tmp_path):
        pool = GlobalElitePool(tmp_path, pool_size=10)
        count = pool.update_contrib("w1", [_genome(5.0), _genome(3.0)])
        assert count == 2
        assert pool.contrib_path("w1").exists()

    def test_merges_with_existing(self, tmp_path):
        pool = GlobalElitePool(tmp_path, pool_size=10)
        pool.update_contrib("w1", [_genome(5.0)])
        count = pool.update_contrib("w1", [_genome(8.0)])
        # Both should be present (pool_size=10, keep=5)
        assert count == 2

    def test_keeps_top_half(self, tmp_path):
        pool = GlobalElitePool(tmp_path, pool_size=4)
        genomes = [_genome(float(i)) for i in range(10)]
        count = pool.update_contrib("w1", genomes)
        assert count == 2  # pool_size=4 → keep=2

    def test_atomic_write(self, tmp_path):
        pool = GlobalElitePool(tmp_path, pool_size=10)
        pool.update_contrib("w1", [_genome(5.0)])
        # No .tmp file should remain
        assert not pool.contrib_path("w1").with_suffix(".tmp").exists()

    def test_filters_invalid_genomes(self, tmp_path):
        pool = GlobalElitePool(tmp_path, pool_size=10)
        count = pool.update_contrib("w1", [_genome(5.0), {"invalid": True}])
        assert count == 1


# ===========================================================================
# write_seed_file / cleanup_seed_file
# ===========================================================================

class TestSeedFileLifecycle:
    def test_returns_none_when_empty_pool(self, tmp_path):
        pool = GlobalElitePool(tmp_path)
        assert pool.write_seed_file("w1") is None

    def test_writes_seed_file(self, tmp_path):
        _write_contrib(tmp_path, "w1", [_genome(5.0)])
        pool = GlobalElitePool(tmp_path)
        path = pool.write_seed_file("w1")
        assert path is not None
        assert path.exists()
        data = json.loads(path.read_text())
        assert "elites" in data
        assert len(data["elites"]) == 1

    def test_seed_count_limit(self, tmp_path):
        _write_contrib(tmp_path, "w1", [_genome(float(i)) for i in range(10)])
        pool = GlobalElitePool(tmp_path, pool_size=20)
        path = pool.write_seed_file("w1", n=3)
        data = json.loads(path.read_text())
        assert len(data["elites"]) == 3

    def test_cleanup_removes_file(self, tmp_path):
        _write_contrib(tmp_path, "w1", [_genome(5.0)])
        pool = GlobalElitePool(tmp_path)
        path = pool.write_seed_file("w1")
        assert path.exists()
        pool.cleanup_seed_file("w1")
        assert not path.exists()

    def test_cleanup_noop_when_no_file(self, tmp_path):
        pool = GlobalElitePool(tmp_path)
        pool.cleanup_seed_file("w1")  # should not raise


# ===========================================================================
# stats
# ===========================================================================

class TestStats:
    def test_empty_pool_stats(self, tmp_path):
        pool = GlobalElitePool(tmp_path)
        s = pool.stats()
        assert s["total_elites"] == 0
        assert s["contributor_count"] == 0
        assert s["top_fitness"] == 0.0
        assert s["avg_fitness"] == 0.0

    def test_stats_with_data(self, tmp_path):
        _write_contrib(tmp_path, "w1", [_genome(10.0), _genome(6.0)])
        _write_contrib(tmp_path, "w2", [_genome(8.0)])
        pool = GlobalElitePool(tmp_path)
        s = pool.stats()
        assert s["total_elites"] == 3
        assert s["contributor_count"] == 2
        assert s["top_fitness"] == 10.0
        assert abs(s["avg_fitness"] - 8.0) < 1e-6


# ===========================================================================
# Path helpers
# ===========================================================================

class TestPaths:
    def test_contrib_path(self, tmp_path):
        pool = GlobalElitePool(tmp_path)
        assert pool.contrib_path("abc") == tmp_path / "elite_contrib_abc.json"

    def test_seed_path(self, tmp_path):
        pool = GlobalElitePool(tmp_path)
        assert pool.seed_path("abc") == tmp_path / "global_elite_abc.json"
