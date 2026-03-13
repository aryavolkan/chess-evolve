"""Tests for the NEAT topology visualizer."""
from __future__ import annotations

import os
import sys

# Add scripts/ to path so we can import the visualizer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from visualize_topology import (  # noqa: E402, I001
    ParsedGenome,
    assign_depths,
    build_html,
    build_population_html,
    compute_layout,
    compute_stats,
    parse_all_hof_genomes,
    parse_genome_dict,
    render_svg,
    serialize_layout,
    weight_color,
    weight_opacity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_genome(
    n_inputs: int = 3,
    n_outputs: int = 2,
    hidden_ids: list[int] | None = None,
    connections: list[tuple[int, int, float]] | None = None,
) -> ParsedGenome:
    """Build a small genome for testing.

    Inputs get IDs 0..n_inputs-1, outputs get IDs n_inputs..n_inputs+n_outputs-1.
    Hidden nodes get the IDs in hidden_ids (default empty).
    Connections are (in_id, out_id, weight) tuples; all enabled.
    """
    if hidden_ids is None:
        hidden_ids = []
    if connections is None:
        connections = []

    nodes = []
    for i in range(n_inputs):
        nodes.append({"id": i, "node_type": 0, "bias": 0.0})
    for i, hid in enumerate(hidden_ids):
        nodes.append({"id": hid, "node_type": 1, "bias": 0.1 * (i + 1)})
    for i in range(n_outputs):
        nodes.append({"id": n_inputs + i, "node_type": 2, "bias": 0.0})

    conns = []
    for i, (src, dst, w) in enumerate(connections):
        conns.append({
            "in_id": src, "out_id": dst, "weight": w,
            "enabled": True, "innovation": i,
        })

    return parse_genome_dict({"nodes": nodes, "connections": conns, "fitness": 1.0})


# ---------------------------------------------------------------------------
# Depth assignment
# ---------------------------------------------------------------------------

class TestDepthAssignment:
    def test_linear_chain(self):
        """Input -> A -> B -> Output: A=depth1, B=depth2."""
        g = _make_genome(
            n_inputs=1, n_outputs=1, hidden_ids=[10, 11],
            connections=[(0, 10, 1.0), (10, 11, 1.0), (11, 1, 1.0)],
        )
        max_d = assign_depths(g)
        node_map = {n.id: n for n in g.nodes}
        assert node_map[10].depth == 1
        assert node_map[11].depth == 2
        assert max_d == 2

    def test_parallel_paths(self):
        """Two hidden nodes at same depth."""
        g = _make_genome(
            n_inputs=1, n_outputs=1, hidden_ids=[10, 11],
            connections=[(0, 10, 1.0), (0, 11, 1.0), (10, 1, 1.0), (11, 1, 1.0)],
        )
        max_d = assign_depths(g)
        node_map = {n.id: n for n in g.nodes}
        assert node_map[10].depth == 1
        assert node_map[11].depth == 1
        assert max_d == 1

    def test_cycle(self):
        """A -> B -> A cycle: both get finite depth, no infinite loop."""
        g = _make_genome(
            n_inputs=1, n_outputs=1, hidden_ids=[10, 11],
            connections=[
                (0, 10, 1.0), (10, 11, 1.0), (11, 10, 1.0), (11, 1, 1.0),
            ],
        )
        max_d = assign_depths(g)
        node_map = {n.id: n for n in g.nodes}
        assert node_map[10].depth >= 1
        assert node_map[11].depth >= 1
        assert max_d < 100  # didn't diverge

    def test_disconnected_hidden(self):
        """Hidden node with no input connections gets fallback depth."""
        g = _make_genome(
            n_inputs=1, n_outputs=1, hidden_ids=[10],
            connections=[(10, 1, 1.0)],  # 10 not reachable from input 0
        )
        max_d = assign_depths(g)
        node_map = {n.id: n for n in g.nodes}
        assert node_map[10].depth == 1  # fallback

    def test_no_hidden(self):
        """Genome with only input->output connections."""
        g = _make_genome(
            n_inputs=2, n_outputs=2,
            connections=[(0, 2, 1.0), (1, 3, 1.0)],
        )
        max_d = assign_depths(g)
        assert max_d == 0


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestStatistics:
    def test_fan_counts(self):
        g = _make_genome(
            n_inputs=2, n_outputs=1, hidden_ids=[10],
            connections=[(0, 10, 1.0), (1, 10, 0.5), (10, 2, -1.0)],
        )
        assign_depths(g)
        stats = compute_stats(g, 1)
        assert stats.n_inputs == 2
        assert stats.n_hidden == 1
        assert stats.n_outputs == 1
        assert stats.n_enabled == 3
        assert stats.n_input_to_hidden == 2
        assert stats.n_hidden_to_output == 1

    def test_connection_categories(self):
        g = _make_genome(
            n_inputs=1, n_outputs=1, hidden_ids=[10, 11],
            connections=[
                (0, 10, 1.0),   # ih
                (10, 11, 0.5),  # hh
                (11, 1, -1.0),  # ho
                (0, 1, 0.3),    # io (skip)
            ],
        )
        assign_depths(g)
        stats = compute_stats(g, 2)
        assert stats.n_input_to_hidden == 1
        assert stats.n_hidden_to_hidden == 1
        assert stats.n_hidden_to_output == 1
        assert stats.n_input_to_output == 1

    def test_weight_stats(self):
        g = _make_genome(
            n_inputs=1, n_outputs=1, hidden_ids=[10],
            connections=[(0, 10, 2.0), (10, 1, -1.0)],
        )
        assign_depths(g)
        stats = compute_stats(g, 1)
        assert stats.weight_min == -1.0
        assert stats.weight_max == 2.0
        assert abs(stats.weight_mean - 0.5) < 0.01


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

class TestLayout:
    def test_no_overlapping_nodes(self):
        g = _make_genome(
            n_inputs=2, n_outputs=2, hidden_ids=[10, 11, 12],
            connections=[
                (0, 10, 1.0), (0, 11, 1.0), (0, 12, 1.0),
                (10, 2, 1.0), (11, 3, 1.0), (12, 2, 1.0),
            ],
        )
        assign_depths(g)
        stats = compute_stats(g, 1)
        layout = compute_layout(g, stats, 1)
        positions = [(n.x, n.y) for n in layout.nodes]
        # All positions should be distinct
        assert len(set(positions)) == len(positions)

    def test_same_depth_same_x(self):
        g = _make_genome(
            n_inputs=1, n_outputs=1, hidden_ids=[10, 11],
            connections=[(0, 10, 1.0), (0, 11, 1.0), (10, 1, 1.0), (11, 1, 1.0)],
        )
        assign_depths(g)
        stats = compute_stats(g, 1)
        layout = compute_layout(g, stats, 1)
        xs = [n.x for n in layout.nodes if n.node_type == 1]
        # Both at depth 1, should have same x
        assert len(set(xs)) == 1

    def test_different_depth_different_x(self):
        g = _make_genome(
            n_inputs=1, n_outputs=1, hidden_ids=[10, 11],
            connections=[(0, 10, 1.0), (10, 11, 1.0), (11, 1, 1.0)],
        )
        assign_depths(g)
        stats = compute_stats(g, 2)
        layout = compute_layout(g, stats, 2)
        node_map = {n.id: n for n in layout.nodes}
        assert node_map[10].x < node_map[11].x


# ---------------------------------------------------------------------------
# Weight coloring
# ---------------------------------------------------------------------------

class TestWeightColor:
    def test_positive_is_green(self):
        c = weight_color(1.5)
        # Green channel should be prominent
        assert "rgb(34," in c

    def test_negative_is_red(self):
        c = weight_color(-1.5)
        # Red channel should be prominent
        assert ",68,68)" in c

    def test_zero_is_green_low_intensity(self):
        c = weight_color(0.0)
        assert "rgb(34," in c

    def test_opacity_scales_with_magnitude(self):
        o_small = weight_opacity(0.1)
        o_large = weight_opacity(2.0)
        assert o_large > o_small


# ---------------------------------------------------------------------------
# SVG and HTML output
# ---------------------------------------------------------------------------

class TestOutput:
    def _make_small_genome_html(self):
        g = _make_genome(
            n_inputs=2, n_outputs=2, hidden_ids=[10, 11],
            connections=[
                (0, 10, 1.0), (1, 11, -0.5),
                (10, 11, 0.3), (10, 2, 1.0), (11, 3, -1.0),
            ],
        )
        max_d = assign_depths(g)
        stats = compute_stats(g, max_d)
        layout = compute_layout(g, stats, max_d)
        svg = render_svg(layout, stats, "white")
        html = build_html(svg, stats, "white")
        return svg, html

    def test_svg_contains_nodes(self):
        svg, _ = self._make_small_genome_html()
        assert 'class="node"' in svg
        assert 'data-id="10"' in svg
        assert 'data-id="11"' in svg

    def test_svg_contains_connections(self):
        svg, _ = self._make_small_genome_html()
        assert 'class="conn' in svg

    def test_svg_contains_summary_bars(self):
        svg, _ = self._make_small_genome_html()
        assert "2 inputs" in svg
        assert "2 outputs" in svg

    def test_html_self_contained(self):
        _, html = self._make_small_genome_html()
        assert "<style>" in html
        assert "<script>" in html
        # No external CDN/resource URLs (xmlns is fine)
        for line in html.splitlines():
            if "http://" in line or "https://" in line:
                assert "xmlns" in line, f"External URL found: {line.strip()}"

    def test_html_contains_stats(self):
        _, html = self._make_small_genome_html()
        assert "Hidden nodes" in html
        assert "Enabled conns" in html
        assert "Network depth" in html

    def test_html_contains_tooltip_js(self):
        _, html = self._make_small_genome_html()
        assert "tooltip" in html
        assert "mouseenter" in html


# ---------------------------------------------------------------------------
# Integration test with real genome
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Population mode
# ---------------------------------------------------------------------------

class TestParseAllHofGenomes:
    def test_loads_all_genomes(self, tmp_path):
        """parse_all_hof_genomes loads all HOF genome strings."""
        genome1 = {
            "nodes": [
                {"id": 0, "node_type": 0, "bias": 0.0},
                {"id": 1, "node_type": 2, "bias": 0.0},
                {"id": 10, "node_type": 1, "bias": 0.1},
            ],
            "connections": [
                {"in_id": 0, "out_id": 10, "weight": 1.0, "enabled": True, "innovation": 0},
                {"in_id": 10, "out_id": 1, "weight": -0.5, "enabled": True, "innovation": 1},
            ],
            "fitness": 5.0,
        }
        genome2 = {
            "nodes": [
                {"id": 0, "node_type": 0, "bias": 0.0},
                {"id": 1, "node_type": 2, "bias": 0.0},
            ],
            "connections": [
                {"in_id": 0, "out_id": 1, "weight": 0.3, "enabled": True, "innovation": 0},
            ],
            "fitness": 2.0,
        }
        import json
        data = {
            "white": json.dumps(genome1),
            "white_hof": [json.dumps(genome1), json.dumps(genome2)],
        }
        p = tmp_path / "genomes.json"
        p.write_text(json.dumps(data))

        results = parse_all_hof_genomes(str(p), "white")
        assert len(results) == 2
        assert results[0].fitness == 5.0
        assert results[1].fitness == 2.0
        assert len(results[0].nodes) == 3
        assert len(results[1].nodes) == 2


class TestSerializeLayout:
    def test_produces_expected_keys(self):
        """serialize_layout produces all required top-level and stats keys."""
        g = _make_genome(
            n_inputs=2, n_outputs=1, hidden_ids=[10],
            connections=[(0, 10, 1.0), (10, 2, -0.5)],
        )
        max_d = assign_depths(g)
        stats = compute_stats(g, max_d)
        layout = compute_layout(g, stats, max_d)
        result = serialize_layout(layout, stats, 7, 3.14)

        # Compact top-level keys
        assert result["i"] == 7
        assert result["f"] == 3.14
        assert isinstance(result["n"], list)  # nodes
        assert isinstance(result["hh"], list)  # hidden-to-hidden
        assert "io" in result  # input-to-output count
        assert len(result["ib"]) == 4
        assert len(result["ob"]) == 4
        assert isinstance(result["s"], dict)

        # Compact stats keys
        expected_stat_keys = {
            "nh", "ne", "nd", "md", "afi", "afo",
            "nih", "nhh", "nho", "nio",
            "wn", "wx", "wm", "ws", "ni", "no",
        }
        assert set(result["s"].keys()) == expected_stat_keys

    def test_node_format(self):
        """Each serialized node is a compact array with 10 elements."""
        g = _make_genome(
            n_inputs=1, n_outputs=1, hidden_ids=[10],
            connections=[(0, 10, 1.0), (10, 1, -1.0)],
        )
        max_d = assign_depths(g)
        stats = compute_stats(g, max_d)
        layout = compute_layout(g, stats, max_d)
        result = serialize_layout(layout, stats, 0, 1.0)

        assert len(result["n"]) == 1
        node = result["n"][0]
        # [id, x, y, r, bias, depth, fan_in, fan_out, ih_w, ho_w]
        assert len(node) == 10
        assert node[0] == 10  # id

    def test_hh_connection_format(self):
        """HH connections use [src_idx, dst_idx, weight] compact format."""
        g = _make_genome(
            n_inputs=1, n_outputs=1, hidden_ids=[10, 11],
            connections=[(0, 10, 1.0), (10, 11, 0.5), (11, 1, -1.0)],
        )
        max_d = assign_depths(g)
        stats = compute_stats(g, max_d)
        layout = compute_layout(g, stats, max_d)
        result = serialize_layout(layout, stats, 0, 1.0)

        assert len(result["hh"]) == 1  # one hh connection
        hh = result["hh"][0]
        assert len(hh) == 3  # [src_idx, dst_idx, weight]
        assert hh[2] == 0.5  # weight


class TestBuildPopulationHtml:
    def _make_sample_layouts(self, n=3):
        layouts = []
        for i in range(n):
            g = _make_genome(
                n_inputs=2, n_outputs=1, hidden_ids=[10 + i],
                connections=[(0, 10 + i, 1.0), (10 + i, 2, -0.5)],
            )
            max_d = assign_depths(g)
            stats = compute_stats(g, max_d)
            layout = compute_layout(g, stats, max_d)
            layouts.append(serialize_layout(layout, stats, i, float(i + 1)))
        return layouts

    def test_html_is_valid(self):
        """build_population_html produces valid HTML with required elements."""
        layouts = self._make_sample_layouts()
        html = build_population_html(layouts, "white")
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html
        assert "<script>" in html
        assert "<style>" in html

    def test_genomes_data_embedded(self):
        """The GENOMES JSON array is embedded in the output."""
        layouts = self._make_sample_layouts()
        html = build_population_html(layouts, "white")
        assert "const GENOMES" in html
        assert '"i":0' in html
        assert '"i":2' in html

    def test_title_contains_color_and_count(self):
        layouts = self._make_sample_layouts(5)
        html = build_population_html(layouts, "black")
        assert "NEAT Population: black (5 genomes)" in html

    def test_slider_present(self):
        layouts = self._make_sample_layouts()
        html = build_population_html(layouts, "white")
        assert 'id="genome-slider"' in html
        assert 'max="2"' in html  # 3 genomes -> max=2

    def test_no_external_urls(self):
        """HTML is self-contained (no external URLs except SVG namespace)."""
        layouts = self._make_sample_layouts()
        html = build_population_html(layouts, "white")
        for line in html.splitlines():
            if "http://" in line or "https://" in line:
                assert "w3.org" in line, f"External URL found: {line.strip()}"

    def test_population_charts_present(self):
        """The three population overview charts are in the output."""
        layouts = self._make_sample_layouts()
        html = build_population_html(layouts, "white")
        assert "Hidden Node Count" in html
        assert "Network Depth" in html
        assert "Fitness" in html


# ---------------------------------------------------------------------------
# Integration test with real genome
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_real_genome_if_available(self, tmp_path):
        """Load real genome file if present; verify no crash."""
        candidates = [
            "neat_best_genomes_128.json.bak",
            "neat_best_genomes.json",
        ]
        genome_path = None
        for name in candidates:
            p = os.path.join(os.path.dirname(__file__), "..", "..", name)
            if os.path.exists(p):
                genome_path = p
                break

        if genome_path is None:
            import pytest
            pytest.skip("No genome file available for integration test")

        from visualize_topology import parse_genome_file
        genome = parse_genome_file(genome_path, "white")
        assert len(genome.nodes) > 100

        max_d = assign_depths(genome)
        stats = compute_stats(genome, max_d)
        assert stats.n_hidden > 0

        layout = compute_layout(genome, stats, max_d)
        assert len(layout.nodes) > 0

        svg = render_svg(layout, stats, "white")
        html = build_html(svg, stats, "white")

        out = tmp_path / "test_topology.html"
        out.write_text(html)
        assert out.stat().st_size > 10000
