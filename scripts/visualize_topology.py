#!/usr/bin/env python3
"""
Visualize NEAT genome topology as an interactive HTML file.

Generates a self-contained HTML page with SVG network diagram showing
hidden node topology, connection weight patterns, and summary statistics.

Usage:
    python scripts/visualize_topology.py --genome neat_best_genomes.json --color white
    python scripts/visualize_topology.py --genome neat_best_genomes.json --color black --hof 0
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Node:
    id: int
    node_type: int  # 0=input, 1=hidden, 2=output
    bias: float
    depth: int = 0
    fan_in: int = 0
    fan_out: int = 0


@dataclass
class Connection:
    in_id: int
    out_id: int
    weight: float
    enabled: bool
    innovation: int


@dataclass
class ParsedGenome:
    nodes: list[Node]
    connections: list[Connection]
    fitness: float = 0.0


@dataclass
class TopologyStats:
    n_inputs: int = 0
    n_hidden: int = 0
    n_outputs: int = 0
    n_enabled: int = 0
    n_disabled: int = 0
    n_input_to_hidden: int = 0
    n_hidden_to_hidden: int = 0
    n_hidden_to_output: int = 0
    n_input_to_output: int = 0
    max_depth: int = 0
    avg_fan_in: float = 0.0
    avg_fan_out: float = 0.0
    weight_min: float = 0.0
    weight_max: float = 0.0
    weight_mean: float = 0.0
    weight_stdev: float = 0.0


@dataclass
class LayoutNode:
    id: int
    x: float
    y: float
    radius: float
    bias: float
    depth: int
    fan_in: int
    fan_out: int
    node_type: int


@dataclass
class LayoutConnection:
    x1: float
    y1: float
    x2: float
    y2: float
    weight: float
    category: str  # "hh", "ih", "ho", "io"


@dataclass
class Layout:
    nodes: list[LayoutNode] = field(default_factory=list)
    connections: list[LayoutConnection] = field(default_factory=list)
    input_bar: tuple[float, float, float, float] = (0, 0, 0, 0)  # x, y, w, h
    output_bar: tuple[float, float, float, float] = (0, 0, 0, 0)
    width: float = 1400
    height: float = 800


# ---------------------------------------------------------------------------
# GenomeParser
# ---------------------------------------------------------------------------

def parse_genome_file(path: str, color: str, hof_index: int | None = None) -> ParsedGenome:
    """Load a genome from neat_best_genomes.json format."""
    data = json.loads(Path(path).read_text())

    if hof_index is not None:
        hof_key = f"{color}_hof"
        if hof_key not in data:
            print(f"Error: '{hof_key}' not found in genome file", file=sys.stderr)
            sys.exit(1)
        hof = data[hof_key]
        if hof_index >= len(hof):
            print(f"Error: HOF index {hof_index} out of range (0-{len(hof)-1})", file=sys.stderr)
            sys.exit(1)
        genome_str = hof[hof_index]
    else:
        if color not in data:
            print(f"Error: '{color}' not found in genome file", file=sys.stderr)
            sys.exit(1)
        genome_str = data[color]

    genome = json.loads(genome_str) if isinstance(genome_str, str) else genome_str

    nodes = [
        Node(id=n["id"], node_type=n["node_type"], bias=n.get("bias", 0.0))
        for n in genome["nodes"]
    ]
    connections = [
        Connection(
            in_id=c["in_id"], out_id=c["out_id"], weight=c["weight"],
            enabled=c["enabled"], innovation=c["innovation"],
        )
        for c in genome["connections"]
    ]
    return ParsedGenome(
        nodes=nodes, connections=connections,
        fitness=genome.get("fitness", 0.0),
    )


def parse_genome_dict(genome: dict) -> ParsedGenome:
    """Parse a genome from an already-loaded dict (for testing)."""
    nodes = [
        Node(id=n["id"], node_type=n["node_type"], bias=n.get("bias", 0.0))
        for n in genome["nodes"]
    ]
    connections = [
        Connection(
            in_id=c["in_id"], out_id=c["out_id"], weight=c["weight"],
            enabled=c["enabled"], innovation=c["innovation"],
        )
        for c in genome["connections"]
    ]
    return ParsedGenome(nodes=nodes, connections=connections, fitness=genome.get("fitness", 0.0))


# ---------------------------------------------------------------------------
# TopologyAnalyzer
# ---------------------------------------------------------------------------

def assign_depths(genome: ParsedGenome) -> int:
    """Assign BFS depth to hidden nodes. Returns max hidden depth."""
    node_map = {n.id: n for n in genome.nodes}
    adj: dict[int, list[int]] = defaultdict(list)

    for c in genome.connections:
        if c.enabled:
            adj[c.in_id].append(c.out_id)

    input_ids = {n.id for n in genome.nodes if n.node_type == 0}
    output_ids = {n.id for n in genome.nodes if n.node_type == 2}
    hidden_ids = {n.id for n in genome.nodes if n.node_type == 1}

    # BFS from all input nodes
    visited: dict[int, int] = {}
    queue: deque[tuple[int, int]] = deque()
    for nid in input_ids:
        visited[nid] = 0
        queue.append((nid, 0))

    while queue:
        nid, d = queue.popleft()
        for neighbor in adj.get(nid, []):
            if neighbor not in visited and neighbor not in output_ids:
                visited[neighbor] = d + 1
                queue.append((neighbor, d + 1))

    # Assign depths
    max_depth = 0
    for nid in hidden_ids:
        depth = visited.get(nid, -1)
        if depth <= 0:
            depth = 1  # fallback for unreached hidden nodes
        node_map[nid].depth = depth
        max_depth = max(max_depth, depth)

    # Assign output nodes depth = max_depth + 1
    out_depth = max_depth + 1
    for nid in output_ids:
        node_map[nid].depth = out_depth

    return max_depth


def compute_stats(genome: ParsedGenome, max_depth: int) -> TopologyStats:
    """Compute topology statistics."""
    node_map = {n.id: n for n in genome.nodes}
    input_ids = {n.id for n in genome.nodes if n.node_type == 0}
    hidden_ids = {n.id for n in genome.nodes if n.node_type == 1}
    output_ids = {n.id for n in genome.nodes if n.node_type == 2}

    enabled = [c for c in genome.connections if c.enabled]
    disabled = [c for c in genome.connections if not c.enabled]

    # Compute fan-in/fan-out for hidden nodes
    for c in enabled:
        src = node_map.get(c.in_id)
        dst = node_map.get(c.out_id)
        if src:
            src.fan_out += 1
        if dst:
            dst.fan_in += 1

    hidden_nodes = [n for n in genome.nodes if n.node_type == 1]

    # Connection categories
    n_ih = n_hh = n_ho = n_io = 0
    for c in enabled:
        src_type = node_map[c.in_id].node_type if c.in_id in node_map else -1
        dst_type = node_map[c.out_id].node_type if c.out_id in node_map else -1
        if src_type == 0 and dst_type == 1:
            n_ih += 1
        elif src_type == 1 and dst_type == 1:
            n_hh += 1
        elif src_type == 1 and dst_type == 2:
            n_ho += 1
        elif src_type == 0 and dst_type == 2:
            n_io += 1

    weights = [c.weight for c in enabled]

    return TopologyStats(
        n_inputs=len(input_ids),
        n_hidden=len(hidden_ids),
        n_outputs=len(output_ids),
        n_enabled=len(enabled),
        n_disabled=len(disabled),
        n_input_to_hidden=n_ih,
        n_hidden_to_hidden=n_hh,
        n_hidden_to_output=n_ho,
        n_input_to_output=n_io,
        max_depth=max_depth,
        avg_fan_in=statistics.mean([n.fan_in for n in hidden_nodes]) if hidden_nodes else 0,
        avg_fan_out=statistics.mean([n.fan_out for n in hidden_nodes]) if hidden_nodes else 0,
        weight_min=min(weights) if weights else 0,
        weight_max=max(weights) if weights else 0,
        weight_mean=statistics.mean(weights) if weights else 0,
        weight_stdev=statistics.stdev(weights) if len(weights) > 1 else 0,
    )


# ---------------------------------------------------------------------------
# LayoutEngine
# ---------------------------------------------------------------------------

def compute_layout(
    genome: ParsedGenome, stats: TopologyStats, max_depth: int,
    max_connections: int = 2000,
) -> Layout:
    """Compute x/y positions for all nodes and connections."""
    canvas_w, canvas_h = 1400, 800
    margin_x, margin_y = 120, 60
    bar_w = 40

    layout = Layout(width=canvas_w, height=canvas_h)

    # Input/output summary bars
    usable_h = canvas_h - 2 * margin_y
    layout.input_bar = (margin_x - bar_w, margin_y, bar_w, usable_h)
    layout.output_bar = (canvas_w - margin_x, margin_y, bar_w, usable_h)

    # Group hidden nodes by depth
    hidden = [n for n in genome.nodes if n.node_type == 1]
    by_depth: dict[int, list[Node]] = defaultdict(list)
    for n in hidden:
        by_depth[n.depth].append(n)

    # Sort within each depth by id for deterministic layout
    for d in by_depth:
        by_depth[d].sort(key=lambda n: n.id)

    # Position hidden nodes
    usable_w = canvas_w - 2 * margin_x - 2 * bar_w
    x_start = margin_x + bar_w + 20
    n_depths = max(1, max_depth)

    node_positions: dict[int, tuple[float, float]] = {}

    for depth, nodes in by_depth.items():
        x = x_start + (depth - 1) / n_depths * usable_w if n_depths > 1 else x_start + usable_w / 2
        n_nodes = len(nodes)
        for i, node in enumerate(nodes):
            y = margin_y + (i + 0.5) / n_nodes * usable_h if n_nodes > 0 else canvas_h / 2
            max_connectivity = max(1, max(n.fan_in + n.fan_out for n in hidden)) if hidden else 1
            radius = 3 + 5 * (node.fan_in + node.fan_out) / max_connectivity
            layout.nodes.append(LayoutNode(
                id=node.id, x=x, y=y, radius=radius,
                bias=node.bias, depth=node.depth,
                fan_in=node.fan_in, fan_out=node.fan_out,
                node_type=1,
            ))
            node_positions[node.id] = (x, y)

    # Build connections
    node_map = {n.id: n for n in genome.nodes}
    enabled = [c for c in genome.connections if c.enabled]

    # Input bar center x, output bar center x
    ib_cx = layout.input_bar[0] + layout.input_bar[2] / 2
    ob_cx = layout.output_bar[0] + layout.output_bar[2] / 2

    # For input->hidden and hidden->output: aggregate by hidden node
    ih_counts: dict[int, int] = defaultdict(int)
    ho_counts: dict[int, int] = defaultdict(int)
    io_count = 0

    hh_conns: list[Connection] = []

    for c in enabled:
        src = node_map.get(c.in_id)
        dst = node_map.get(c.out_id)
        if not src or not dst:
            continue
        if src.node_type == 0 and dst.node_type == 1:
            ih_counts[dst.id] += 1
        elif src.node_type == 1 and dst.node_type == 1:
            hh_conns.append(c)
        elif src.node_type == 1 and dst.node_type == 2:
            ho_counts[src.id] += 1
        elif src.node_type == 0 and dst.node_type == 2:
            io_count += 1

    # Input->hidden aggregate lines
    max_ih = max(ih_counts.values()) if ih_counts else 1
    for nid, count in ih_counts.items():
        if nid in node_positions:
            nx, ny = node_positions[nid]
            # y position on input bar proportional to hidden node y
            layout.connections.append(LayoutConnection(
                x1=ib_cx + bar_w / 2, y1=ny, x2=nx, y2=ny,
                weight=count / max_ih, category="ih",
            ))

    # Hidden->output aggregate lines
    max_ho = max(ho_counts.values()) if ho_counts else 1
    for nid, count in ho_counts.items():
        if nid in node_positions:
            nx, ny = node_positions[nid]
            layout.connections.append(LayoutConnection(
                x1=nx, y1=ny, x2=ob_cx - bar_w / 2, y2=ny,
                weight=count / max_ho, category="ho",
            ))

    # Hidden->hidden individual lines (capped)
    if len(hh_conns) <= max_connections:
        for c in hh_conns:
            if c.in_id in node_positions and c.out_id in node_positions:
                x1, y1 = node_positions[c.in_id]
                x2, y2 = node_positions[c.out_id]
                layout.connections.append(LayoutConnection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    weight=c.weight, category="hh",
                ))
    else:
        # Aggregate by depth-pair when too many
        depth_pair_weights: dict[tuple[int, int], list[float]] = defaultdict(list)
        for c in hh_conns:
            src = node_map.get(c.in_id)
            dst = node_map.get(c.out_id)
            if src and dst:
                depth_pair_weights[(src.depth, dst.depth)].append(c.weight)
        for (d1, d2), weights in depth_pair_weights.items():
            nodes_d1 = by_depth.get(d1, [])
            nodes_d2 = by_depth.get(d2, [])
            if nodes_d1 and nodes_d2:
                cx1 = statistics.mean([node_positions[n.id][0] for n in nodes_d1 if n.id in node_positions])
                cy1 = statistics.mean([node_positions[n.id][1] for n in nodes_d1 if n.id in node_positions])
                cx2 = statistics.mean([node_positions[n.id][0] for n in nodes_d2 if n.id in node_positions])
                cy2 = statistics.mean([node_positions[n.id][1] for n in nodes_d2 if n.id in node_positions])
                layout.connections.append(LayoutConnection(
                    x1=cx1, y1=cy1, x2=cx2, y2=cy2,
                    weight=statistics.mean(weights), category="hh",
                ))

    # Input->output skip indicator
    if io_count > 0:
        layout.connections.append(LayoutConnection(
            x1=ib_cx + bar_w / 2, y1=canvas_h / 2,
            x2=ob_cx - bar_w / 2, y2=canvas_h / 2,
            weight=io_count, category="io",
        ))

    return layout


# ---------------------------------------------------------------------------
# SVGRenderer
# ---------------------------------------------------------------------------

def weight_color(w: float) -> str:
    """Map weight to green (positive) or red (negative) hex color."""
    if w >= 0:
        intensity = min(255, int(80 + 175 * min(1.0, abs(w) / 2.0)))
        return f"rgb(34,{intensity},94)"
    else:
        intensity = min(255, int(80 + 175 * min(1.0, abs(w) / 2.0)))
        return f"rgb({intensity},68,68)"


def weight_opacity(w: float) -> float:
    """Map |weight| to opacity."""
    return max(0.1, min(0.85, abs(w) / 2.5))


def render_svg(layout: Layout, stats: TopologyStats, color: str) -> str:
    """Generate SVG markup for the network topology."""
    parts: list[str] = []
    w, h = layout.width, layout.height

    parts.append(f'<svg id="topo-svg" viewBox="0 0 {w} {h}" '
                 f'xmlns="http://www.w3.org/2000/svg" style="width:100%;height:100%">')

    # Background
    parts.append(f'<rect width="{w}" height="{h}" fill="#0f0f23"/>')

    # Connections (draw before nodes so nodes are on top)
    for conn in layout.connections:
        if conn.category == "hh":
            c = weight_color(conn.weight)
            o = weight_opacity(conn.weight)
            parts.append(
                f'<line class="conn hh" x1="{conn.x1:.1f}" y1="{conn.y1:.1f}" '
                f'x2="{conn.x2:.1f}" y2="{conn.y2:.1f}" '
                f'stroke="{c}" stroke-opacity="{o:.2f}" stroke-width="1" '
                f'data-weight="{conn.weight:.3f}"/>'
            )
        elif conn.category == "ih":
            o = max(0.08, min(0.5, conn.weight))
            parts.append(
                f'<line class="conn ih" x1="{conn.x1:.1f}" y1="{conn.y1:.1f}" '
                f'x2="{conn.x2:.1f}" y2="{conn.y2:.1f}" '
                f'stroke="#4a90d9" stroke-opacity="{o:.2f}" stroke-width="1"/>'
            )
        elif conn.category == "ho":
            o = max(0.08, min(0.5, conn.weight))
            parts.append(
                f'<line class="conn ho" x1="{conn.x1:.1f}" y1="{conn.y1:.1f}" '
                f'x2="{conn.x2:.1f}" y2="{conn.y2:.1f}" '
                f'stroke="#d9904a" stroke-opacity="{o:.2f}" stroke-width="1"/>'
            )
        elif conn.category == "io":
            parts.append(
                f'<line class="conn io" x1="{conn.x1:.1f}" y1="{conn.y1:.1f}" '
                f'x2="{conn.x2:.1f}" y2="{conn.y2:.1f}" '
                f'stroke="#888" stroke-opacity="0.3" stroke-width="2" '
                f'stroke-dasharray="8,4"/>'
            )

    # Input summary bar
    ix, iy, iw, ih_ = layout.input_bar
    parts.append(f'<rect x="{ix}" y="{iy}" width="{iw}" height="{ih_}" '
                 f'fill="#4a90d9" rx="5" opacity="0.8"/>')
    parts.append(f'<text x="{ix + iw/2}" y="{iy - 10}" text-anchor="middle" '
                 f'fill="#4a90d9" font-size="13" font-family="monospace">'
                 f'{stats.n_inputs} inputs</text>')

    # Output summary bar
    ox, oy, ow, oh = layout.output_bar
    parts.append(f'<rect x="{ox}" y="{oy}" width="{ow}" height="{oh}" '
                 f'fill="#d94a4a" rx="5" opacity="0.8"/>')
    parts.append(f'<text x="{ox + ow/2}" y="{oy - 10}" text-anchor="middle" '
                 f'fill="#d94a4a" font-size="13" font-family="monospace">'
                 f'{stats.n_outputs} outputs</text>')

    # Hidden nodes
    for node in layout.nodes:
        if node.node_type == 1:
            # Color by bias: blue-purple gradient
            bias_norm = max(-1, min(1, node.bias))
            r = int(100 + 80 * max(0, bias_norm))
            b = int(100 + 80 * max(0, -bias_norm))
            g = 80
            parts.append(
                f'<circle class="node" cx="{node.x:.1f}" cy="{node.y:.1f}" '
                f'r="{node.radius:.1f}" fill="rgb({r},{g},{b})" '
                f'stroke="#444" stroke-width="0.5" '
                f'data-id="{node.id}" data-bias="{node.bias:.4f}" '
                f'data-depth="{node.depth}" '
                f'data-fanin="{node.fan_in}" data-fanout="{node.fan_out}"/>'
            )

    # Depth labels along bottom
    depths_seen: set[int] = set()
    for node in layout.nodes:
        if node.node_type == 1 and node.depth not in depths_seen:
            depths_seen.add(node.depth)
            parts.append(
                f'<text x="{node.x:.1f}" y="{h - 15}" text-anchor="middle" '
                f'fill="#666" font-size="10" font-family="monospace">d{node.depth}</text>'
            )

    # Title
    parts.append(
        f'<text x="{w/2}" y="25" text-anchor="middle" fill="#ccc" '
        f'font-size="16" font-family="monospace" font-weight="bold">'
        f'NEAT Topology: {color}</text>'
    )

    parts.append('</svg>')
    return '\n'.join(parts)


# ---------------------------------------------------------------------------
# HTMLBuilder
# ---------------------------------------------------------------------------

def build_html(svg: str, stats: TopologyStats, color: str) -> str:
    """Wrap SVG in self-contained HTML with CSS and JS."""
    stats_html = f"""
    <div id="stats-panel">
      <h3>Topology Stats</h3>
      <table>
        <tr><td>Hidden nodes</td><td>{stats.n_hidden}</td></tr>
        <tr><td>Enabled conns</td><td>{stats.n_enabled:,}</td></tr>
        <tr><td>Disabled conns</td><td>{stats.n_disabled:,}</td></tr>
        <tr><td>Network depth</td><td>{stats.max_depth}</td></tr>
        <tr><td>Avg fan-in</td><td>{stats.avg_fan_in:.1f}</td></tr>
        <tr><td>Avg fan-out</td><td>{stats.avg_fan_out:.1f}</td></tr>
        <tr class="sep"><td colspan="2"></td></tr>
        <tr><td>Input &rarr; Hidden</td><td>{stats.n_input_to_hidden:,}</td></tr>
        <tr><td>Hidden &rarr; Hidden</td><td>{stats.n_hidden_to_hidden:,}</td></tr>
        <tr><td>Hidden &rarr; Output</td><td>{stats.n_hidden_to_output:,}</td></tr>
        <tr><td>Input &rarr; Output</td><td>{stats.n_input_to_output:,}</td></tr>
        <tr class="sep"><td colspan="2"></td></tr>
        <tr><td>Weight range</td><td>[{stats.weight_min:.2f}, {stats.weight_max:.2f}]</td></tr>
        <tr><td>Weight mean</td><td>{stats.weight_mean:.3f}</td></tr>
        <tr><td>Weight stdev</td><td>{stats.weight_stdev:.3f}</td></tr>
      </table>
      <div class="legend">
        <h4>Legend</h4>
        <div><span class="dot" style="background:#4a90d9"></span> Input connections</div>
        <div><span class="dot" style="background:#22c55e"></span> Positive weight</div>
        <div><span class="dot" style="background:#ef4444"></span> Negative weight</div>
        <div><span class="dot" style="background:#d9904a"></span> Output connections</div>
      </div>
      <div class="hint">Scroll to zoom, drag to pan, click node to highlight</div>
    </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NEAT Topology: {color}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: #0a0a1a; color: #ccc; font-family: 'Courier New', monospace; overflow: hidden; }}
#svg-container {{ width: 100vw; height: 100vh; cursor: grab; }}
#svg-container.panning {{ cursor: grabbing; }}
#stats-panel {{
  position: fixed; top: 12px; right: 12px;
  background: rgba(15,15,35,0.92); border: 1px solid #333;
  padding: 16px; border-radius: 8px; font-size: 12px;
  max-width: 260px; z-index: 10;
  backdrop-filter: blur(8px);
}}
#stats-panel h3 {{ color: #aaa; margin-bottom: 10px; font-size: 14px; }}
#stats-panel h4 {{ color: #888; margin: 8px 0 4px; font-size: 11px; }}
#stats-panel table {{ width: 100%; border-collapse: collapse; }}
#stats-panel td {{ padding: 2px 4px; }}
#stats-panel td:first-child {{ color: #888; }}
#stats-panel td:last-child {{ text-align: right; color: #ddd; }}
#stats-panel tr.sep td {{ height: 6px; }}
.legend div {{ margin: 2px 0; display: flex; align-items: center; gap: 6px; color: #999; }}
.dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
.hint {{ margin-top: 10px; color: #555; font-size: 10px; font-style: italic; }}
#tooltip {{
  position: fixed; display: none; background: rgba(10,10,30,0.95);
  border: 1px solid #555; padding: 10px; border-radius: 6px;
  font-size: 12px; pointer-events: none; z-index: 20;
  min-width: 140px;
}}
#tooltip .label {{ color: #888; }}
#tooltip .value {{ color: #eee; float: right; }}
.node {{ cursor: pointer; transition: opacity 0.15s; }}
.node:hover {{ stroke: #fff !important; stroke-width: 2 !important; }}
.conn {{ pointer-events: none; transition: opacity 0.2s; }}
.dimmed {{ opacity: 0.04 !important; }}
</style>
</head>
<body>
<div id="svg-container">
{svg}
</div>
{stats_html}
<div id="tooltip"></div>
<script>
(function() {{
  const svg = document.getElementById('topo-svg');
  const container = document.getElementById('svg-container');
  const tooltip = document.getElementById('tooltip');
  let vb = svg.viewBox.baseVal;
  let isPanning = false, startX, startY, startVBX, startVBY;

  // Zoom
  container.addEventListener('wheel', function(e) {{
    e.preventDefault();
    const scale = e.deltaY > 0 ? 1.1 : 1/1.1;
    const rect = svg.getBoundingClientRect();
    const mx = (e.clientX - rect.left) / rect.width * vb.width + vb.x;
    const my = (e.clientY - rect.top) / rect.height * vb.height + vb.y;
    vb.x = mx - (mx - vb.x) * scale;
    vb.y = my - (my - vb.y) * scale;
    vb.width *= scale;
    vb.height *= scale;
  }}, {{passive: false}});

  // Pan
  container.addEventListener('mousedown', function(e) {{
    if (e.target.classList.contains('node')) return;
    isPanning = true;
    startX = e.clientX; startY = e.clientY;
    startVBX = vb.x; startVBY = vb.y;
    container.classList.add('panning');
  }});
  window.addEventListener('mousemove', function(e) {{
    if (!isPanning) return;
    const rect = svg.getBoundingClientRect();
    const dx = (e.clientX - startX) / rect.width * vb.width;
    const dy = (e.clientY - startY) / rect.height * vb.height;
    vb.x = startVBX - dx;
    vb.y = startVBY - dy;
  }});
  window.addEventListener('mouseup', function() {{
    isPanning = false;
    container.classList.remove('panning');
  }});

  // Tooltips
  document.querySelectorAll('.node').forEach(function(node) {{
    node.addEventListener('mouseenter', function(e) {{
      tooltip.innerHTML =
        '<div><span class="label">Node </span><span class="value">' + node.dataset.id + '</span></div>' +
        '<div><span class="label">Depth </span><span class="value">' + node.dataset.depth + '</span></div>' +
        '<div><span class="label">Bias </span><span class="value">' + parseFloat(node.dataset.bias).toFixed(4) + '</span></div>' +
        '<div><span class="label">Fan-in </span><span class="value">' + node.dataset.fanin + '</span></div>' +
        '<div><span class="label">Fan-out </span><span class="value">' + node.dataset.fanout + '</span></div>';
      tooltip.style.display = 'block';
    }});
    node.addEventListener('mousemove', function(e) {{
      tooltip.style.left = (e.clientX + 15) + 'px';
      tooltip.style.top = (e.clientY + 15) + 'px';
    }});
    node.addEventListener('mouseleave', function() {{
      tooltip.style.display = 'none';
    }});
  }});

  // Click highlight
  let selectedNode = null;
  document.querySelectorAll('.node').forEach(function(node) {{
    node.addEventListener('click', function(e) {{
      e.stopPropagation();
      const nid = node.dataset.id;
      if (selectedNode === nid) {{
        // Deselect
        selectedNode = null;
        document.querySelectorAll('.conn, .node').forEach(el => el.classList.remove('dimmed'));
        return;
      }}
      selectedNode = nid;
      const cx = parseFloat(node.getAttribute('cx'));
      const cy = parseFloat(node.getAttribute('cy'));
      // Dim everything first
      document.querySelectorAll('.conn, .node').forEach(el => el.classList.add('dimmed'));
      node.classList.remove('dimmed');
      // Highlight connected lines
      document.querySelectorAll('.conn.hh').forEach(function(line) {{
        const lx1 = parseFloat(line.getAttribute('x1'));
        const ly1 = parseFloat(line.getAttribute('y1'));
        const lx2 = parseFloat(line.getAttribute('x2'));
        const ly2 = parseFloat(line.getAttribute('y2'));
        const eps = 0.5;
        if ((Math.abs(lx1-cx)<eps && Math.abs(ly1-cy)<eps) ||
            (Math.abs(lx2-cx)<eps && Math.abs(ly2-cy)<eps)) {{
          line.classList.remove('dimmed');
          // Also un-dim the other endpoint node
          document.querySelectorAll('.node').forEach(function(n2) {{
            const n2x = parseFloat(n2.getAttribute('cx'));
            const n2y = parseFloat(n2.getAttribute('cy'));
            if ((Math.abs(lx1-n2x)<eps && Math.abs(ly1-n2y)<eps) ||
                (Math.abs(lx2-n2x)<eps && Math.abs(ly2-n2y)<eps)) {{
              n2.classList.remove('dimmed');
            }}
          }});
        }}
      }});
      // Also un-dim ih/ho aggregate lines for this node
      document.querySelectorAll('.conn.ih, .conn.ho').forEach(function(line) {{
        const lx1 = parseFloat(line.getAttribute('x1'));
        const ly1 = parseFloat(line.getAttribute('y1'));
        const lx2 = parseFloat(line.getAttribute('x2'));
        const ly2 = parseFloat(line.getAttribute('y2'));
        const eps = 0.5;
        if ((Math.abs(lx1-cx)<eps && Math.abs(ly1-cy)<eps) ||
            (Math.abs(lx2-cx)<eps && Math.abs(ly2-cy)<eps)) {{
          line.classList.remove('dimmed');
        }}
      }});
    }});
  }});
  // Click background to deselect
  svg.addEventListener('click', function(e) {{
    if (e.target === svg || e.target.tagName === 'rect') {{
      selectedNode = null;
      document.querySelectorAll('.conn, .node').forEach(el => el.classList.remove('dimmed'));
    }}
  }});
}})();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize NEAT genome topology as interactive HTML",
    )
    parser.add_argument("--genome", default="neat_best_genomes.json",
                        help="Path to genome JSON file (default: neat_best_genomes.json)")
    parser.add_argument("--color", choices=["white", "black"], default="white",
                        help="Which population's best genome to visualize")
    parser.add_argument("--hof", type=int, default=None, metavar="INDEX",
                        help="Visualize HOF genome at this index instead of best")
    parser.add_argument("--output", default="topology.html",
                        help="Output HTML file path (default: topology.html)")
    parser.add_argument("--max-connections", type=int, default=2000,
                        help="Max hidden-to-hidden connections to render individually")
    args = parser.parse_args()

    genome_path = Path(args.genome)
    if not genome_path.exists():
        print(f"Error: genome file not found: {genome_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading genome from {genome_path}...")
    genome = parse_genome_file(str(genome_path), args.color, args.hof)
    print(f"  Parsed: {len(genome.nodes)} nodes, {len(genome.connections)} connections")

    print("Analyzing topology...")
    max_depth = assign_depths(genome)
    stats = compute_stats(genome, max_depth)
    print(f"  {stats.n_hidden} hidden nodes, {stats.n_enabled} enabled connections, depth={stats.max_depth}")

    print("Computing layout...")
    layout = compute_layout(genome, stats, max_depth, args.max_connections)
    print(f"  {len(layout.nodes)} visible nodes, {len(layout.connections)} rendered connections")

    print("Generating HTML...")
    svg = render_svg(layout, stats, args.color)
    html = build_html(svg, stats, args.color)

    out_path = Path(args.output)
    out_path.write_text(html)
    print(f"  Written to {out_path} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
