use serde::Deserialize;

/// Minimal genome structs for deserialization (subset of neat-ga's types).
#[derive(Deserialize)]
struct NodeGene {
    id: u32,
    node_type: u8,
    bias: f32,
}

#[derive(Deserialize)]
struct ConnectionGene {
    in_id: u32,
    out_id: u32,
    weight: f32,
    enabled: bool,
}

#[derive(Deserialize)]
struct GenomeData {
    nodes: Vec<NodeGene>,
    connections: Vec<ConnectionGene>,
}

/// Compiled sparse network in topological order for fast forward pass.
///
/// Uses CSR-style edge storage for cache locality.
pub struct SparseNetwork {
    pub node_count: usize,
    pub input_ids: Vec<u32>,
    pub output_ids: Vec<u32>,
    /// Topological evaluation order (non-input nodes only).
    pub eval_order: Vec<u32>,
    /// Bias per node position.
    pub biases: Vec<f32>,
    /// CSR edge storage: edges into node at position p are
    /// edge_targets[edge_offsets[p]..edge_offsets[p+1]] with
    /// corresponding weights in edge_weights.
    pub edge_sources: Vec<usize>, // source node position for each edge
    pub edge_weights: Vec<f32>,
    pub edge_offsets: Vec<usize>, // edge_offsets[p]..edge_offsets[p+1]
    /// Node ID -> position mapping. Indexed by node ID.
    pub id_to_pos: Vec<usize>,
    /// Maximum node ID + 1 (size of id_to_pos).
    pub max_id: usize,
}

impl SparseNetwork {
    /// Build a sparse network from genome JSON.
    pub fn from_genome_json(json: &str) -> Option<Self> {
        let data: GenomeData = serde_json::from_str(json).ok()?;
        Self::from_genome_data(&data)
    }

    fn from_genome_data(data: &GenomeData) -> Option<Self> {
        if data.nodes.is_empty() {
            return None;
        }

        let max_id = data.nodes.iter().map(|n| n.id).max().unwrap_or(0) as usize + 1;
        let node_count = data.nodes.len();

        // Assign positions: order by type (input=0, hidden=1, output=2), then ID
        let mut sorted_nodes: Vec<&NodeGene> = data.nodes.iter().collect();
        sorted_nodes.sort_by(|a, b| a.node_type.cmp(&b.node_type).then(a.id.cmp(&b.id)));

        let mut id_to_pos = vec![usize::MAX; max_id];
        let mut biases = vec![0.0f32; node_count];
        let mut input_ids = Vec::new();
        let mut output_ids = Vec::new();

        for (pos, node) in sorted_nodes.iter().enumerate() {
            id_to_pos[node.id as usize] = pos;
            biases[pos] = node.bias;
            match node.node_type {
                0 => input_ids.push(node.id),
                2 => output_ids.push(node.id),
                _ => {}
            }
        }

        // Build adjacency for topological sort (only enabled connections)
        let mut in_degree = vec![0u32; node_count];
        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); node_count];
        // Also collect incoming edges per node position
        let mut incoming_edges: Vec<Vec<(usize, f32)>> = vec![Vec::new(); node_count];

        for conn in &data.connections {
            if !conn.enabled {
                continue;
            }
            let src_pos = id_to_pos
                .get(conn.in_id as usize)
                .copied()
                .unwrap_or(usize::MAX);
            let dst_pos = id_to_pos
                .get(conn.out_id as usize)
                .copied()
                .unwrap_or(usize::MAX);
            if src_pos == usize::MAX || dst_pos == usize::MAX {
                continue;
            }
            in_degree[dst_pos] += 1;
            adjacency[src_pos].push(dst_pos);
            incoming_edges[dst_pos].push((src_pos, conn.weight));
        }

        // Kahn's algorithm for topological sort
        let mut queue: Vec<usize> = Vec::new();
        for pos in 0..node_count {
            if in_degree[pos] == 0 {
                queue.push(pos);
            }
        }

        let mut topo_order: Vec<usize> = Vec::with_capacity(node_count);
        let mut head = 0;
        while head < queue.len() {
            let pos = queue[head];
            head += 1;
            topo_order.push(pos);
            for &downstream in &adjacency[pos] {
                in_degree[downstream] -= 1;
                if in_degree[downstream] == 0 {
                    queue.push(downstream);
                }
            }
        }

        // Fallback if cycle detected: use type-based ordering
        if topo_order.len() < node_count {
            topo_order.clear();
            for pos in 0..node_count {
                topo_order.push(pos);
            }
        }

        // eval_order = non-input nodes in topological order
        let input_set: std::collections::HashSet<usize> = sorted_nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.node_type == 0)
            .map(|(pos, _)| pos)
            .collect();

        let eval_order: Vec<u32> = topo_order
            .iter()
            .filter(|pos| !input_set.contains(pos))
            .map(|&pos| pos as u32)
            .collect();

        // Build CSR edge storage
        let mut edge_sources = Vec::new();
        let mut edge_weights = Vec::new();
        let mut edge_offsets = vec![0usize; node_count + 1];

        for pos in 0..node_count {
            edge_offsets[pos] = edge_sources.len();
            for &(src_pos, weight) in &incoming_edges[pos] {
                edge_sources.push(src_pos);
                edge_weights.push(weight);
            }
        }
        edge_offsets[node_count] = edge_sources.len();

        Some(Self {
            node_count,
            input_ids,
            output_ids,
            eval_order,
            biases,
            edge_sources,
            edge_weights,
            edge_offsets,
            id_to_pos,
            max_id,
        })
    }

    /// Forward pass with tanh activation. Writes outputs into `outputs`.
    pub fn forward_into(&self, inputs: &[f32], activations: &mut [f32], outputs: &mut [f32]) {
        // Zero activations
        for a in activations.iter_mut() {
            *a = 0.0;
        }

        // Set input activations
        for (i, &input_id) in self.input_ids.iter().enumerate() {
            if let Some(&pos) = self.id_to_pos.get(input_id as usize) {
                if pos < activations.len() {
                    activations[pos] = if i < inputs.len() { inputs[i] } else { 0.0 };
                }
            }
        }

        // Evaluate non-input nodes in topological order.
        // Uses unchecked indexing on edge_offsets/biases (guarded by pos check)
        // but safe indexing on edge_sources/edge_weights to handle malformed genomes
        // that violate the CSR invariant.
        let n_edges = self.edge_sources.len();
        for &pos_u32 in &self.eval_order {
            let pos = pos_u32 as usize;
            if pos + 1 >= self.edge_offsets.len() || pos >= activations.len() {
                continue;
            }
            // SAFETY: pos+1 < edge_offsets.len() checked above
            let start = unsafe { *self.edge_offsets.get_unchecked(pos) };
            let end = unsafe { *self.edge_offsets.get_unchecked(pos + 1) };
            let mut sum = unsafe { *self.biases.get_unchecked(pos) };
            // Clamp end to actual edge array length to guard against malformed CSR
            let safe_end = end.min(n_edges);
            for e in start..safe_end {
                let src = self.edge_sources[e];
                let w = self.edge_weights[e];
                let a = if src < activations.len() {
                    activations[src]
                } else {
                    0.0
                };
                sum += a * w;
            }
            activations[pos] = sum.tanh();
        }

        // Read output activations
        for (i, &out_id) in self.output_ids.iter().enumerate() {
            if let Some(&pos) = self.id_to_pos.get(out_id as usize) {
                if i < outputs.len() && pos < activations.len() {
                    outputs[i] = activations[pos];
                }
            }
        }
    }
}
