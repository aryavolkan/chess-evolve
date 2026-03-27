use std::collections::{HashMap, HashSet};

use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::config::NeatConfig;
use crate::innovation::InnovationTracker;

#[derive(Clone, Serialize, Deserialize)]
pub struct NodeGene {
    pub id: u32,
    pub node_type: u8, // 0=input, 1=hidden, 2=output
    pub bias: f32,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ConnectionGene {
    pub in_id: u32,
    pub out_id: u32,
    pub weight: f32,
    pub enabled: bool,
    pub innovation: u32,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct NeatGenome {
    pub nodes: Vec<NodeGene>,
    pub connections: Vec<ConnectionGene>,
    #[serde(default)]
    pub fitness: f32,
    #[serde(default)]
    pub adjusted_fitness: f32,
}

impl NeatGenome {
    /// Create a minimal genome with input + output nodes, no connections.
    pub fn create_minimal(input_count: usize, output_count: usize) -> Self {
        let mut nodes = Vec::with_capacity(input_count + output_count);
        for i in 0..input_count {
            nodes.push(NodeGene {
                id: i as u32,
                node_type: 0,
                bias: 0.0,
            });
        }
        for i in 0..output_count {
            nodes.push(NodeGene {
                id: (input_count + i) as u32,
                node_type: 2,
                bias: 0.0,
            });
        }
        Self {
            nodes,
            connections: Vec::new(),
            fitness: 0.0,
            adjusted_fitness: 0.0,
        }
    }

    /// Create a genome with sparse random initial connections.
    pub fn create_basic(
        input_count: usize,
        output_count: usize,
        initial_conns_per_output: usize,
        tracker: &mut InnovationTracker,
        rng: &mut impl Rng,
    ) -> Self {
        let mut genome = Self::create_minimal(input_count, output_count);

        // Randomize output biases
        for node in &mut genome.nodes {
            if node.node_type == 2 {
                node.bias = rng.gen_range(-1.0..1.0);
            }
        }

        let input_ids: Vec<u32> = genome
            .nodes
            .iter()
            .filter(|n| n.node_type == 0)
            .map(|n| n.id)
            .collect();
        let output_ids: Vec<u32> = genome
            .nodes
            .iter()
            .filter(|n| n.node_type == 2)
            .map(|n| n.id)
            .collect();

        if input_ids.is_empty() || output_ids.is_empty() {
            return genome;
        }

        let k = initial_conns_per_output.max(1).min(input_ids.len());
        for &out_id in &output_ids {
            // Reservoir sampling to pick k random inputs
            let mut chosen: Vec<u32> = input_ids[..k].to_vec();
            for (i, &inp) in input_ids.iter().enumerate().skip(k) {
                let j = rng.gen_range(0..=i);
                if j < k {
                    chosen[j] = inp;
                }
            }
            for &in_id in &chosen {
                let innov = tracker.get_innovation(in_id, out_id);
                genome.connections.push(ConnectionGene {
                    in_id,
                    out_id,
                    weight: rng.gen_range(-2.0..2.0),
                    enabled: true,
                    innovation: innov,
                });
            }
        }

        genome
    }

    /// Clone topology from a template, randomize weights and biases.
    pub fn create_from_topology(template: &NeatGenome, rng: &mut impl Rng) -> Self {
        let nodes = template
            .nodes
            .iter()
            .map(|n| NodeGene {
                id: n.id,
                node_type: n.node_type,
                bias: rng.gen_range(-1.0..1.0),
            })
            .collect();
        let connections = template
            .connections
            .iter()
            .map(|c| ConnectionGene {
                in_id: c.in_id,
                out_id: c.out_id,
                weight: rng.gen_range(-2.0..2.0),
                enabled: c.enabled,
                innovation: c.innovation,
            })
            .collect();
        Self {
            nodes,
            connections,
            fitness: 0.0,
            adjusted_fitness: 0.0,
        }
    }

    /// Apply all mutation operators. Mirrors neat_genome.gd:mutate().
    pub fn mutate(
        &mut self,
        config: &NeatConfig,
        tracker: &mut InnovationTracker,
        rng: &mut impl Rng,
    ) {
        if rng.gen::<f32>() < config.weight_mutate_rate {
            self.mutate_weights(config, rng);
        }
        if rng.gen::<f32>() < config.add_connection_rate {
            for _ in 0..config.add_connection_count {
                self.mutate_add_connection(tracker, rng);
            }
        }
        if rng.gen::<f32>() < config.add_node_rate {
            for _ in 0..config.add_node_count {
                self.mutate_add_node(tracker, rng);
            }
        }
        if rng.gen::<f32>() < config.disable_connection_rate {
            self.mutate_disable_connection(rng);
        }
        if rng.gen::<f32>() < config.prune_rate {
            self.prune();
        }
    }

    /// Perturb or reset connection weights.
    pub fn mutate_weights(&mut self, config: &NeatConfig, rng: &mut impl Rng) {
        let normal = Normal::new(0.0f32, config.weight_perturb_strength).unwrap();
        for conn in &mut self.connections {
            if rng.gen::<f32>() < config.weight_mutate_rate {
                if rng.gen::<f32>() < config.weight_perturb_rate {
                    conn.weight += normal.sample(rng);
                } else {
                    conn.weight =
                        rng.gen_range(-config.weight_reset_range..config.weight_reset_range);
                }
            }
        }
    }

    /// Add a random connection between valid nodes. 10 retries.
    pub fn mutate_add_connection(&mut self, tracker: &mut InnovationTracker, rng: &mut impl Rng) {
        let possible_inputs: Vec<u32> = self
            .nodes
            .iter()
            .filter(|n| n.node_type != 2) // inputs + hidden
            .map(|n| n.id)
            .collect();
        let possible_outputs: Vec<u32> = self
            .nodes
            .iter()
            .filter(|n| n.node_type != 0) // hidden + outputs
            .map(|n| n.id)
            .collect();

        if possible_inputs.is_empty() || possible_outputs.is_empty() {
            return;
        }

        for _ in 0..10 {
            let in_id = possible_inputs[rng.gen_range(0..possible_inputs.len())];
            let out_id = possible_outputs[rng.gen_range(0..possible_outputs.len())];

            if in_id == out_id {
                continue;
            }
            if self
                .connections
                .iter()
                .any(|c| c.in_id == in_id && c.out_id == out_id)
            {
                continue;
            }
            if would_create_cycle(in_id, out_id, &self.connections) {
                continue;
            }

            let innov = tracker.get_innovation(in_id, out_id);
            self.connections.push(ConnectionGene {
                in_id,
                out_id,
                weight: rng.gen_range(-2.0..2.0),
                enabled: true,
                innovation: innov,
            });
            return;
        }
    }

    /// Split a random enabled connection: disable old, add node + 2 new connections.
    pub fn mutate_add_node(&mut self, tracker: &mut InnovationTracker, rng: &mut impl Rng) {
        let enabled_indices: Vec<usize> = self
            .connections
            .iter()
            .enumerate()
            .filter(|(_, c)| c.enabled)
            .map(|(i, _)| i)
            .collect();

        if enabled_indices.is_empty() {
            return;
        }

        let idx = enabled_indices[rng.gen_range(0..enabled_indices.len())];
        let old_in = self.connections[idx].in_id;
        let old_out = self.connections[idx].out_id;
        let old_weight = self.connections[idx].weight;
        self.connections[idx].enabled = false;

        let new_node_id = tracker.allocate_node_id();
        self.nodes.push(NodeGene {
            id: new_node_id,
            node_type: 1,
            bias: 0.0,
        });

        let innov1 = tracker.get_innovation(old_in, new_node_id);
        self.connections.push(ConnectionGene {
            in_id: old_in,
            out_id: new_node_id,
            weight: 1.0,
            enabled: true,
            innovation: innov1,
        });

        let innov2 = tracker.get_innovation(new_node_id, old_out);
        self.connections.push(ConnectionGene {
            in_id: new_node_id,
            out_id: old_out,
            weight: old_weight,
            enabled: true,
            innovation: innov2,
        });
    }

    /// Disable a random enabled connection.
    pub fn mutate_disable_connection(&mut self, rng: &mut impl Rng) {
        let enabled_indices: Vec<usize> = self
            .connections
            .iter()
            .enumerate()
            .filter(|(_, c)| c.enabled)
            .map(|(i, _)| i)
            .collect();

        if !enabled_indices.is_empty() {
            let idx = enabled_indices[rng.gen_range(0..enabled_indices.len())];
            self.connections[idx].enabled = false;
        }
    }

    /// NEAT crossover: align by innovation number. Better parent's excess/disjoint always kept.
    pub fn crossover(
        parent_a: &NeatGenome,
        parent_b: &NeatGenome,
        config: &NeatConfig,
        rng: &mut impl Rng,
    ) -> NeatGenome {
        // a = better fitness parent
        let (a, b) = if parent_a.fitness >= parent_b.fitness {
            (parent_a, parent_b)
        } else {
            (parent_b, parent_a)
        };
        let equal_fitness = (a.fitness - b.fitness).abs() < 0.001;

        let map_a: HashMap<u32, &ConnectionGene> =
            a.connections.iter().map(|c| (c.innovation, c)).collect();
        let map_b: HashMap<u32, &ConnectionGene> =
            b.connections.iter().map(|c| (c.innovation, c)).collect();

        let mut all_innovs: HashSet<u32> = HashSet::new();
        for c in &a.connections {
            all_innovs.insert(c.innovation);
        }
        for c in &b.connections {
            all_innovs.insert(c.innovation);
        }

        let mut child_conns = Vec::new();
        for &innov in &all_innovs {
            let in_a = map_a.get(&innov);
            let in_b = map_b.get(&innov);

            match (in_a, in_b) {
                (Some(ga), Some(gb)) => {
                    // Matching gene: pick randomly from either parent
                    let source = if rng.gen::<f32>() < 0.5 { *ga } else { *gb };
                    let mut gene = source.clone();
                    // If disabled in either parent, chance of inheriting disabled
                    if !ga.enabled || !gb.enabled {
                        gene.enabled = rng.gen::<f32>() >= config.disabled_gene_inherit_rate;
                    }
                    child_conns.push(gene);
                }
                (Some(ga), None) => {
                    // Excess/disjoint from better parent: always keep
                    child_conns.push((*ga).clone());
                }
                (None, Some(gb)) => {
                    // Excess/disjoint from worse parent: only keep if equal fitness
                    if equal_fitness {
                        child_conns.push((*gb).clone());
                    }
                }
                (None, None) => unreachable!(),
            }
        }

        // Collect needed node IDs
        let mut needed_ids: HashSet<u32> = HashSet::new();
        for node in &a.nodes {
            if node.node_type == 0 || node.node_type == 2 {
                needed_ids.insert(node.id);
            }
        }
        for conn in &child_conns {
            needed_ids.insert(conn.in_id);
            needed_ids.insert(conn.out_id);
        }

        let node_map_a: HashMap<u32, &NodeGene> = a.nodes.iter().map(|n| (n.id, n)).collect();
        let node_map_b: HashMap<u32, &NodeGene> = b.nodes.iter().map(|n| (n.id, n)).collect();

        let mut child_nodes: Vec<NodeGene> = Vec::new();
        for &nid in &needed_ids {
            if let Some(n) = node_map_a.get(&nid) {
                child_nodes.push((*n).clone());
            } else if let Some(n) = node_map_b.get(&nid) {
                child_nodes.push((*n).clone());
            }
        }
        // Sort: inputs first, then hidden, then output; within type by ID
        child_nodes.sort_by(|x, y| x.node_type.cmp(&y.node_type).then(x.id.cmp(&y.id)));

        NeatGenome {
            nodes: child_nodes,
            connections: child_conns,
            fitness: 0.0,
            adjusted_fitness: 0.0,
        }
    }

    /// Compatibility distance between two genomes. Mirrors neat_genome.gd:compatibility().
    ///
    /// Uses sorted merge-join on innovation numbers instead of building two
    /// HashMaps per call — O(n log n) sort + O(n) scan vs O(n) HashMap build
    /// with high constant factor from hashing and heap allocation.
    pub fn compatibility_distance(a: &NeatGenome, b: &NeatGenome, config: &NeatConfig) -> f32 {
        if a.connections.is_empty() && b.connections.is_empty() {
            return 0.0;
        }

        // Build sorted (innovation, weight) arrays
        let mut sorted_a: Vec<(u32, f32)> = a
            .connections
            .iter()
            .map(|g| (g.innovation, g.weight))
            .collect();
        let mut sorted_b: Vec<(u32, f32)> = b
            .connections
            .iter()
            .map(|g| (g.innovation, g.weight))
            .collect();
        sorted_a.sort_unstable_by_key(|&(innov, _)| innov);
        sorted_b.sort_unstable_by_key(|&(innov, _)| innov);

        let max_innov_a = sorted_a.last().map_or(0, |&(i, _)| i);
        let max_innov_b = sorted_b.last().map_or(0, |&(i, _)| i);
        let smaller_max = max_innov_a.min(max_innov_b);

        let mut excess: u32 = 0;
        let mut disjoint: u32 = 0;
        let mut weight_diff_sum: f32 = 0.0;
        let mut matching_count: u32 = 0;

        // Merge-join scan
        let mut ia = 0;
        let mut ib = 0;
        while ia < sorted_a.len() && ib < sorted_b.len() {
            let (innov_a, wa) = sorted_a[ia];
            let (innov_b, wb) = sorted_b[ib];
            if innov_a == innov_b {
                weight_diff_sum += (wa - wb).abs();
                matching_count += 1;
                ia += 1;
                ib += 1;
            } else if innov_a < innov_b {
                if innov_a > smaller_max {
                    excess += 1;
                } else {
                    disjoint += 1;
                }
                ia += 1;
            } else {
                if innov_b > smaller_max {
                    excess += 1;
                } else {
                    disjoint += 1;
                }
                ib += 1;
            }
        }
        // Remaining genes
        for &(innov, _) in &sorted_a[ia..] {
            if innov > smaller_max {
                excess += 1;
            } else {
                disjoint += 1;
            }
        }
        for &(innov, _) in &sorted_b[ib..] {
            if innov > smaller_max {
                excess += 1;
            } else {
                disjoint += 1;
            }
        }

        let n = (a.connections.len().max(b.connections.len())) as f32;
        let n = if n < 20.0 { 1.0 } else { n };

        let avg_weight_diff = if matching_count > 0 {
            weight_diff_sum / matching_count as f32
        } else {
            0.0
        };

        (config.c1_excess * excess as f32 / n)
            + (config.c2_disjoint * disjoint as f32 / n)
            + (config.c3_weight_diff * avg_weight_diff)
    }

    /// Remove disabled connections and orphaned hidden nodes.
    /// Orphans = hidden nodes not on any enabled path from an input to an output.
    pub fn prune(&mut self) {
        // Drop disabled connections
        self.connections.retain(|c| c.enabled);

        // Build forward and reverse adjacency from enabled connections
        let mut fwd: HashMap<u32, Vec<u32>> = HashMap::new();
        let mut rev: HashMap<u32, Vec<u32>> = HashMap::new();
        for c in &self.connections {
            fwd.entry(c.in_id).or_default().push(c.out_id);
            rev.entry(c.out_id).or_default().push(c.in_id);
        }

        let input_ids: HashSet<u32> = self
            .nodes
            .iter()
            .filter(|n| n.node_type == 0)
            .map(|n| n.id)
            .collect();
        let output_ids: HashSet<u32> = self
            .nodes
            .iter()
            .filter(|n| n.node_type == 2)
            .map(|n| n.id)
            .collect();

        // Forward reachable from inputs
        let mut fwd_reach = input_ids.clone();
        let mut stack: Vec<u32> = input_ids.iter().copied().collect();
        while let Some(n) = stack.pop() {
            if let Some(neighbors) = fwd.get(&n) {
                for &nxt in neighbors {
                    if fwd_reach.insert(nxt) {
                        stack.push(nxt);
                    }
                }
            }
        }

        // Backward reachable from outputs
        let mut bwd_reach = output_ids.clone();
        let mut stack: Vec<u32> = output_ids.iter().copied().collect();
        while let Some(n) = stack.pop() {
            if let Some(neighbors) = rev.get(&n) {
                for &prev in neighbors {
                    if bwd_reach.insert(prev) {
                        stack.push(prev);
                    }
                }
            }
        }

        // Keep inputs, outputs, and hidden nodes on an active path
        self.nodes.retain(|n| {
            n.node_type == 0
                || n.node_type == 2
                || (fwd_reach.contains(&n.id) && bwd_reach.contains(&n.id))
        });

        // Also drop connections referencing removed nodes
        let live_ids: HashSet<u32> = self.nodes.iter().map(|n| n.id).collect();
        self.connections
            .retain(|c| live_ids.contains(&c.in_id) && live_ids.contains(&c.out_id));
    }

    pub fn enabled_connection_count(&self) -> usize {
        self.connections.iter().filter(|c| c.enabled).count()
    }

    pub fn hidden_node_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.node_type == 1).count()
    }

    /// Compute network depth (longest input→output path) and width (max nodes at any depth level).
    ///
    /// Depth is measured via topological BFS through enabled connections.
    /// Input nodes have depth 0. Each downstream node gets max(predecessor depths) + 1.
    /// Returns (max_depth_of_output_nodes, max_nodes_at_any_depth_level).
    pub fn topology_depth_width(&self) -> (usize, usize) {
        let node_ids: HashSet<u32> = self.nodes.iter().map(|n| n.id).collect();

        // Build adjacency (forward) and in-degree from enabled connections
        let mut adj: HashMap<u32, Vec<u32>> = HashMap::new();
        let mut in_degree: HashMap<u32, usize> = HashMap::new();
        for &id in &node_ids {
            in_degree.insert(id, 0);
        }
        for conn in &self.connections {
            if !conn.enabled {
                continue;
            }
            if !node_ids.contains(&conn.in_id) || !node_ids.contains(&conn.out_id) {
                continue;
            }
            adj.entry(conn.in_id).or_default().push(conn.out_id);
            *in_degree.entry(conn.out_id).or_insert(0) += 1;
        }

        // Kahn's BFS — assign depth per node
        let mut depth: HashMap<u32, usize> = HashMap::new();
        let mut queue: Vec<u32> = Vec::new();
        for (&id, &deg) in &in_degree {
            if deg == 0 {
                depth.insert(id, 0);
                queue.push(id);
            }
        }

        let mut head = 0;
        while head < queue.len() {
            let cur = queue[head];
            head += 1;
            let cur_depth = depth[&cur];
            if let Some(neighbors) = adj.get(&cur) {
                for &nxt in neighbors {
                    // Update depth to max of all predecessors + 1
                    let new_depth = cur_depth + 1;
                    let entry = depth.entry(nxt).or_insert(0);
                    if new_depth > *entry {
                        *entry = new_depth;
                    }
                    let deg = in_degree.get_mut(&nxt).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push(nxt);
                    }
                }
            }
        }

        // Depth = max depth among output nodes (or all nodes if no outputs reached)
        let max_depth = self
            .nodes
            .iter()
            .filter(|n| n.node_type == 2)
            .filter_map(|n| depth.get(&n.id))
            .copied()
            .max()
            .unwrap_or(0);

        // Width = max count of nodes sharing the same depth
        let mut level_counts: HashMap<usize, usize> = HashMap::new();
        for &d in depth.values() {
            *level_counts.entry(d).or_insert(0) += 1;
        }
        let max_width = level_counts.values().copied().max().unwrap_or(0);

        (max_depth, max_width)
    }
}

/// DFS cycle check: would adding an edge from `from_id` to `to_id` create a cycle?
fn would_create_cycle(from_id: u32, to_id: u32, connections: &[ConnectionGene]) -> bool {
    // Check if to_id can reach from_id through enabled connections
    let mut visited = HashSet::new();
    let mut stack = vec![to_id];
    while let Some(current) = stack.pop() {
        if current == from_id {
            return true;
        }
        if !visited.insert(current) {
            continue;
        }
        for conn in connections {
            if conn.enabled && conn.in_id == current {
                stack.push(conn.out_id);
            }
        }
    }
    false
}
