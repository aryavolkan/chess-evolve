use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Global innovation tracker for NEAT. Mirrors ai/neat_innovation.gd.
///
/// The cache uses string keys ("in_id:out_id") for JSON serialization compatibility.
#[derive(Clone, Serialize, Deserialize)]
pub struct InnovationTracker {
    pub next_innovation: u32,
    pub next_node_id: u32,
    /// Per-generation cache: "in_id:out_id" -> innovation number.
    /// Cleared each generation so the same structural mutation in different
    /// genomes within a generation gets the same innovation number.
    cache: HashMap<String, u32>,
}

impl InnovationTracker {
    pub fn new(initial_node_id: u32) -> Self {
        Self {
            next_innovation: 0,
            next_node_id: initial_node_id,
            cache: HashMap::new(),
        }
    }

    fn cache_key(in_id: u32, out_id: u32) -> String {
        format!("{}:{}", in_id, out_id)
    }

    /// Get or allocate an innovation number for a connection (in_id -> out_id).
    pub fn get_innovation(&mut self, in_id: u32, out_id: u32) -> u32 {
        let key = Self::cache_key(in_id, out_id);
        if let Some(&innov) = self.cache.get(&key) {
            return innov;
        }
        let innov = self.next_innovation;
        self.cache.insert(key, innov);
        self.next_innovation += 1;
        innov
    }

    /// Allocate a new node ID (monotonic counter).
    pub fn allocate_node_id(&mut self) -> u32 {
        let id = self.next_node_id;
        self.next_node_id += 1;
        id
    }

    /// Clear per-generation cache. Counters are preserved.
    pub fn reset_generation_cache(&mut self) {
        self.cache.clear();
    }

    /// Seed tracker state from an existing genome so new mutations get
    /// non-colliding innovation numbers and node IDs.
    pub fn seed_from_genome(&mut self, genome: &super::genome::NeatGenome) {
        for node in &genome.nodes {
            if node.id >= self.next_node_id {
                self.next_node_id = node.id + 1;
            }
        }
        for conn in &genome.connections {
            if conn.innovation >= self.next_innovation {
                self.next_innovation = conn.innovation + 1;
            }
            let key = Self::cache_key(conn.in_id, conn.out_id);
            self.cache.insert(key, conn.innovation);
        }
    }
}
