use serde::{Deserialize, Serialize};

use crate::config::NeatConfig;
use crate::genome::NeatGenome;

/// A NEAT species: group of similar genomes.
#[derive(Clone, Serialize, Deserialize)]
pub struct Species {
    pub id: u32,
    pub representative: NeatGenome,
    pub member_indices: Vec<usize>,
    pub best_fitness: f32,
    pub best_fitness_ever: f32,
    pub generations_without_improvement: u32,
    pub age: u32,
}

impl Species {
    pub fn new(id: u32, representative: NeatGenome) -> Self {
        Self {
            id,
            representative,
            member_indices: Vec::new(),
            best_fitness: f32::NEG_INFINITY,
            best_fitness_ever: f32::NEG_INFINITY,
            generations_without_improvement: 0,
            age: 0,
        }
    }

    pub fn is_stagnant(&self, threshold: usize) -> bool {
        self.generations_without_improvement >= threshold as u32
    }
}

/// Assign genomes to species by compatibility distance to representatives.
/// Returns updated species list.
pub fn speciate(
    population: &[NeatGenome],
    existing_species: &[Species],
    config: &NeatConfig,
    next_species_id: &mut u32,
) -> Vec<Species> {
    // Start with existing species, clear members
    let mut species_list: Vec<Species> = existing_species
        .iter()
        .map(|s| {
            let mut s = s.clone();
            s.member_indices.clear();
            s
        })
        .collect();

    // Assign each genome to a species
    for (i, genome) in population.iter().enumerate() {
        let mut placed = false;
        for species in &mut species_list {
            let dist = NeatGenome::compatibility_distance(genome, &species.representative, config);
            if dist < config.compatibility_threshold {
                species.member_indices.push(i);
                placed = true;
                break;
            }
        }
        if !placed {
            let mut new_sp = Species::new(*next_species_id, genome.clone());
            new_sp.member_indices.push(i);
            *next_species_id += 1;
            species_list.push(new_sp);
        }
    }

    // Remove empty species
    species_list.retain(|s| !s.member_indices.is_empty());

    // Update representatives (random member)
    for species in &mut species_list {
        if !species.member_indices.is_empty() {
            // Use first member as representative (deterministic; caller shuffles if needed)
            let rep_idx = species.member_indices[0];
            species.representative = population[rep_idx].clone();
        }
    }

    species_list
}

/// Calculate adjusted fitness for all members of a species (fitness / species_size).
pub fn calculate_adjusted_fitness(population: &mut [NeatGenome], species: &Species) {
    let size = species.member_indices.len() as f32;
    if size == 0.0 {
        return;
    }
    for &idx in &species.member_indices {
        population[idx].adjusted_fitness = population[idx].fitness / size;
    }
}

/// Update best fitness tracking for a species.
pub fn update_best_fitness(species: &mut Species, population: &[NeatGenome]) {
    let mut current_best = f32::NEG_INFINITY;
    for &idx in &species.member_indices {
        if population[idx].fitness > current_best {
            current_best = population[idx].fitness;
        }
    }

    if current_best > species.best_fitness_ever {
        species.best_fitness_ever = current_best;
        species.generations_without_improvement = 0;
    } else {
        species.generations_without_improvement += 1;
    }

    species.best_fitness = current_best;
    species.age += 1;
}

/// Get total adjusted fitness for a species.
pub fn total_adjusted_fitness(population: &[NeatGenome], species: &Species) -> f32 {
    species
        .member_indices
        .iter()
        .map(|&idx| population[idx].adjusted_fitness)
        .sum()
}

/// Adjust compatibility threshold to converge on target species count.
///
/// Uses proportional control with clamping to prevent the wild oscillation
/// (2-190 species) caused by the old fixed-step approach.
pub fn adjust_threshold(species_count: usize, config: &mut NeatConfig) {
    let target = config.target_species_count as f32;
    let actual = species_count as f32;
    if target > 0.0 && (actual - target).abs() > 0.5 {
        let error = (actual - target) / target;
        let adjustment = error * config.threshold_step;
        let clamped = adjustment.clamp(-0.5, 0.5);
        config.compatibility_threshold += clamped;
    }
    config.compatibility_threshold = config.compatibility_threshold.max(0.3);
}

/// Cull stagnant species, protecting the best min_species_protected.
pub fn cull_stagnant(species_list: &mut Vec<Species>, config: &NeatConfig) {
    if species_list.len() <= config.min_species_protected {
        return;
    }

    // Sort by best_fitness_ever descending
    species_list.sort_by(|a, b| {
        b.best_fitness_ever
            .partial_cmp(&a.best_fitness_ever)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut surviving = Vec::new();
    for (i, species) in species_list.iter().enumerate() {
        if i < config.min_species_protected
            || !species.is_stagnant(config.stagnation_kill_threshold)
        {
            surviving.push(species.clone());
        }
    }

    *species_list = surviving;
}
