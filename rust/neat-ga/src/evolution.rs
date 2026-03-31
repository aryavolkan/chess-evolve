use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::config::NeatConfig;
use crate::genome::NeatGenome;
use crate::innovation::InnovationTracker;
use crate::species::{self, Species};

/// Stats returned after each generation.
#[derive(Clone, Serialize, Deserialize)]
pub struct EvolutionStats {
    pub species_count: usize,
    pub best_fitness: f32,
    pub avg_fitness: f32,
    pub avg_connections: f32,
    pub avg_nodes: f32,
    pub best_connections: usize,
    pub best_nodes: usize,
    pub avg_depth: f32,
    pub avg_width: f32,
}

/// Run one generation of NEAT evolution.
///
/// Steps:
/// 1. Apply fitness values to population
/// 2. Speciate
/// 3. Calculate adjusted fitness
/// 4. Track best fitness, update stagnation
/// 5. Cull stagnant species
/// 6. Allocate offspring proportional to adjusted fitness
/// 7. Reproduce: elites + crossover/mutation
/// 8. Balance to exact population size
/// 9. Reset innovation cache
/// 10. Adjust compatibility threshold
pub fn evolve_neat_generation(
    population: &mut Vec<NeatGenome>,
    fitness: &[f32],
    species_list: &mut Vec<Species>,
    config: &mut NeatConfig,
    tracker: &mut InnovationTracker,
    next_species_id: &mut u32,
    rng: &mut impl Rng,
) -> EvolutionStats {
    let pop_size = config.population_size;

    // Apply fitness values
    for (i, genome) in population.iter_mut().enumerate() {
        genome.fitness = if i < fitness.len() { fitness[i] } else { 0.0 };
    }

    // Speciate
    *species_list = species::speciate(population, species_list, config, next_species_id);

    if species_list.is_empty() {
        // Fallback: reinitialize
        let stats = compute_stats(population, species_list);
        return stats;
    }

    // Calculate adjusted fitness and track stagnation
    let mut gen_best_fitness = f32::NEG_INFINITY;
    let mut gen_best_idx: Option<usize> = None;

    for sp in species_list.iter_mut() {
        species::calculate_adjusted_fitness(population, sp);
        species::update_best_fitness(sp, population);

        // Find best genome
        for &idx in &sp.member_indices {
            if population[idx].fitness > gen_best_fitness {
                gen_best_fitness = population[idx].fitness;
                gen_best_idx = Some(idx);
            }
        }
    }

    // Cull stagnant species
    species::cull_stagnant(species_list, config);

    if species_list.is_empty() {
        let stats = compute_stats(population, species_list);
        return stats;
    }

    // Calculate total adjusted fitness after culling
    let mut total_adjusted: f32 = 0.0;
    for sp in species_list.iter() {
        total_adjusted += species::total_adjusted_fitness(population, sp);
    }

    // Allocate offspring and reproduce
    let mut new_population: Vec<NeatGenome> = Vec::with_capacity(pop_size);

    for sp_idx in 0..species_list.len() {
        let sp = &species_list[sp_idx];
        let sp_adjusted = species::total_adjusted_fitness(population, sp);

        let offspring_count = if total_adjusted > 0.0 {
            ((sp_adjusted / total_adjusted) * pop_size as f32).round() as usize
        } else {
            (pop_size as f32 / species_list.len() as f32).ceil() as usize
        };
        let offspring_count = offspring_count.max(1);

        // Sort members by fitness descending
        let mut sorted_members: Vec<usize> = sp.member_indices.clone();
        sorted_members.sort_by(|&a, &b| {
            population[b]
                .fitness
                .partial_cmp(&population[a].fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if sorted_members.is_empty() {
            continue;
        }

        // Elites
        let elite_count = (sorted_members.len() as f32 * config.elite_fraction)
            .ceil()
            .max(1.0) as usize;
        let elites_to_add = elite_count.min(offspring_count);
        for i in 0..elites_to_add {
            new_population.push(population[sorted_members[i]].clone());
        }

        // Survival pool
        let pool_size = (sorted_members.len() as f32 * config.survival_fraction).max(1.0) as usize;
        let pool: Vec<usize> = sorted_members[..pool_size.min(sorted_members.len())].to_vec();

        let remaining = offspring_count.saturating_sub(elites_to_add);
        for _ in 0..remaining {
            let child = if rng.gen::<f32>() < config.crossover_rate && pool.len() >= 2 {
                let pa_idx = pool[rng.gen_range(0..pool.len())];
                let pb_idx = if rng.gen::<f32>() < config.interspecies_crossover_rate
                    && species_list.len() > 1
                {
                    // Interspecies crossover: pick from a different species
                    let other_sp_idx = rng.gen_range(0..species_list.len());
                    let other_sp = &species_list[other_sp_idx];
                    if !other_sp.member_indices.is_empty() {
                        other_sp.member_indices[rng.gen_range(0..other_sp.member_indices.len())]
                    } else {
                        pool[rng.gen_range(0..pool.len())]
                    }
                } else {
                    pool[rng.gen_range(0..pool.len())]
                };
                NeatGenome::crossover(&population[pa_idx], &population[pb_idx], config, rng)
            } else {
                let parent_idx = pool[rng.gen_range(0..pool.len())];
                population[parent_idx].clone()
            };

            let mut child = child;
            child.mutate(config, tracker, rng);
            new_population.push(child);
        }
    }

    // Balance population to exact size
    while new_population.len() > pop_size {
        new_population.pop();
    }
    while new_population.len() < pop_size {
        // Fill with mutated copies of existing genomes
        if let Some(best_idx) = gen_best_idx {
            let mut filler = population[best_idx].clone();
            filler.mutate(config, tracker, rng);
            new_population.push(filler);
        } else if !new_population.is_empty() {
            let src = rng.gen_range(0..new_population.len());
            let mut filler = new_population[src].clone();
            filler.mutate(config, tracker, rng);
            new_population.push(filler);
        } else {
            break;
        }
    }

    let stats = compute_stats(&new_population, species_list);

    // Replace population
    *population = new_population;

    // Reset innovation cache for next generation
    tracker.reset_generation_cache();

    // Adjust compatibility threshold
    species::adjust_threshold(stats.species_count, config);

    stats
}

fn compute_stats(population: &[NeatGenome], species_list: &[Species]) -> EvolutionStats {
    let mut best_fitness = f32::NEG_INFINITY;
    let mut total_fitness: f32 = 0.0;
    let mut total_conns: usize = 0;
    let mut total_nodes: usize = 0;
    let mut best_conns: usize = 0;
    let mut best_nodes: usize = 0;
    let mut best_idx: usize = 0;

    for (i, genome) in population.iter().enumerate() {
        if genome.fitness > best_fitness {
            best_fitness = genome.fitness;
            best_conns = genome.enabled_connection_count();
            best_nodes = genome.nodes.len();
            best_idx = i;
        }
        total_fitness += genome.fitness;
        total_conns += genome.enabled_connection_count();
        total_nodes += genome.nodes.len();
    }

    // Only compute expensive topology_depth_width for the best genome
    // (used for logging only — computing for all N genomes is wasteful).
    let (best_depth, best_width) = if !population.is_empty() {
        population[best_idx].topology_depth_width()
    } else {
        (0, 0)
    };

    let n = population.len().max(1) as f32;

    EvolutionStats {
        species_count: species_list.len(),
        best_fitness,
        avg_fitness: total_fitness / n,
        avg_connections: total_conns as f32 / n,
        avg_nodes: total_nodes as f32 / n,
        best_connections: best_conns,
        best_nodes,
        avg_depth: best_depth as f32,
        avg_width: best_width as f32,
    }
}
