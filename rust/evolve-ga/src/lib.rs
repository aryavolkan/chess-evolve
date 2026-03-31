//! evolve-ga: High-performance genetic algorithm operators for chess neuroevolution.
//!
//! This crate provides Rust implementations of selection, crossover, mutation,
//! speciation, and island-model operations that can be called from Python via PyO3.
//! All operators work on flat f32 weight vectors representing neural network genomes.

use pyo3::prelude::*;

mod crossover;
mod island;
mod mutation;
mod selection;
mod speciation;

/// The Python module exposed by this crate.
#[pymodule]
fn evolve_ga(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Selection strategies
    m.add_function(wrap_pyfunction!(selection::tournament_select, m)?)?;
    m.add_function(wrap_pyfunction!(selection::tournament_select_batch, m)?)?;
    m.add_function(wrap_pyfunction!(selection::sus_select, m)?)?;
    m.add_function(wrap_pyfunction!(selection::rank_select, m)?)?;

    // Crossover operators
    m.add_function(wrap_pyfunction!(crossover::two_point_crossover, m)?)?;
    m.add_function(wrap_pyfunction!(crossover::sbx_crossover, m)?)?;
    m.add_function(wrap_pyfunction!(crossover::blx_alpha_crossover, m)?)?;

    // Mutation operators
    m.add_function(wrap_pyfunction!(mutation::gaussian_mutate, m)?)?;
    m.add_function(wrap_pyfunction!(mutation::adaptive_gaussian_mutate, m)?)?;
    m.add_function(wrap_pyfunction!(mutation::one_fifth_rule_adapt, m)?)?;
    m.add_function(wrap_pyfunction!(mutation::cauchy_mutate, m)?)?;

    // Speciation / fitness sharing
    m.add_function(wrap_pyfunction!(speciation::assign_species, m)?)?;
    m.add_function(wrap_pyfunction!(speciation::assign_species_flat, m)?)?;
    m.add_function(wrap_pyfunction!(speciation::shared_fitness, m)?)?;

    // Island model
    m.add_function(wrap_pyfunction!(island::migrate, m)?)?;

    // Batch population operations
    m.add_function(wrap_pyfunction!(evolve_generation, m)?)?;
    m.add_function(wrap_pyfunction!(evolve_generation_flat, m)?)?;

    Ok(())
}

/// Run a full generational step: select parents, crossover, mutate, and return
/// the new population.
///
/// This is the main entry point for running a complete generational cycle from
/// Python. It combines elitism, tournament selection, configurable crossover,
/// and Gaussian mutation into a single optimized call that avoids Python-Rust
/// boundary overhead for each individual operation.
///
/// Arguments:
///     population: list of flat weight vectors (list[list[float]])
///     fitness: fitness values aligned with population (list[float])
///     elite_count: number of top individuals to preserve unchanged
///     mutation_rate: probability of mutating each weight
///     mutation_strength: std-dev of Gaussian noise added on mutation
///     crossover_rate: probability of applying crossover vs cloning
///     tournament_k: tournament size for selection
///     crossover_mode: "two_point", "sbx", or "blx_alpha"
///     sbx_eta: distribution index for SBX crossover (default 2.0)
///     blx_alpha: extension factor for BLX-alpha crossover (default 0.5)
///
/// Returns:
///     New population as list[list[float]]
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    population,
    fitness,
    elite_count = 3,
    mutation_rate = 0.15,
    mutation_strength = 0.2,
    crossover_rate = 0.7,
    tournament_k = 3,
    crossover_mode = "two_point",
    sbx_eta = 2.0,
    blx_alpha = 0.5,
))]
fn evolve_generation(
    population: Vec<Vec<f32>>,
    fitness: Vec<f32>,
    elite_count: usize,
    mutation_rate: f64,
    mutation_strength: f64,
    crossover_rate: f64,
    tournament_k: usize,
    crossover_mode: &str,
    sbx_eta: f64,
    blx_alpha: f64,
) -> PyResult<Vec<Vec<f32>>> {
    use rand::Rng;

    let pop_size = population.len();
    if pop_size == 0 || fitness.len() != pop_size {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "population and fitness must be non-empty and same length",
        ));
    }

    // Sort indices by fitness descending
    let mut indices: Vec<usize> = (0..pop_size).collect();
    indices.sort_by(|&a, &b| {
        fitness[b]
            .partial_cmp(&fitness[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut new_pop: Vec<Vec<f32>> = Vec::with_capacity(pop_size);

    // Elitism: copy top individuals unchanged
    let actual_elite = elite_count.min(pop_size);
    for &idx in indices.iter().take(actual_elite) {
        new_pop.push(population[idx].clone());
    }

    // Fill remaining slots
    let mut rng = rand::thread_rng();
    while new_pop.len() < pop_size {
        let parent_a_idx = selection::tournament_select_impl(&fitness, tournament_k, &mut rng);

        let child = if rng.gen::<f64>() < crossover_rate {
            let parent_b_idx = selection::tournament_select_impl(&fitness, tournament_k, &mut rng);
            match crossover_mode {
                "sbx" => crossover::sbx_crossover_impl(
                    &population[parent_a_idx],
                    &population[parent_b_idx],
                    sbx_eta,
                    &mut rng,
                ),
                "blx_alpha" => crossover::blx_alpha_crossover_impl(
                    &population[parent_a_idx],
                    &population[parent_b_idx],
                    blx_alpha,
                    &mut rng,
                ),
                _ => crossover::two_point_crossover_impl(
                    &population[parent_a_idx],
                    &population[parent_b_idx],
                    &mut rng,
                ),
            }
        } else {
            population[parent_a_idx].clone()
        };

        let mutated =
            mutation::gaussian_mutate_impl(&child, mutation_rate, mutation_strength, &mut rng);
        new_pop.push(mutated);
    }

    Ok(new_pop)
}

/// Flat-bytes version of evolve_generation.
/// Accepts numpy .tobytes() and returns flat bytes for numpy.frombuffer().
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    population_bytes,
    genome_size,
    fitness,
    elite_count = 3,
    mutation_rate = 0.15,
    mutation_strength = 0.2,
    crossover_rate = 0.7,
    tournament_k = 3,
    crossover_mode = "two_point",
    sbx_eta = 2.0,
    blx_alpha = 0.5,
))]
fn evolve_generation_flat(
    population_bytes: &[u8],
    genome_size: usize,
    fitness: Vec<f32>,
    elite_count: usize,
    mutation_rate: f64,
    mutation_strength: f64,
    crossover_rate: f64,
    tournament_k: usize,
    crossover_mode: &str,
    sbx_eta: f64,
    blx_alpha: f64,
) -> PyResult<Vec<u8>> {
    use rand::Rng;

    if genome_size == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "genome_size must be > 0",
        ));
    }

    let float_bytes = std::mem::size_of::<f32>();
    let stride = genome_size * float_bytes;
    let pop_size = population_bytes.len() / stride;

    if pop_size == 0 || fitness.len() != pop_size {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "population and fitness must be non-empty and same length",
        ));
    }

    // Reinterpret as f32 slices (zero-copy)
    let get_individual = |i: usize| -> &[f32] {
        let start = i * stride;
        unsafe {
            std::slice::from_raw_parts(
                population_bytes[start..].as_ptr() as *const f32,
                genome_size,
            )
        }
    };

    // Sort indices by fitness descending
    let mut indices: Vec<usize> = (0..pop_size).collect();
    indices.sort_by(|&a, &b| {
        fitness[b]
            .partial_cmp(&fitness[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Output buffer: flat f32 array
    let mut output = Vec::with_capacity(pop_size * genome_size);

    // Elitism
    let actual_elite = elite_count.min(pop_size);
    for &idx in indices.iter().take(actual_elite) {
        output.extend_from_slice(get_individual(idx));
    }

    let mut rng = rand::thread_rng();
    while output.len() < pop_size * genome_size {
        let parent_a_idx = selection::tournament_select_impl(&fitness, tournament_k, &mut rng);

        let child = if rng.gen::<f64>() < crossover_rate {
            let parent_b_idx = selection::tournament_select_impl(&fitness, tournament_k, &mut rng);
            match crossover_mode {
                "sbx" => crossover::sbx_crossover_impl(
                    get_individual(parent_a_idx),
                    get_individual(parent_b_idx),
                    sbx_eta,
                    &mut rng,
                ),
                "blx_alpha" => crossover::blx_alpha_crossover_impl(
                    get_individual(parent_a_idx),
                    get_individual(parent_b_idx),
                    blx_alpha,
                    &mut rng,
                ),
                _ => crossover::two_point_crossover_impl(
                    get_individual(parent_a_idx),
                    get_individual(parent_b_idx),
                    &mut rng,
                ),
            }
        } else {
            get_individual(parent_a_idx).to_vec()
        };

        let mutated =
            mutation::gaussian_mutate_impl(&child, mutation_rate, mutation_strength, &mut rng);
        output.extend_from_slice(&mutated);
    }

    // Convert to bytes
    let result_bytes: Vec<u8> = unsafe {
        let ptr = output.as_ptr() as *const u8;
        std::slice::from_raw_parts(ptr, output.len() * float_bytes).to_vec()
    };

    Ok(result_bytes)
}
