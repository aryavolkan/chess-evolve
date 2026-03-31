//! Speciation and fitness sharing for maintaining population diversity.
//!
//! Implements distance-based speciation (assigning individuals to species based
//! on weight-vector distance) and fitness sharing (adjusting fitness by species
//! size to prevent any single niche from dominating).

use pyo3::prelude::*;

/// Assign individuals to species based on Euclidean distance between weight vectors.
///
/// Uses a compatibility threshold to determine whether an individual belongs to
/// an existing species (represented by its centroid/representative) or should
/// start a new species. This is inspired by NEAT's speciation but adapted for
/// fixed-topology networks where distance is simply the L2 norm of weight
/// differences.
///
/// The algorithm:
/// 1. Start with no species (or use provided representatives from last gen).
/// 2. For each individual, compute distance to each existing species representative.
/// 3. If the minimum distance is below the threshold, assign to that species.
/// 4. Otherwise, create a new species with this individual as representative.
///
/// Arguments:
///     population: list of weight vectors (list[list[float]])
///     threshold: compatibility distance threshold
///     representatives: optional list of existing species representatives from
///         the previous generation (list[list[float]]). If not provided, species
///         are assigned from scratch.
///
/// Returns:
///     Tuple of (species_ids, new_representatives) where:
///     - species_ids: list[int] mapping each individual to a species ID
///     - new_representatives: list[list[float]] -- one representative per species
#[pyfunction]
#[pyo3(signature = (population, threshold = 3.0, representatives = None))]
pub fn assign_species(
    population: Vec<Vec<f32>>,
    threshold: f64,
    representatives: Option<Vec<Vec<f32>>>,
) -> PyResult<(Vec<usize>, Vec<Vec<f32>>)> {
    if population.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    let threshold = threshold as f32;
    let mut species_reps: Vec<Vec<f32>> = representatives.unwrap_or_default();
    let mut species_ids = Vec::with_capacity(population.len());

    for individual in &population {
        let mut assigned = false;
        let mut min_dist = f32::INFINITY;
        let mut best_species = 0;

        for (sid, rep) in species_reps.iter().enumerate() {
            let dist = euclidean_distance(individual, rep);
            if dist < min_dist {
                min_dist = dist;
                best_species = sid;
            }
        }

        if min_dist <= threshold && !species_reps.is_empty() {
            species_ids.push(best_species);
            assigned = true;
        }

        if !assigned {
            // Create new species
            let new_id = species_reps.len();
            species_reps.push(individual.clone());
            species_ids.push(new_id);
        }
    }

    Ok((species_ids, species_reps))
}

/// Compute shared fitness values (fitness sharing / niching).
///
/// Adjusts each individual's fitness by dividing by its species size.
/// This prevents large species from dominating the population and encourages
/// diversity by giving small, novel species a proportionally higher per-capita
/// fitness allocation.
///
/// Shared fitness for individual i in species s:
///   f_shared(i) = f(i) / |s|
///
/// Optionally applies a minimum species size to prevent division by very small
/// numbers (which would give outsized fitness to lone individuals).
///
/// Arguments:
///     fitness: raw fitness values (list[float])
///     species_ids: species assignment for each individual (list[int])
///     min_species_size: minimum species size for sharing denominator (default 1)
///
/// Returns:
///     Shared fitness values (list[float])
#[pyfunction]
#[pyo3(signature = (fitness, species_ids, min_species_size = 1))]
pub fn shared_fitness(
    fitness: Vec<f32>,
    species_ids: Vec<usize>,
    min_species_size: usize,
) -> PyResult<Vec<f32>> {
    if fitness.len() != species_ids.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "fitness and species_ids must have same length",
        ));
    }

    let n = fitness.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    // Count species sizes
    let max_species = species_ids.iter().cloned().max().unwrap_or(0) + 1;
    let mut species_sizes = vec![0usize; max_species];
    for &sid in &species_ids {
        if sid < max_species {
            species_sizes[sid] += 1;
        }
    }

    let min_size = min_species_size.max(1);

    // Compute shared fitness
    let mut shared = Vec::with_capacity(n);
    for i in 0..n {
        let sid = species_ids[i];
        let size = if sid < max_species {
            species_sizes[sid].max(min_size)
        } else {
            min_size
        };
        shared.push(fitness[i] / size as f32);
    }

    Ok(shared)
}

/// Compute Euclidean distance between two weight vectors.
///
/// Uses iterator zip for better auto-vectorization (compiler gets aliasing info).
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
        .sqrt()
}

/// Assign species using flat byte buffer (avoids Python list-of-lists overhead).
///
/// population_bytes: numpy .tobytes() — flat f32 array (pop_size * genome_size * 4 bytes)
/// genome_size: number of floats per individual
/// representatives_bytes: optional flat byte buffer for previous reps (n_reps * genome_size * 4 bytes)
/// n_reps: number of representatives (needed to split the flat buffer)
#[pyfunction]
#[pyo3(signature = (population_bytes, genome_size, threshold = 3.0, representatives_bytes = None, n_reps = 0))]
pub fn assign_species_flat(
    population_bytes: &[u8],
    genome_size: usize,
    threshold: f64,
    representatives_bytes: Option<&[u8]>,
    n_reps: usize,
) -> PyResult<(Vec<usize>, Vec<u8>)> {
    if genome_size == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "genome_size must be > 0",
        ));
    }

    let float_bytes = std::mem::size_of::<f32>();
    let stride = genome_size * float_bytes;
    let pop_size = population_bytes.len() / stride;

    if pop_size == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    let threshold = threshold as f32;

    // Parse representatives from flat bytes
    let initial_reps: Vec<&[f32]> = if let Some(rep_bytes) = representatives_bytes {
        (0..n_reps)
            .map(|i| {
                let start = i * stride;
                let end = start + stride;
                let bytes = &rep_bytes[start..end];
                // Safety: f32 alignment from numpy tobytes is guaranteed
                unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, genome_size) }
            })
            .collect()
    } else {
        Vec::new()
    };

    // Reinterpret population bytes as f32 slices
    let pop_slices: Vec<&[f32]> = (0..pop_size)
        .map(|i| {
            let start = i * stride;
            let bytes = &population_bytes[start..start + stride];
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, genome_size) }
        })
        .collect();

    let mut species_ids = Vec::with_capacity(pop_size);
    let mut owned_reps: Vec<Vec<f32>> = initial_reps.iter().map(|s| s.to_vec()).collect();

    for individual in &pop_slices {
        let mut min_dist = f32::INFINITY;
        let mut best_species = 0;

        for (sid, rep) in owned_reps.iter().enumerate() {
            let dist = euclidean_distance(individual, rep);
            if dist < min_dist {
                min_dist = dist;
                best_species = sid;
            }
        }

        if min_dist <= threshold && !owned_reps.is_empty() {
            species_ids.push(best_species);
        } else {
            let new_id = owned_reps.len();
            owned_reps.push(individual.to_vec());
            species_ids.push(new_id);
        }
    }

    // Return reps as flat bytes
    let mut reps_bytes = Vec::with_capacity(owned_reps.len() * stride);
    for rep in &owned_reps {
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(rep.as_ptr() as *const u8, rep.len() * float_bytes)
        };
        reps_bytes.extend_from_slice(bytes);
    }

    Ok((species_ids, reps_bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_individuals_same_species() {
        let pop = vec![vec![1.0f32; 10], vec![1.0f32; 10], vec![1.0f32; 10]];
        let (ids, reps) = assign_species(pop, 3.0, None).unwrap();
        assert_eq!(ids.len(), 3);
        assert_eq!(reps.len(), 1);
        assert!(ids.iter().all(|&id| id == 0));
    }

    #[test]
    fn distant_individuals_different_species() {
        let pop = vec![
            vec![0.0f32; 10],
            vec![100.0f32; 10], // Very far from [0; 10]
        ];
        let (ids, reps) = assign_species(pop, 1.0, None).unwrap();
        assert_eq!(ids.len(), 2);
        assert_ne!(ids[0], ids[1]);
        assert_eq!(reps.len(), 2);
    }

    #[test]
    fn shared_fitness_divides_by_species_size() {
        let fitness = vec![10.0f32, 10.0, 10.0, 20.0];
        let species_ids = vec![0, 0, 0, 1]; // species 0 has 3, species 1 has 1
        let shared = shared_fitness(fitness, species_ids, 1).unwrap();
        assert_eq!(shared.len(), 4);
        assert!((shared[0] - 10.0 / 3.0).abs() < 1e-5);
        assert!((shared[1] - 10.0 / 3.0).abs() < 1e-5);
        assert!((shared[2] - 10.0 / 3.0).abs() < 1e-5);
        assert!((shared[3] - 20.0).abs() < 1e-5);
    }

    #[test]
    fn shared_fitness_min_species_size() {
        let fitness = vec![10.0f32];
        let species_ids = vec![0];
        let shared = shared_fitness(fitness, species_ids, 5).unwrap();
        assert!((shared[0] - 10.0 / 5.0).abs() < 1e-5);
    }

    #[test]
    fn assign_species_with_previous_reps() {
        let reps = vec![vec![0.0f32; 5], vec![10.0f32; 5]];
        let pop = vec![
            vec![0.1f32; 5], // Close to rep 0
            vec![9.9f32; 5], // Close to rep 1
            vec![0.2f32; 5], // Close to rep 0
        ];
        let (ids, _) = assign_species(pop, 3.0, Some(reps)).unwrap();
        assert_eq!(ids[0], 0);
        assert_eq!(ids[1], 1);
        assert_eq!(ids[2], 0);
    }

    #[test]
    fn euclidean_distance_correct() {
        let a = vec![0.0f32, 0.0, 0.0];
        let b = vec![3.0f32, 4.0, 0.0];
        assert!((euclidean_distance(&a, &b) - 5.0).abs() < 1e-5);
    }
}
