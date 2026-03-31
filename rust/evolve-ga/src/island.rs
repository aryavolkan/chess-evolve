//! Island model for parallel evolution with periodic migration.
//!
//! Implements migration between sub-populations (islands). Each island evolves
//! independently, and periodically the best individuals migrate to neighboring
//! islands. This helps maintain diversity while allowing specialization.

use pyo3::prelude::*;
use rand::seq::SliceRandom;

/// Perform migration between islands.
///
/// Implements ring-topology migration: each island sends its best `n_migrants`
/// individuals to the next island (wrapping around). The migrants replace the
/// worst individuals on the receiving island.
///
/// This is the simplest and most common island model topology. Ring topology
/// ensures that information flows gradually through the population, preventing
/// premature convergence while still sharing good genetic material.
///
/// Arguments:
///     islands: list of islands, each island is a list of weight vectors
///         (list[list[list[float]]])
///     island_fitness: fitness for each individual on each island
///         (list[list[float]])
///     n_migrants: number of individuals to migrate from each island (default 2)
///     topology: migration topology, one of "ring" or "random" (default "ring")
///
/// Returns:
///     Tuple of (new_islands, new_fitness) with migrations applied
#[allow(clippy::type_complexity)]
#[pyfunction]
#[pyo3(signature = (islands, island_fitness, n_migrants = 2, topology = "ring"))]
pub fn migrate(
    islands: Vec<Vec<Vec<f32>>>,
    island_fitness: Vec<Vec<f32>>,
    n_migrants: usize,
    topology: &str,
) -> PyResult<(Vec<Vec<Vec<f32>>>, Vec<Vec<f32>>)> {
    let n_islands = islands.len();
    if n_islands < 2 {
        return Ok((islands, island_fitness));
    }
    if islands.len() != island_fitness.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "islands and island_fitness must have same length",
        ));
    }
    for i in 0..n_islands {
        if islands[i].len() != island_fitness[i].len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "island {} population and fitness length mismatch",
                i
            )));
        }
    }

    // Determine migration pairs based on topology
    let pairs: Vec<(usize, usize)> = match topology {
        "random" => {
            let mut rng = rand::thread_rng();
            let mut sources: Vec<usize> = (0..n_islands).collect();
            sources.shuffle(&mut rng);
            let mut destinations: Vec<usize> = (0..n_islands).collect();
            // Ensure no island sends to itself
            loop {
                destinations.shuffle(&mut rng);
                if sources.iter().zip(destinations.iter()).all(|(s, d)| s != d) {
                    break;
                }
            }
            sources.into_iter().zip(destinations).collect()
        }
        _ => {
            // Ring topology: island i sends to island (i+1) % n
            (0..n_islands).map(|i| (i, (i + 1) % n_islands)).collect()
        }
    };

    // Clone islands and fitness for modification
    let mut new_islands = islands;
    let mut new_fitness = island_fitness;

    for (src, dst) in pairs {
        let src_pop_size = new_islands[src].len();
        let dst_pop_size = new_islands[dst].len();
        let actual_migrants = n_migrants.min(src_pop_size).min(dst_pop_size);

        if actual_migrants == 0 {
            continue;
        }

        // Find the best `actual_migrants` on source island
        let mut src_indices: Vec<usize> = (0..src_pop_size).collect();
        src_indices.sort_by(|&a, &b| {
            new_fitness[src][b]
                .partial_cmp(&new_fitness[src][a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let best_src_indices: Vec<usize> = src_indices[..actual_migrants].to_vec();

        // Find the worst `actual_migrants` on destination island
        let mut dst_indices: Vec<usize> = (0..dst_pop_size).collect();
        dst_indices.sort_by(|&a, &b| {
            new_fitness[dst][a]
                .partial_cmp(&new_fitness[dst][b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let worst_dst_indices: Vec<usize> = dst_indices[..actual_migrants].to_vec();

        // Collect migrants from source (clones)
        let migrants: Vec<(Vec<f32>, f32)> = best_src_indices
            .iter()
            .map(|&i| (new_islands[src][i].clone(), new_fitness[src][i]))
            .collect();

        // Replace worst on destination
        for (m_idx, &dst_idx) in worst_dst_indices.iter().enumerate() {
            new_islands[dst][dst_idx] = migrants[m_idx].0.clone();
            new_fitness[dst][dst_idx] = migrants[m_idx].1;
        }
    }

    Ok((new_islands, new_fitness))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_island(n: usize, base_val: f32) -> (Vec<Vec<f32>>, Vec<f32>) {
        let pop: Vec<Vec<f32>> = (0..n).map(|i| vec![base_val + i as f32; 5]).collect();
        let fitness: Vec<f32> = (0..n).map(|i| i as f32).collect();
        (pop, fitness)
    }

    #[test]
    fn ring_migration_moves_best_to_next() {
        let (pop0, fit0) = make_island(5, 0.0);
        let (pop1, fit1) = make_island(5, 100.0);

        let islands = vec![pop0, pop1];
        let fitness = vec![fit0, fit1];

        let (new_islands, new_fitness) = migrate(islands, fitness, 1, "ring").unwrap();

        assert_eq!(new_islands.len(), 2);
        assert_eq!(new_islands[0].len(), 5);
        assert_eq!(new_islands[1].len(), 5);

        // Best of island 0 (index 4, fitness 4.0) should migrate to island 1
        // replacing worst of island 1 (index 0, fitness 0.0)
        assert_eq!(new_fitness[1][0], 4.0);
    }

    #[test]
    fn single_island_no_migration() {
        let (pop, fit) = make_island(5, 0.0);
        let (new_islands, new_fitness) =
            migrate(vec![pop.clone()], vec![fit.clone()], 2, "ring").unwrap();
        assert_eq!(new_islands, vec![pop]);
        assert_eq!(new_fitness, vec![fit]);
    }

    #[test]
    fn migration_preserves_population_sizes() {
        let (pop0, fit0) = make_island(10, 0.0);
        let (pop1, fit1) = make_island(10, 50.0);
        let (pop2, fit2) = make_island(10, 100.0);

        let islands = vec![pop0, pop1, pop2];
        let fitness = vec![fit0, fit1, fit2];

        let (new_islands, new_fitness) = migrate(islands, fitness, 3, "ring").unwrap();

        for i in 0..3 {
            assert_eq!(new_islands[i].len(), 10);
            assert_eq!(new_fitness[i].len(), 10);
        }
    }

    #[test]
    fn random_topology_no_self_migration() {
        let (pop0, fit0) = make_island(5, 0.0);
        let (pop1, fit1) = make_island(5, 50.0);
        let (pop2, fit2) = make_island(5, 100.0);

        let islands = vec![pop0, pop1, pop2];
        let fitness = vec![fit0, fit1, fit2];

        // Just check it doesn't panic
        let (new_islands, _) = migrate(islands, fitness, 1, "random").unwrap();
        assert_eq!(new_islands.len(), 3);
    }
}
