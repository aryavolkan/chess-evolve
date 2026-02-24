//! Selection strategies for genetic algorithms.
//!
//! Provides tournament selection, Stochastic Universal Sampling (SUS),
//! and rank-based selection -- all operating on fitness arrays.

use pyo3::prelude::*;
use rand::Rng;

/// Perform tournament selection and return the index of the winner.
///
/// Picks `k` random individuals and returns the one with the highest fitness.
///
/// Arguments:
///     fitness: list of fitness values
///     k: tournament size (default 3)
///
/// Returns:
///     Index of the selected individual
#[pyfunction]
#[pyo3(signature = (fitness, k = 3))]
pub fn tournament_select(fitness: Vec<f32>, k: usize) -> PyResult<usize> {
    if fitness.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "fitness must be non-empty",
        ));
    }
    let mut rng = rand::thread_rng();
    Ok(tournament_select_impl(&fitness, k, &mut rng))
}

/// Select `n` parents using tournament selection, returning their indices.
///
/// Arguments:
///     fitness: list of fitness values
///     n: number of parents to select
///     k: tournament size (default 3)
///
/// Returns:
///     List of selected indices
#[pyfunction]
#[pyo3(signature = (fitness, n, k = 3))]
pub fn tournament_select_batch(fitness: Vec<f32>, n: usize, k: usize) -> PyResult<Vec<usize>> {
    if fitness.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "fitness must be non-empty",
        ));
    }
    let mut rng = rand::thread_rng();
    let mut selected = Vec::with_capacity(n);
    for _ in 0..n {
        selected.push(tournament_select_impl(&fitness, k, &mut rng));
    }
    Ok(selected)
}

/// Internal implementation of tournament selection, usable from other modules.
pub fn tournament_select_impl<R: Rng>(fitness: &[f32], k: usize, rng: &mut R) -> usize {
    let n = fitness.len();
    let k = k.max(1).min(n);
    let mut best_idx = rng.gen_range(0..n);
    for _ in 1..k {
        let idx = rng.gen_range(0..n);
        if fitness[idx] > fitness[best_idx] {
            best_idx = idx;
        }
    }
    best_idx
}

/// Stochastic Universal Sampling (SUS).
///
/// Selects `n` individuals using evenly-spaced pointers on a roulette wheel,
/// reducing selection variance compared to standard roulette wheel selection.
/// Fitness values are shifted so the minimum becomes a small positive number
/// to handle negative fitness gracefully.
///
/// Arguments:
///     fitness: list of fitness values
///     n: number of individuals to select
///
/// Returns:
///     List of selected indices
#[pyfunction]
pub fn sus_select(fitness: Vec<f32>, n: usize) -> PyResult<Vec<usize>> {
    if fitness.is_empty() || n == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "fitness must be non-empty and n > 0",
        ));
    }

    let pop_size = fitness.len();

    // Shift fitness to be all positive (min becomes 1e-6)
    let min_f = fitness.iter().cloned().fold(f32::INFINITY, f32::min);
    let shifted: Vec<f32> = fitness.iter().map(|&f| f - min_f + 1e-6).collect();
    let total: f32 = shifted.iter().sum();

    if total <= 0.0 {
        // Degenerate case: uniform selection
        let mut rng = rand::thread_rng();
        return Ok((0..n).map(|_| rng.gen_range(0..pop_size)).collect());
    }

    let step = total / n as f32;
    let mut rng = rand::thread_rng();
    let start: f32 = rng.gen::<f32>() * step;

    let mut selected = Vec::with_capacity(n);
    let mut cumulative = shifted[0];
    let mut idx = 0;

    for i in 0..n {
        let pointer = start + i as f32 * step;
        while cumulative < pointer && idx + 1 < pop_size {
            idx += 1;
            cumulative += shifted[idx];
        }
        selected.push(idx);
    }

    Ok(selected)
}

/// Rank-based selection.
///
/// Assigns selection probabilities based on rank rather than raw fitness,
/// reducing the impact of super-fit individuals dominating selection.
/// Uses linear ranking: P(rank_i) proportional to
///   (2 - sp) + 2*(sp - 1)*(rank - 1)/(N-1)
/// where sp is selective pressure (1.0 to 2.0, default 1.5).
///
/// Arguments:
///     fitness: list of fitness values
///     n: number of individuals to select
///     selective_pressure: ranking pressure in [1.0, 2.0] (default 1.5)
///
/// Returns:
///     List of selected indices
#[pyfunction]
#[pyo3(signature = (fitness, n, selective_pressure = 1.5))]
pub fn rank_select(fitness: Vec<f32>, n: usize, selective_pressure: f64) -> PyResult<Vec<usize>> {
    if fitness.is_empty() || n == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "fitness must be non-empty and n > 0",
        ));
    }
    let sp = selective_pressure.clamp(1.0, 2.0);
    let pop_size = fitness.len();

    // Sort indices by fitness ascending (worst first, best last = highest rank)
    let mut indices: Vec<usize> = (0..pop_size).collect();
    indices.sort_by(|&a, &b| {
        fitness[a]
            .partial_cmp(&fitness[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Assign rank-based probabilities (linear ranking)
    // rank 0 = worst, rank N-1 = best
    let denom = if pop_size > 1 {
        (pop_size - 1) as f64
    } else {
        1.0
    };
    let mut probs = vec![0.0f64; pop_size];
    let mut total_prob = 0.0;
    for (rank, &original_idx) in indices.iter().enumerate() {
        let p = (2.0 - sp) + 2.0 * (sp - 1.0) * rank as f64 / denom;
        probs[original_idx] = p.max(0.0);
        total_prob += probs[original_idx];
    }

    if total_prob <= 0.0 {
        let mut rng = rand::thread_rng();
        return Ok((0..n).map(|_| rng.gen_range(0..pop_size)).collect());
    }

    // SUS-style sampling with rank probabilities
    let step = total_prob / n as f64;
    let mut rng = rand::thread_rng();
    let start: f64 = rng.gen::<f64>() * step;

    let mut selected = Vec::with_capacity(n);
    let mut cumulative = probs[0];
    let mut idx = 0;

    for i in 0..n {
        let pointer = start + i as f64 * step;
        while cumulative < pointer && idx + 1 < pop_size {
            idx += 1;
            cumulative += probs[idx];
        }
        selected.push(idx);
    }

    Ok(selected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tournament_selects_best_in_large_k() {
        let fitness = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        let mut rng = rand::thread_rng();
        // With k = pop_size, should always select the best
        for _ in 0..100 {
            let idx = tournament_select_impl(&fitness, 5, &mut rng);
            assert_eq!(idx, 1); // fitness[1] = 5.0 is the max
        }
    }

    #[test]
    fn sus_returns_correct_count() {
        let fitness = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = sus_select(fitness, 3).unwrap();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn rank_select_returns_correct_count() {
        let fitness = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = rank_select(fitness, 4, 1.5).unwrap();
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn rank_select_biases_toward_best() {
        let fitness = vec![0.0, 0.0, 0.0, 0.0, 100.0];
        let mut counts = [0u32; 5];
        for _ in 0..10000 {
            let result = rank_select(fitness.clone(), 1, 2.0).unwrap();
            counts[result[0]] += 1;
        }
        // With max selective pressure, best individual (idx 4) should be most often
        assert!(counts[4] > counts[0]);
    }

    #[test]
    fn tournament_batch_returns_correct_count() {
        let fitness = vec![1.0, 2.0, 3.0];
        let result = tournament_select_batch(fitness, 5, 2).unwrap();
        assert_eq!(result.len(), 5);
    }
}
