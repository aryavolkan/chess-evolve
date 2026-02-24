//! Mutation operators for neural network weight vectors.
//!
//! Provides Gaussian mutation (standard and adaptive), Cauchy mutation
//! for heavy-tailed exploration, and the 1/5th success rule for step-size
//! adaptation inspired by Evolution Strategies / CMA-ES.

use pyo3::prelude::*;
use rand::Rng;
use rand_distr::{Cauchy, Distribution, Normal};

/// Gaussian mutation with geometric skip optimization.
///
/// Each weight is mutated with probability `mutation_rate` by adding
/// Gaussian noise N(0, strength). Uses the geometric-skip technique
/// (inverse CDF of geometric distribution) to only generate random numbers
/// for actual mutations -- O(k) instead of O(n) where k = expected mutations.
///
/// This matches the mutation strategy used in the existing GDScript
/// implementation.
///
/// Arguments:
///     weights: weight vector to mutate (not modified in place)
///     mutation_rate: probability each weight is mutated (0.0 to 1.0)
///     strength: standard deviation of Gaussian noise
///
/// Returns:
///     Mutated weight vector
#[pyfunction]
#[pyo3(signature = (weights, mutation_rate = 0.15, strength = 0.2))]
pub fn gaussian_mutate(weights: Vec<f32>, mutation_rate: f64, strength: f64) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    gaussian_mutate_impl(&weights, mutation_rate, strength, &mut rng)
}

/// Internal Gaussian mutation implementation.
pub fn gaussian_mutate_impl<R: Rng>(
    weights: &[f32],
    mutation_rate: f64,
    strength: f64,
    rng: &mut R,
) -> Vec<f32> {
    let mut result = weights.to_vec();
    let n = result.len();
    if n == 0 || mutation_rate <= 0.0 {
        return result;
    }

    let normal = Normal::new(0.0, strength).unwrap();

    if mutation_rate >= 1.0 {
        for w in result.iter_mut() {
            *w += normal.sample(rng) as f32;
        }
        return result;
    }

    // Geometric skip: jump to next mutation index using inverse CDF
    let log1mp = (1.0 - mutation_rate).ln();
    let mut i = (rng.gen::<f64>().max(1e-15).ln() / log1mp).floor() as usize;
    while i < n {
        result[i] += normal.sample(rng) as f32;
        let skip = (rng.gen::<f64>().max(1e-15).ln() / log1mp).floor() as usize;
        i += 1 + skip;
    }

    result
}

/// Adaptive Gaussian mutation with per-gene step sizes.
///
/// Each weight has its own step size (sigma) that co-evolves with the weight.
/// Step sizes are mutated first using the log-normal rule:
///   sigma_i' = sigma_i * exp(tau_global * N(0,1) + tau_local * N_i(0,1))
/// Then weights are mutated:
///   w_i' = w_i + sigma_i' * N_i(0,1)
///
/// The tau parameters control the learning rates for step-size adaptation:
///   tau_global = 1 / sqrt(2*n)
///   tau_local = 1 / sqrt(2*sqrt(n))
///
/// This is inspired by CMA-ES self-adaptation and is more powerful than
/// a single global mutation rate because it allows different regions of the
/// weight vector to evolve at different speeds.
///
/// Arguments:
///     weights: weight vector
///     sigmas: per-weight step sizes (same length as weights)
///     tau_global: global learning rate (default auto-computed from length)
///     tau_local: local learning rate (default auto-computed from length)
///     sigma_min: minimum allowed step size (prevents convergence to zero)
///
/// Returns:
///     Tuple of (mutated_weights, mutated_sigmas)
#[pyfunction]
#[pyo3(signature = (weights, sigmas, tau_global = None, tau_local = None, sigma_min = 1e-5))]
pub fn adaptive_gaussian_mutate(
    weights: Vec<f32>,
    sigmas: Vec<f32>,
    tau_global: Option<f64>,
    tau_local: Option<f64>,
    sigma_min: f64,
) -> PyResult<(Vec<f32>, Vec<f32>)> {
    if weights.len() != sigmas.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "weights and sigmas must have same length",
        ));
    }

    let n = weights.len();
    if n == 0 {
        return Ok((weights, sigmas));
    }

    let n_f = n as f64;
    let tau_g = tau_global.unwrap_or(1.0 / (2.0 * n_f).sqrt());
    let tau_l = tau_local.unwrap_or(1.0 / (2.0 * n_f.sqrt()).sqrt());

    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    let global_perturbation = tau_g * normal.sample(&mut rng);

    let mut new_weights = Vec::with_capacity(n);
    let mut new_sigmas = Vec::with_capacity(n);

    for i in 0..n {
        // Adapt step size
        let local_perturbation = tau_l * normal.sample(&mut rng);
        let new_sigma = (sigmas[i] as f64 * (global_perturbation + local_perturbation).exp())
            .max(sigma_min) as f32;

        // Mutate weight using adapted step size
        let new_weight = weights[i] + new_sigma * normal.sample(&mut rng) as f32;

        new_sigmas.push(new_sigma);
        new_weights.push(new_weight);
    }

    Ok((new_weights, new_sigmas))
}

/// Apply the 1/5th success rule to adapt a global mutation step size.
///
/// This is a simple but effective step-size adaptation rule from Evolution
/// Strategies: if more than 1/5 of mutations are successful (improved fitness),
/// increase the step size to explore more; if fewer succeed, decrease it to
/// exploit the current region.
///
/// The rule is: sigma *= factor^sign, where sign = +1 if success_rate > 1/5,
/// -1 otherwise.
///
/// Arguments:
///     current_sigma: current step size
///     success_count: number of successful mutations in the window
///     total_count: total number of mutations in the window
///     factor: adaptation factor (default 0.85; values in (0, 1))
///     sigma_min: minimum step size floor
///     sigma_max: maximum step size ceiling
///
/// Returns:
///     Adapted step size
#[pyfunction]
#[pyo3(signature = (
    current_sigma,
    success_count,
    total_count,
    factor = 0.85,
    sigma_min = 0.001,
    sigma_max = 1.0,
))]
pub fn one_fifth_rule_adapt(
    current_sigma: f64,
    success_count: usize,
    total_count: usize,
    factor: f64,
    sigma_min: f64,
    sigma_max: f64,
) -> f64 {
    if total_count == 0 {
        return current_sigma;
    }

    let success_rate = success_count as f64 / total_count as f64;
    let target = 0.2; // 1/5 rule

    let new_sigma = if success_rate > target {
        // Too many successes: step size too small, increase it
        current_sigma / factor
    } else if success_rate < target {
        // Too few successes: step size too large, decrease it
        current_sigma * factor
    } else {
        current_sigma
    };

    new_sigma.clamp(sigma_min, sigma_max)
}

/// Cauchy mutation for heavy-tailed exploration.
///
/// Similar to Gaussian mutation but uses a Cauchy distribution, which has
/// heavier tails. This means occasional large mutations that can help escape
/// local optima. Useful in the later stages of evolution when the population
/// may be converging prematurely.
///
/// Arguments:
///     weights: weight vector to mutate
///     mutation_rate: probability each weight is mutated
///     scale: scale parameter of the Cauchy distribution (analogous to strength)
///
/// Returns:
///     Mutated weight vector
#[pyfunction]
#[pyo3(signature = (weights, mutation_rate = 0.15, scale = 0.1))]
pub fn cauchy_mutate(weights: Vec<f32>, mutation_rate: f64, scale: f64) -> Vec<f32> {
    let n = weights.len();
    if n == 0 || mutation_rate <= 0.0 {
        return weights;
    }

    let mut result = weights;
    let mut rng = rand::thread_rng();
    let cauchy = Cauchy::new(0.0f64, scale).unwrap();

    if mutation_rate >= 1.0 {
        for w in result.iter_mut() {
            *w += cauchy.sample(&mut rng) as f32;
        }
        return result;
    }

    // Geometric skip
    let log1mp = (1.0 - mutation_rate).ln();
    let mut i = (rng.gen::<f64>().max(1e-15).ln() / log1mp).floor() as usize;
    while i < n {
        result[i] += cauchy.sample(&mut rng) as f32;
        let skip = (rng.gen::<f64>().max(1e-15).ln() / log1mp).floor() as usize;
        i += 1 + skip;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_mutate_preserves_length() {
        let w = vec![0.0f32; 100];
        let result = gaussian_mutate(w, 0.5, 0.1);
        assert_eq!(result.len(), 100);
    }

    #[test]
    fn gaussian_mutate_zero_rate_no_change() {
        let w = vec![1.0f32; 100];
        let result = gaussian_mutate(w.clone(), 0.0, 0.5);
        assert_eq!(result, w);
    }

    #[test]
    fn gaussian_mutate_full_rate_changes_all() {
        let w = vec![0.0f32; 1000];
        let result = gaussian_mutate(w, 1.0, 0.5);
        // With rate=1.0, every element should be mutated
        let changed = result.iter().filter(|&&v| v != 0.0).count();
        assert!(
            changed > 990,
            "expected almost all to change, got {}",
            changed
        );
    }

    #[test]
    fn adaptive_mutation_adjusts_sigmas() {
        let w = vec![0.0f32; 50];
        let sigmas = vec![0.1f32; 50];
        let (new_w, new_s) =
            adaptive_gaussian_mutate(w, sigmas.clone(), None, None, 1e-5).unwrap();
        assert_eq!(new_w.len(), 50);
        assert_eq!(new_s.len(), 50);
        let sigma_changed = new_s
            .iter()
            .zip(sigmas.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(sigma_changed, "sigmas should be adapted");
    }

    #[test]
    fn one_fifth_rule_increases_on_success() {
        let sigma = 0.1;
        let new_sigma = one_fifth_rule_adapt(sigma, 5, 10, 0.85, 0.001, 1.0);
        assert!(
            new_sigma > sigma,
            "sigma should increase with 50% success rate"
        );
    }

    #[test]
    fn one_fifth_rule_decreases_on_failure() {
        let sigma = 0.1;
        let new_sigma = one_fifth_rule_adapt(sigma, 0, 10, 0.85, 0.001, 1.0);
        assert!(
            new_sigma < sigma,
            "sigma should decrease with 0% success rate"
        );
    }

    #[test]
    fn one_fifth_rule_respects_bounds() {
        let low = one_fifth_rule_adapt(0.001, 0, 10, 0.85, 0.001, 1.0);
        assert!(low >= 0.001);

        let high = one_fifth_rule_adapt(1.0, 10, 10, 0.85, 0.001, 1.0);
        assert!(high <= 1.0);
    }

    #[test]
    fn cauchy_mutate_preserves_length() {
        let w = vec![0.0f32; 100];
        let result = cauchy_mutate(w, 0.5, 0.1);
        assert_eq!(result.len(), 100);
    }
}
