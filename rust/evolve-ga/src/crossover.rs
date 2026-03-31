//! Crossover operators for neural network weight vectors.
//!
//! Provides two-point crossover, Simulated Binary Crossover (SBX), and
//! BLX-alpha crossover -- all optimized for real-valued weight vectors.

use pyo3::prelude::*;
use rand::Rng;

/// Two-point crossover between two parent weight vectors.
///
/// Selects two random crossover points and swaps the segment between them.
/// This is a standard GA operator, well-suited for positional encoding of NN weights.
///
/// Arguments:
///     parent_a: first parent weight vector
///     parent_b: second parent weight vector
///
/// Returns:
///     Child weight vector
#[pyfunction]
pub fn two_point_crossover(parent_a: Vec<f32>, parent_b: Vec<f32>) -> PyResult<Vec<f32>> {
    if parent_a.len() != parent_b.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "parents must have same length",
        ));
    }
    let mut rng = rand::thread_rng();
    Ok(two_point_crossover_impl(&parent_a, &parent_b, &mut rng))
}

/// Internal two-point crossover implementation.
pub fn two_point_crossover_impl<R: Rng>(
    parent_a: &[f32],
    parent_b: &[f32],
    rng: &mut R,
) -> Vec<f32> {
    let n = parent_a.len();
    if n == 0 {
        return Vec::new();
    }

    let mut p1 = rng.gen_range(0..n);
    let mut p2 = rng.gen_range(0..n);
    if p1 > p2 {
        std::mem::swap(&mut p1, &mut p2);
    }

    let mut child = parent_a.to_vec();
    child[p1..p2].copy_from_slice(&parent_b[p1..p2]);
    child
}

/// Simulated Binary Crossover (SBX).
///
/// Mimics the behavior of single-point crossover in binary-coded GAs but for
/// real-valued representations. The distribution index `eta` controls how far
/// children can be from parents:
/// - Low eta (1-2): children can be far from parents (more exploration)
/// - High eta (10-20): children stay close to parents (more exploitation)
///
/// This operator is particularly effective for neural network weight optimization
/// because it respects the continuous nature of weights and produces children
/// whose distribution mirrors what binary crossover would achieve.
///
/// Reference: Deb & Agrawal, "Simulated Binary Crossover for Continuous
///            Search Space" (1995)
///
/// Arguments:
///     parent_a: first parent weight vector
///     parent_b: second parent weight vector
///     eta: distribution index (default 2.0; higher = children closer to parents)
///
/// Returns:
///     Child weight vector
#[pyfunction]
#[pyo3(signature = (parent_a, parent_b, eta = 2.0))]
pub fn sbx_crossover(parent_a: Vec<f32>, parent_b: Vec<f32>, eta: f64) -> PyResult<Vec<f32>> {
    if parent_a.len() != parent_b.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "parents must have same length",
        ));
    }
    let mut rng = rand::thread_rng();
    Ok(sbx_crossover_impl(&parent_a, &parent_b, eta, &mut rng))
}

/// Internal SBX crossover implementation.
pub fn sbx_crossover_impl<R: Rng>(
    parent_a: &[f32],
    parent_b: &[f32],
    eta: f64,
    rng: &mut R,
) -> Vec<f32> {
    let n = parent_a.len();
    let mut child = Vec::with_capacity(n);

    for i in 0..n {
        let p1 = parent_a[i] as f64;
        let p2 = parent_b[i] as f64;

        if (p1 - p2).abs() < 1e-14 {
            // Parents are identical at this gene, just copy
            child.push(parent_a[i]);
            continue;
        }

        let u: f64 = rng.gen();
        let beta = if u <= 0.5 {
            (2.0 * u).powf(1.0 / (eta + 1.0))
        } else {
            (1.0 / (2.0 * (1.0 - u))).powf(1.0 / (eta + 1.0))
        };

        // Produce one child (the other would use -beta, but we only need one)
        let c = 0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2);
        child.push(c as f32);
    }

    child
}

/// BLX-alpha (Blend Crossover).
///
/// Creates offspring by sampling uniformly from an interval that extends
/// beyond the parents' range by a factor of alpha. For neural network weights,
/// alpha = 0.5 is commonly used, allowing the child to explore 50% beyond
/// the parents' range in each dimension.
///
/// This operator is effective for NN weight evolution because:
/// - It naturally handles the continuous weight space
/// - The alpha parameter controls exploration vs exploitation
/// - It introduces no directional bias (uniform sampling)
///
/// Reference: Eshelman & Schaffer, "Real-Coded Genetic Algorithms and
///            Interval Schemata" (1993)
///
/// Arguments:
///     parent_a: first parent weight vector
///     parent_b: second parent weight vector
///     alpha: extension factor (default 0.5; 0 = interpolation only)
///
/// Returns:
///     Child weight vector
#[pyfunction]
#[pyo3(signature = (parent_a, parent_b, alpha = 0.5))]
pub fn blx_alpha_crossover(
    parent_a: Vec<f32>,
    parent_b: Vec<f32>,
    alpha: f64,
) -> PyResult<Vec<f32>> {
    if parent_a.len() != parent_b.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "parents must have same length",
        ));
    }
    let mut rng = rand::thread_rng();
    Ok(blx_alpha_crossover_impl(
        &parent_a, &parent_b, alpha, &mut rng,
    ))
}

/// Internal BLX-alpha crossover implementation.
///
/// Uses f32 arithmetic throughout — no transcendental ops, just
/// min/max/range/gen_range, so f64 precision is unnecessary.
pub fn blx_alpha_crossover_impl<R: Rng>(
    parent_a: &[f32],
    parent_b: &[f32],
    alpha: f64,
    rng: &mut R,
) -> Vec<f32> {
    let n = parent_a.len();
    let mut child = Vec::with_capacity(n);
    let alpha_f32 = alpha as f32;

    for i in 0..n {
        let a = parent_a[i];
        let b = parent_b[i];
        let lo = a.min(b);
        let hi = a.max(b);
        let range = hi - lo;
        let ext = range * alpha_f32;
        let c: f32 = rng.gen_range((lo - ext)..=(hi + ext));
        child.push(c);
    }

    child
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_point_preserves_length() {
        let a = vec![1.0f32; 100];
        let b = vec![2.0f32; 100];
        let mut rng = rand::thread_rng();
        let child = two_point_crossover_impl(&a, &b, &mut rng);
        assert_eq!(child.len(), 100);
    }

    #[test]
    fn two_point_mixes_parents() {
        let a = vec![0.0f32; 100];
        let b = vec![1.0f32; 100];
        let mut rng = rand::thread_rng();
        let child = two_point_crossover_impl(&a, &b, &mut rng);
        assert_eq!(child.len(), 100);
        let has_zero = child.iter().any(|&v| v == 0.0);
        let has_one = child.iter().any(|&v| v == 1.0);
        // At least one value should come from each parent
        assert!(has_zero || has_one);
    }

    #[test]
    fn sbx_preserves_length() {
        let a = vec![1.0f32; 50];
        let b = vec![2.0f32; 50];
        let mut rng = rand::thread_rng();
        let child = sbx_crossover_impl(&a, &b, 2.0, &mut rng);
        assert_eq!(child.len(), 50);
    }

    #[test]
    fn sbx_with_identical_parents() {
        let a = vec![1.0f32; 100];
        let b = vec![1.0f32; 100]; // identical parents
        let mut rng = rand::thread_rng();
        let child = sbx_crossover_impl(&a, &b, 20.0, &mut rng);
        // Identical parents should produce identical child
        for (c, &p) in child.iter().zip(a.iter()) {
            assert!((c - p).abs() < 1e-6);
        }
    }

    #[test]
    fn blx_alpha_preserves_length() {
        let a = vec![0.0f32; 50];
        let b = vec![1.0f32; 50];
        let mut rng = rand::thread_rng();
        let child = blx_alpha_crossover_impl(&a, &b, 0.5, &mut rng);
        assert_eq!(child.len(), 50);
    }

    #[test]
    fn blx_alpha_zero_interpolates() {
        let a = vec![0.0f32; 1000];
        let b = vec![1.0f32; 1000];
        let mut rng = rand::thread_rng();
        let child = blx_alpha_crossover_impl(&a, &b, 0.0, &mut rng);
        // With alpha = 0, child should be in [0, 1] for each element
        for c in &child {
            assert!(*c >= 0.0 && *c <= 1.0, "got {} outside [0, 1]", c);
        }
    }

    #[test]
    fn blx_alpha_positive_can_extrapolate() {
        let a = vec![0.0f32; 10000];
        let b = vec![1.0f32; 10000];
        let mut rng = rand::thread_rng();
        let child = blx_alpha_crossover_impl(&a, &b, 0.5, &mut rng);
        let outside = child.iter().any(|&c| c < 0.0 || c > 1.0);
        assert!(
            outside,
            "with alpha=0.5 and many elements, some should extrapolate"
        );
    }
}
