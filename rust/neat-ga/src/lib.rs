use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::SeedableRng;

pub mod config;
pub mod evolution;
pub mod genome;
pub mod innovation;
pub mod species;

/// Create an initial NEAT population. Returns a list of genome JSON strings.
#[pyfunction]
fn create_population(py: Python<'_>, config_json: &str) -> PyResult<Vec<String>> {
    let config: config::NeatConfig = serde_json::from_str(config_json).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Invalid config JSON: {e}"))
    })?;

    let node_count = (config.input_count + config.output_count) as u32;
    let mut tracker = innovation::InnovationTracker::new(node_count);
    let mut rng = rand::rngs::SmallRng::from_entropy();

    let mut pop = Vec::with_capacity(config.population_size);
    for _ in 0..config.population_size {
        let genome = genome::NeatGenome::create_basic(
            config.input_count,
            config.output_count,
            config.initial_connections_per_output,
            &mut tracker,
            &mut rng,
        );
        let json = serde_json::to_string(&genome).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Serialize error: {e}"))
        })?;
        pop.push(json);
    }

    // Return tracker state so Python can persist it
    let tracker_json = serde_json::to_string(&tracker).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Serialize tracker error: {e}"))
    })?;

    // Use py to create a dict with population + tracker
    let result = PyDict::new(py);
    result.set_item("population", pop)?;
    result.set_item("tracker", tracker_json)?;

    // Actually, the plan says return list[str], so let's keep it simple
    // and return a dict with both
    // Wait, the plan interface says -> list[str], but we need tracker too.
    // Let's return a dict to be more useful.
    Ok(result
        .get_item("population")?
        .unwrap()
        .extract::<Vec<String>>()?)
}

/// Create an initial NEAT population with tracker state.
/// Returns dict with "population" (list[str]) and "tracker" (str).
#[pyfunction]
fn create_population_with_tracker(
    py: Python<'_>,
    config_json: &str,
) -> PyResult<Py<PyDict>> {
    let config: config::NeatConfig = serde_json::from_str(config_json).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Invalid config JSON: {e}"))
    })?;

    let node_count = (config.input_count + config.output_count) as u32;
    let mut tracker = innovation::InnovationTracker::new(node_count);
    let mut rng = rand::rngs::SmallRng::from_entropy();

    let mut pop = Vec::with_capacity(config.population_size);
    for _ in 0..config.population_size {
        let genome = genome::NeatGenome::create_basic(
            config.input_count,
            config.output_count,
            config.initial_connections_per_output,
            &mut tracker,
            &mut rng,
        );
        let json = serde_json::to_string(&genome).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Serialize error: {e}"))
        })?;
        pop.push(json);
    }

    let tracker_json = serde_json::to_string(&tracker).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Serialize tracker error: {e}"))
    })?;

    let result = PyDict::new(py);
    result.set_item("population", pop)?;
    result.set_item("tracker", tracker_json)?;

    Ok(result.into())
}

/// Run one NEAT evolution step.
///
/// Arguments:
///     population_json: list of genome JSON strings
///     fitness: fitness values aligned with population
///     species_json: serialized species state (or "[]" for initial)
///     config_json: NeatConfig JSON
///     tracker_json: InnovationTracker JSON
///
/// Returns dict with:
///     "population": list[str]  (new genome JSONs)
///     "species": str           (updated species state)
///     "tracker": str           (updated tracker state)
///     "config": str            (updated config — threshold may have changed)
///     "stats": dict            (species_count, best_fitness, etc.)
#[pyfunction]
fn evolve_neat_generation(
    py: Python<'_>,
    population_json: Vec<String>,
    fitness: Vec<f32>,
    species_json: &str,
    config_json: &str,
    tracker_json: &str,
) -> PyResult<Py<PyDict>> {
    let mut config: config::NeatConfig = serde_json::from_str(config_json).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Invalid config JSON: {e}"))
    })?;
    let mut tracker: innovation::InnovationTracker =
        serde_json::from_str(tracker_json).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid tracker JSON: {e}"))
        })?;

    // Deserialize species. Accept "[]" or "null" for initial state.
    let mut species_list: Vec<species::Species> = if species_json.is_empty()
        || species_json == "[]"
        || species_json == "null"
    {
        Vec::new()
    } else {
        serde_json::from_str(species_json).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid species JSON: {e}"))
        })?
    };

    // Deserialize next_species_id from species list
    let mut next_species_id = species_list.iter().map(|s| s.id + 1).max().unwrap_or(0);

    // Deserialize population
    let mut population: Vec<genome::NeatGenome> = population_json
        .iter()
        .map(|s| serde_json::from_str(s))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid genome JSON: {e}"))
        })?;

    let mut rng = rand::rngs::SmallRng::from_entropy();

    let stats = evolution::evolve_neat_generation(
        &mut population,
        &fitness,
        &mut species_list,
        &mut config,
        &mut tracker,
        &mut next_species_id,
        &mut rng,
    );

    // Serialize results
    let new_pop_json: Vec<String> = population
        .iter()
        .map(|g| serde_json::to_string(g))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Serialize population error: {e}"))
        })?;

    let new_species_json = serde_json::to_string(&species_list).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Serialize species error: {e}"))
    })?;

    let new_tracker_json = serde_json::to_string(&tracker).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Serialize tracker error: {e}"))
    })?;

    let new_config_json = serde_json::to_string(&config).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Serialize config error: {e}"))
    })?;

    let stats_dict = PyDict::new(py);
    stats_dict.set_item("species_count", stats.species_count)?;
    stats_dict.set_item("best_fitness", stats.best_fitness)?;
    stats_dict.set_item("avg_fitness", stats.avg_fitness)?;
    stats_dict.set_item("avg_connections", stats.avg_connections)?;
    stats_dict.set_item("avg_nodes", stats.avg_nodes)?;
    stats_dict.set_item("best_connections", stats.best_connections)?;
    stats_dict.set_item("best_nodes", stats.best_nodes)?;

    let result = PyDict::new(py);
    result.set_item("population", new_pop_json)?;
    result.set_item("species", new_species_json)?;
    result.set_item("tracker", new_tracker_json)?;
    result.set_item("config", new_config_json)?;
    result.set_item("stats", stats_dict)?;

    Ok(result.into())
}

/// Create a seeded NEAT population from a template genome.
///
/// Clones the template topology for all individuals but randomizes weights/biases.
/// The tracker is initialized from the template to avoid innovation number collisions.
#[pyfunction]
fn create_seeded_population(
    py: Python<'_>,
    config_json: &str,
    seed_genome_json: &str,
) -> PyResult<Py<PyDict>> {
    let config: config::NeatConfig = serde_json::from_str(config_json).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Invalid config JSON: {e}"))
    })?;
    let seed: genome::NeatGenome = serde_json::from_str(seed_genome_json).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Invalid seed genome JSON: {e}"))
    })?;

    let node_count = seed.nodes.iter().map(|n| n.id).max().unwrap_or(0) + 1;
    let mut tracker = innovation::InnovationTracker::new(node_count);
    tracker.seed_from_genome(&seed);
    let mut rng = rand::rngs::SmallRng::from_entropy();

    let mut pop = Vec::with_capacity(config.population_size);
    for _ in 0..config.population_size {
        let g = genome::NeatGenome::create_from_topology(&seed, &mut rng);
        let json = serde_json::to_string(&g).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Serialize error: {e}"))
        })?;
        pop.push(json);
    }

    let tracker_json = serde_json::to_string(&tracker).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Serialize tracker error: {e}"))
    })?;

    let result = PyDict::new(py);
    result.set_item("population", pop)?;
    result.set_item("tracker", tracker_json)?;

    Ok(result.into())
}

/// Compute compatibility distance between two genomes (for debugging).
#[pyfunction]
fn compatibility_distance(
    genome_a_json: &str,
    genome_b_json: &str,
    config_json: &str,
) -> PyResult<f32> {
    let a: genome::NeatGenome = serde_json::from_str(genome_a_json).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Invalid genome A JSON: {e}"))
    })?;
    let b: genome::NeatGenome = serde_json::from_str(genome_b_json).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Invalid genome B JSON: {e}"))
    })?;
    let config: config::NeatConfig = serde_json::from_str(config_json).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Invalid config JSON: {e}"))
    })?;

    Ok(genome::NeatGenome::compatibility_distance(&a, &b, &config))
}

#[pymodule]
fn neat_ga(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(create_population, m)?)?;
    m.add_function(wrap_pyfunction!(create_population_with_tracker, m)?)?;
    m.add_function(wrap_pyfunction!(create_seeded_population, m)?)?;
    m.add_function(wrap_pyfunction!(evolve_neat_generation, m)?)?;
    m.add_function(wrap_pyfunction!(compatibility_distance, m)?)?;
    Ok(())
}
