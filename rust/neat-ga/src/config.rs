use serde::{Deserialize, Serialize};

/// All NEAT hyperparameters. Mirrors ai/neat_config.gd.
#[derive(Clone, Serialize, Deserialize)]
pub struct NeatConfig {
    #[serde(default = "default_input_count")]
    pub input_count: usize,
    #[serde(default = "default_output_count")]
    pub output_count: usize,
    #[serde(default = "default_population_size")]
    pub population_size: usize,
    #[serde(default = "default_initial_connections_per_output")]
    pub initial_connections_per_output: usize,

    // Compatibility / speciation
    #[serde(default = "default_compatibility_threshold")]
    pub compatibility_threshold: f32,
    #[serde(default = "default_c1_excess")]
    pub c1_excess: f32,
    #[serde(default = "default_c2_disjoint")]
    pub c2_disjoint: f32,
    #[serde(default = "default_c3_weight_diff")]
    pub c3_weight_diff: f32,
    #[serde(default = "default_target_species_count")]
    pub target_species_count: usize,
    #[serde(default = "default_threshold_step")]
    pub threshold_step: f32,

    // Mutation rates
    #[serde(default = "default_weight_mutate_rate")]
    pub weight_mutate_rate: f32,
    #[serde(default = "default_weight_perturb_rate")]
    pub weight_perturb_rate: f32,
    #[serde(default = "default_weight_perturb_strength")]
    pub weight_perturb_strength: f32,
    #[serde(default = "default_weight_reset_range")]
    pub weight_reset_range: f32,
    #[serde(default = "default_add_node_rate")]
    pub add_node_rate: f32,
    #[serde(default = "default_add_connection_rate")]
    pub add_connection_rate: f32,
    #[serde(default = "default_disable_connection_rate")]
    pub disable_connection_rate: f32,
    #[serde(default = "default_add_node_count")]
    pub add_node_count: usize,
    #[serde(default = "default_add_connection_count")]
    pub add_connection_count: usize,

    // Pruning & parsimony
    #[serde(default = "default_prune_rate")]
    pub prune_rate: f32,
    #[serde(default = "default_complexity_cost")]
    pub complexity_cost: f32,

    // Reproduction
    #[serde(default = "default_elite_fraction")]
    pub elite_fraction: f32,
    #[serde(default = "default_survival_fraction")]
    pub survival_fraction: f32,
    #[serde(default = "default_crossover_rate")]
    pub crossover_rate: f32,
    #[serde(default = "default_interspecies_crossover_rate")]
    pub interspecies_crossover_rate: f32,
    #[serde(default = "default_disabled_gene_inherit_rate")]
    pub disabled_gene_inherit_rate: f32,

    // Stagnation
    #[serde(default = "default_stagnation_threshold")]
    pub stagnation_threshold: usize,
    #[serde(default = "default_stagnation_kill_threshold")]
    pub stagnation_kill_threshold: usize,
    #[serde(default = "default_min_species_protected")]
    pub min_species_protected: usize,
}

fn default_input_count() -> usize { 389 }
fn default_output_count() -> usize { 128 }
fn default_population_size() -> usize { 50 }
fn default_initial_connections_per_output() -> usize { 5 }
fn default_compatibility_threshold() -> f32 { 3.0 }
fn default_c1_excess() -> f32 { 1.0 }
fn default_c2_disjoint() -> f32 { 1.0 }
fn default_c3_weight_diff() -> f32 { 0.4 }
fn default_target_species_count() -> usize { 4 }
fn default_threshold_step() -> f32 { 0.3 }
fn default_weight_mutate_rate() -> f32 { 0.8 }
fn default_weight_perturb_rate() -> f32 { 0.9 }
fn default_weight_perturb_strength() -> f32 { 0.3 }
fn default_weight_reset_range() -> f32 { 2.0 }
fn default_add_node_rate() -> f32 { 0.10 }
fn default_add_connection_rate() -> f32 { 0.20 }
fn default_disable_connection_rate() -> f32 { 0.01 }
fn default_add_node_count() -> usize { 1 }
fn default_add_connection_count() -> usize { 1 }
fn default_prune_rate() -> f32 { 0.1 }
fn default_complexity_cost() -> f32 { 0.0 }
fn default_elite_fraction() -> f32 { 0.1 }
fn default_survival_fraction() -> f32 { 0.5 }
fn default_crossover_rate() -> f32 { 0.75 }
fn default_interspecies_crossover_rate() -> f32 { 0.001 }
fn default_disabled_gene_inherit_rate() -> f32 { 0.75 }
fn default_stagnation_threshold() -> usize { 15 }
fn default_stagnation_kill_threshold() -> usize { 25 }
fn default_min_species_protected() -> usize { 2 }

impl Default for NeatConfig {
    fn default() -> Self {
        Self {
            input_count: default_input_count(),
            output_count: default_output_count(),
            population_size: default_population_size(),
            initial_connections_per_output: default_initial_connections_per_output(),
            compatibility_threshold: default_compatibility_threshold(),
            c1_excess: default_c1_excess(),
            c2_disjoint: default_c2_disjoint(),
            c3_weight_diff: default_c3_weight_diff(),
            target_species_count: default_target_species_count(),
            threshold_step: default_threshold_step(),
            weight_mutate_rate: default_weight_mutate_rate(),
            weight_perturb_rate: default_weight_perturb_rate(),
            weight_perturb_strength: default_weight_perturb_strength(),
            weight_reset_range: default_weight_reset_range(),
            add_node_rate: default_add_node_rate(),
            add_connection_rate: default_add_connection_rate(),
            disable_connection_rate: default_disable_connection_rate(),
            add_node_count: default_add_node_count(),
            add_connection_count: default_add_connection_count(),
            prune_rate: default_prune_rate(),
            complexity_cost: default_complexity_cost(),
            elite_fraction: default_elite_fraction(),
            survival_fraction: default_survival_fraction(),
            crossover_rate: default_crossover_rate(),
            interspecies_crossover_rate: default_interspecies_crossover_rate(),
            disabled_gene_inherit_rate: default_disabled_gene_inherit_rate(),
            stagnation_threshold: default_stagnation_threshold(),
            stagnation_kill_threshold: default_stagnation_kill_threshold(),
            min_species_protected: default_min_species_protected(),
        }
    }
}
