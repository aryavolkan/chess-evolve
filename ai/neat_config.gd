extends RefCounted
class_name NeatConfig

## All NEAT hyperparameters in one place for easy tuning.

# ============================================================
# NETWORK ARCHITECTURE
# ============================================================

var input_count: int = 389
var output_count: int = 128
var use_bias: bool = false
var allow_recurrent: bool = false

# ============================================================
# COMPATIBILITY / SPECIATION
# ============================================================

var compatibility_threshold: float = 3.0
var c1_excess: float = 1.0
var c2_disjoint: float = 1.0
var c3_weight_diff: float = 0.4
var target_species_count: int = 4
var threshold_step: float = 0.3

# ============================================================
# MUTATION RATES
# ============================================================

var weight_mutate_rate: float = 0.8
var weight_perturb_rate: float = 0.9
var weight_perturb_strength: float = 0.3
var weight_reset_range: float = 2.0
var add_node_rate: float = 0.03
var add_connection_rate: float = 0.05
var disable_connection_rate: float = 0.01

# ============================================================
# REPRODUCTION
# ============================================================

var population_size: int = 50
var max_generations: int = 0
var elite_fraction: float = 0.1
var survival_fraction: float = 0.5
var interspecies_crossover_rate: float = 0.001
var crossover_rate: float = 0.75
var disabled_gene_inherit_rate: float = 0.75

# ============================================================
# STAGNATION
# ============================================================

var stagnation_threshold: int = 15
var stagnation_kill_threshold: int = 25
var min_species_protected: int = 2

# ============================================================
# INITIAL POPULATION
# ============================================================

var initial_connections_per_output: int = 10  ## Sparse initial topology (K inputs per output)

# ============================================================
# PARSIMONY (optional complexity penalty)
# ============================================================

var parsimony_coefficient: float = 0.0


func _init() -> void:
	pass


func duplicate() -> NeatConfig:
	var copy := NeatConfig.new()
	copy.input_count = input_count
	copy.output_count = output_count
	copy.use_bias = use_bias
	copy.allow_recurrent = allow_recurrent
	copy.compatibility_threshold = compatibility_threshold
	copy.c1_excess = c1_excess
	copy.c2_disjoint = c2_disjoint
	copy.c3_weight_diff = c3_weight_diff
	copy.target_species_count = target_species_count
	copy.threshold_step = threshold_step
	copy.weight_mutate_rate = weight_mutate_rate
	copy.weight_perturb_rate = weight_perturb_rate
	copy.weight_perturb_strength = weight_perturb_strength
	copy.weight_reset_range = weight_reset_range
	copy.add_node_rate = add_node_rate
	copy.add_connection_rate = add_connection_rate
	copy.disable_connection_rate = disable_connection_rate
	copy.population_size = population_size
	copy.max_generations = max_generations
	copy.elite_fraction = elite_fraction
	copy.survival_fraction = survival_fraction
	copy.interspecies_crossover_rate = interspecies_crossover_rate
	copy.crossover_rate = crossover_rate
	copy.disabled_gene_inherit_rate = disabled_gene_inherit_rate
	copy.stagnation_threshold = stagnation_threshold
	copy.stagnation_kill_threshold = stagnation_kill_threshold
	copy.min_species_protected = min_species_protected
	copy.initial_connections_per_output = initial_connections_per_output
	copy.parsimony_coefficient = parsimony_coefficient
	return copy
