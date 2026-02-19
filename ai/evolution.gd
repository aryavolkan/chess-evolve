class_name ChessEvolution
extends RefCounted

## Manages a population of chess-playing neural networks with coevolution.
## Two populations: white and black, evolved against each other.

signal generation_complete(gen: int, white_best: float, black_best: float)

var NeuralNetworkScript = preload("res://ai/neural_network.gd")

var population_size: int
var white_pop: Array = []
var black_pop: Array = []
var white_fitness: PackedFloat32Array
var black_fitness: PackedFloat32Array
var generation: int = 0

var elite_count: int
var mutation_rate: float
var mutation_strength: float
var crossover_rate: float

# Adaptive mutation configuration
var adaptive_mutation: bool = true
var mutation_rate_min: float = 0.05
var mutation_rate_max: float = 0.35
var mutation_strength_min: float = 0.05
var mutation_strength_max: float = 0.4
var stagnation_window: int = 5
var _best_history: Array = []

var input_size: int
var hidden_size: int
var output_size: int

var best_white_fitness: float = 0.0
var best_black_fitness: float = 0.0
var all_time_best_white = null
var all_time_best_black = null
var all_time_best_white_fitness: float = -INF
var all_time_best_black_fitness: float = -INF

var last_white_avg: float = 0.0
var last_black_avg: float = 0.0
var _fitness_cleared: bool = true

# Hall of Fame: Maintain a persistent collection of past best individuals
# to prevent cycling in coevolution and ensure diverse opponents
var hall_of_fame: Array = []        # Best white players from past generations
var black_hall_of_fame: Array = []  # Best black players from past generations
const HALL_OF_FAME_SIZE := 20      # Maximum number of individuals to keep


func _init(
	p_pop_size: int = 50,
	p_input_size: int = 389,
	p_hidden_size: int = 64,
	p_output_size: int = 128,
	p_elite_count: int = 5,
	p_mutation_rate: float = 0.15,
	p_mutation_strength: float = 0.2,
	p_crossover_rate: float = 0.7
) -> void:
	population_size = p_pop_size
	input_size = p_input_size
	hidden_size = p_hidden_size
	output_size = p_output_size
	elite_count = p_elite_count
	mutation_rate = p_mutation_rate
	mutation_strength = p_mutation_strength
	crossover_rate = p_crossover_rate

	white_fitness.resize(p_pop_size)
	black_fitness.resize(p_pop_size)
	_initialize_population()


func _initialize_population() -> void:
	white_pop.clear()
	black_pop.clear()
	for i in population_size:
		white_pop.append(NeuralNetworkScript.new(input_size, hidden_size, output_size))
		black_pop.append(NeuralNetworkScript.new(input_size, hidden_size, output_size))
	generation = 0
	_reset_fitness()


func _reset_fitness() -> void:
	white_fitness.fill(0.0)
	black_fitness.fill(0.0)
	_fitness_cleared = true


func set_fitness(color: int, index: int, fitness: float) -> void:
	_fitness_cleared = false
	if color == 0:
		white_fitness[index] = fitness
	else:
		black_fitness[index] = fitness


func get_network(color: int, index: int):
	return white_pop[index] if color == 0 else black_pop[index]


func evolve() -> void:
	## Evolve both populations independently based on their fitness.
	white_pop = _evolve_population(white_pop, white_fitness)
	black_pop = _evolve_population(black_pop, black_fitness)

	# Track bests
	var w_best_idx := _best_index(white_fitness)
	var b_best_idx := _best_index(black_fitness)
	best_white_fitness = white_fitness[w_best_idx]
	best_black_fitness = black_fitness[b_best_idx]

	if all_time_best_white == null or best_white_fitness > all_time_best_white_fitness:
		all_time_best_white = white_pop[w_best_idx].clone()
		all_time_best_white_fitness = best_white_fitness
	if all_time_best_black == null or best_black_fitness > all_time_best_black_fitness:
		all_time_best_black = black_pop[b_best_idx].clone()
		all_time_best_black_fitness = best_black_fitness

	# Add best individuals to Hall of Fame
	_update_hall_of_fame(white_pop[w_best_idx], best_white_fitness, true)
	_update_hall_of_fame(black_pop[b_best_idx], best_black_fitness, false)

	# Cache average fitness before clearing arrays
	last_white_avg = get_avg_fitness(0)
	last_black_avg = get_avg_fitness(1)

	if adaptive_mutation:
		_update_mutation_schedule(max(best_white_fitness, best_black_fitness))

	generation += 1
	_reset_fitness()
	generation_complete.emit(generation, best_white_fitness, best_black_fitness)


func _evolve_population(pop: Array, fitness: PackedFloat32Array) -> Array:
	var new_pop: Array = []

	# Sort by fitness descending
	var indices := []
	for i in pop.size():
		indices.append(i)
	indices.sort_custom(func(a, b): return fitness[a] > fitness[b])

	# Elitism
	for i in mini(elite_count, pop.size()):
		new_pop.append(pop[indices[i]].clone())

	# Fill rest with tournament selection + crossover/mutation
	while new_pop.size() < population_size:
		var parent_a = _tournament_select(pop, fitness)
		if randf() < crossover_rate:
			var parent_b = _tournament_select(pop, fitness)
			var child = parent_a.crossover_with(parent_b)
			child.mutate(mutation_rate, mutation_strength)
			new_pop.append(child)
		else:
			var child = parent_a.clone()
			child.mutate(mutation_rate, mutation_strength)
			new_pop.append(child)

	return new_pop


func _tournament_select(pop: Array, fitness: PackedFloat32Array, k: int = 3):
	var best_idx := randi() % pop.size()
	for _i in range(1, k):
		var idx := randi() % pop.size()
		if fitness[idx] > fitness[best_idx]:
			best_idx = idx
	return pop[best_idx]


func _best_index(fitness: PackedFloat32Array) -> int:
	var best := 0
	for i in range(1, fitness.size()):
		if fitness[i] > fitness[best]:
			best = i
	return best


func get_avg_fitness(color: int) -> float:
	if _fitness_cleared:
		return last_white_avg if color == 0 else last_black_avg
	var f := white_fitness if color == 0 else black_fitness
	var total := 0.0
	for v in f: total += v
	return total / f.size() if f.size() > 0 else 0.0


func _update_mutation_schedule(best_fitness: float) -> void:
	_best_history.append(best_fitness)
	if _best_history.size() > stagnation_window:
		_best_history.pop_front()

	var improved := false
	if _best_history.size() >= 2:
		improved = best_fitness > _best_history[0]

	if improved:
		mutation_rate = maxf(mutation_rate_min, mutation_rate * 0.9)
		mutation_strength = maxf(mutation_strength_min, mutation_strength * 0.9)
	else:
		# If stagnant, nudge mutation upward
		mutation_rate = minf(mutation_rate_max, mutation_rate * 1.1)
		mutation_strength = minf(mutation_strength_max, mutation_strength * 1.1)


func _update_hall_of_fame(individual, fitness: float, is_white: bool) -> void:
	## Add a high-performing individual to the Hall of Fame.
	## Keeps only the best HALL_OF_FAME_SIZE individuals, sorted by fitness.
	var hof := hall_of_fame if is_white else black_hall_of_fame
	
	# Create entry with fitness score for sorting
	var entry := {
		"network": individual.clone(),  # Clone to freeze the weights
		"fitness": fitness,
		"generation": generation
	}
	
	# Add to hall of fame
	hof.append(entry)
	
	# Sort by fitness descending and keep only the best
	hof.sort_custom(func(a, b): return a.fitness > b.fitness)
	if hof.size() > HALL_OF_FAME_SIZE:
		hof.resize(HALL_OF_FAME_SIZE)


func get_hall_of_fame_opponent(color: int):
	## Get a random opponent from the Hall of Fame.
	## Returns null if Hall of Fame is empty.
	var hof := hall_of_fame if color == 0 else black_hall_of_fame
	if hof.is_empty():
		return null
	
	var idx := randi() % hof.size()
	return hof[idx].network


func has_hall_of_fame(color: int) -> bool:
	## Check if Hall of Fame has any members for the given color.
	var hof := hall_of_fame if color == 0 else black_hall_of_fame
	return not hof.is_empty()
