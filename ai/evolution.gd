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

var input_size: int
var hidden_size: int
var output_size: int

var best_white_fitness: float = 0.0
var best_black_fitness: float = 0.0
var all_time_best_white = null
var all_time_best_black = null
var all_time_best_white_fitness: float = -INF
var all_time_best_black_fitness: float = -INF


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


func set_fitness(color: int, index: int, fitness: float) -> void:
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
	var f := white_fitness if color == 0 else black_fitness
	var total := 0.0
	for v in f: total += v
	return total / f.size() if f.size() > 0 else 0.0
