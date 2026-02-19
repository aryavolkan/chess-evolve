class_name ChessNeatEvolution
extends RefCounted

## Coevolution wrapper around NEAT populations for white and black.

signal generation_complete(gen: int, white_best: float, black_best: float)

const NeatEvolutionScript = preload("res://ai/neat_evolution.gd")
const NeatConfigScript = preload("res://ai/neat_config.gd")
const NeatNetworkScript = preload("res://ai/neat_network.gd")

var population_size: int
var white_evolution: NeatEvolution
var black_evolution: NeatEvolution
var white_fitness: PackedFloat32Array
var black_fitness: PackedFloat32Array
var generation: int = 0

var best_white_fitness: float = 0.0
var best_black_fitness: float = 0.0
var all_time_best_white = null
var all_time_best_black = null
var all_time_best_white_fitness: float = -INF
var all_time_best_black_fitness: float = -INF

var last_white_avg: float = 0.0
var last_black_avg: float = 0.0
var _fitness_cleared: bool = true

var hall_of_fame: Array = []
var black_hall_of_fame: Array = []
const HALL_OF_FAME_SIZE := 20


func _init(p_pop_size = 50, p_config: NeatConfig = null) -> void:
	var config: NeatConfig
	if p_pop_size is NeatConfig:
		config = p_pop_size
		population_size = config.population_size
	else:
		population_size = int(p_pop_size)
		config = p_config if p_config else NeatConfigScript.new()
		config.population_size = population_size
	white_evolution = NeatEvolutionScript.new(config.duplicate())
	black_evolution = NeatEvolutionScript.new(config.duplicate())

	white_fitness.resize(population_size)
	black_fitness.resize(population_size)
	_reset_fitness()


func _reset_fitness() -> void:
	white_fitness.fill(0.0)
	black_fitness.fill(0.0)
	_fitness_cleared = true


func set_fitness(color: int, index: int, fitness: float) -> void:
	_fitness_cleared = false
	if color == 0:
		white_fitness[index] = fitness
		white_evolution.set_fitness(index, fitness)
	else:
		black_fitness[index] = fitness
		black_evolution.set_fitness(index, fitness)


func get_network(color: int, index: int):
	return white_evolution.get_network(index) if color == 0 else black_evolution.get_network(index)


func get_best_fitness() -> float:
	return maxf(best_white_fitness, best_black_fitness)


func evolve_one_generation() -> void:
	var denom := float(maxi(population_size, 1))
	for i in population_size:
		var score := float(generation) + float(i) / denom
		set_fitness(0, i, score)
		set_fitness(1, i, score)
	evolve()


func evolve() -> void:
	white_evolution.evolve()
	black_evolution.evolve()

	generation = white_evolution.generation

	var w_best_idx := _best_index(white_fitness)
	var b_best_idx := _best_index(black_fitness)
	best_white_fitness = white_fitness[w_best_idx]
	best_black_fitness = black_fitness[b_best_idx]

	if all_time_best_white == null or best_white_fitness > all_time_best_white_fitness:
		all_time_best_white = white_evolution.population[w_best_idx].copy()
		all_time_best_white_fitness = best_white_fitness
	if all_time_best_black == null or best_black_fitness > all_time_best_black_fitness:
		all_time_best_black = black_evolution.population[b_best_idx].copy()
		all_time_best_black_fitness = best_black_fitness

	_update_hall_of_fame(white_evolution.population[w_best_idx], best_white_fitness, true)
	_update_hall_of_fame(black_evolution.population[b_best_idx], best_black_fitness, false)

	last_white_avg = get_avg_fitness(0)
	last_black_avg = get_avg_fitness(1)

	_reset_fitness()
	generation_complete.emit(generation, best_white_fitness, best_black_fitness)


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
	for v in f:
		total += v
	return total / f.size() if f.size() > 0 else 0.0


func _update_hall_of_fame(individual: NeatGenome, fitness: float, is_white: bool) -> void:
	var hof := hall_of_fame if is_white else black_hall_of_fame
	var entry := {
		"genome": individual.copy(),
		"fitness": fitness,
		"generation": generation
	}
	
	hof.append(entry)
	hof.sort_custom(func(a, b): return a.fitness > b.fitness)
	if hof.size() > HALL_OF_FAME_SIZE:
		hof.resize(HALL_OF_FAME_SIZE)


func get_hall_of_fame_opponent(color: int):
	var hof := hall_of_fame if color == 0 else black_hall_of_fame
	if hof.is_empty():
		return null
	var idx := randi() % hof.size()
	return NeatNetworkScript.from_genome(hof[idx].genome)


func has_hall_of_fame(color: int) -> bool:
	var hof := hall_of_fame if color == 0 else black_hall_of_fame
	return not hof.is_empty()
