class_name NeatEvolution
extends RefCounted

## NEAT evolution manager for a single population.

signal generation_complete(
        generation: int, best_fitness: float, avg_fitness: float, min_fitness: float)

var config: NeatConfig
var innovation_tracker: NeatInnovation
var species_list: Array = []
var population: Array = []
var population_size: int = 0
var generation: int = 0

var best_fitness: float = -INF
var best_genome: NeatGenome = null
var all_time_best_fitness: float = -INF
var all_time_best_genome: NeatGenome = null

var seed_genome: NeatGenome = null
var _next_species_id: int = 0


func _init(p_config: NeatConfig, p_seed_genome: NeatGenome = null) -> void:
    config = p_config
    population_size = config.population_size
    var node_count := config.input_count + config.output_count + int(config.use_bias)
    innovation_tracker = NeatInnovation.new(node_count)
    seed_genome = p_seed_genome
    if seed_genome:
        innovation_tracker.seed_from_genome(seed_genome)
    _initialize_population()


func _initialize_population() -> void:
    population.clear()
    for i in config.population_size:
        var genome: NeatGenome
        if seed_genome:
            genome = NeatGenome.create_from_topology(seed_genome, config, innovation_tracker)
        else:
            genome = NeatGenome.create(config, innovation_tracker)
            genome.create_basic()
        population.append(genome)
    population_size = config.population_size
    generation = 0


func get_individual(index: int) -> NeatGenome:
    return population[index]


func get_network(index: int) -> NeatNetwork:
    return NeatNetwork.from_genome(population[index])


func set_fitness(index: int, fitness: float) -> void:
    population[index].fitness = fitness


func evolve() -> void:
    var spec_result: Dictionary = NeatSpecies.speciate(
            population, species_list, config, _get_next_species_id())
    species_list = spec_result.species
    _set_next_species_id(spec_result.next_id)

    if species_list.is_empty():
        _initialize_population()
        return

    var total_adjusted: float = 0.0
    var gen_best_fitness: float = -INF
    var gen_best_genome: NeatGenome = null

    for species in species_list:
        species.calculate_adjusted_fitness()
        species.update_best_fitness()
        var sp_best = species.get_best_genome()
        if sp_best and sp_best.fitness > gen_best_fitness:
            gen_best_fitness = sp_best.fitness
            gen_best_genome = sp_best
        total_adjusted += species.get_total_adjusted_fitness()

    best_fitness = gen_best_fitness
    best_genome = gen_best_genome.copy() if gen_best_genome else null

    if best_fitness > all_time_best_fitness:
        all_time_best_fitness = best_fitness
        all_time_best_genome = best_genome.copy() if best_genome else null

    _cull_stagnant_species()

    if species_list.is_empty():
        _initialize_population()
        return

    var new_population: Array = []

    for species in species_list:
        var sp_adjusted: float = species.get_total_adjusted_fitness()
        var offspring_count: int
        if total_adjusted > 0:
            offspring_count = int(round(sp_adjusted / total_adjusted * config.population_size))
        else:
            offspring_count = int(ceil(float(config.population_size) / species_list.size()))

        offspring_count = maxi(offspring_count, 1)

        var sorted_members: Array = species.get_sorted_members()
        if sorted_members.is_empty():
            continue

        var elite_count: int = maxi(1, int(sorted_members.size() * config.elite_fraction))
        for i in mini(elite_count, offspring_count):
            new_population.append(sorted_members[i].copy())

        var pool_size: int = maxi(1, int(sorted_members.size() * config.survival_fraction))
        var pool: Array = sorted_members.slice(0, pool_size)

        var remaining: int = offspring_count - mini(elite_count, offspring_count)
        for i in remaining:
            var child: NeatGenome
            if randf() < config.crossover_rate and pool.size() >= 2:
                var parent_a: NeatGenome = pool[randi() % pool.size()]
                var parent_b: NeatGenome
                if randf() < config.interspecies_crossover_rate and species_list.size() > 1:
                    var other_species = species_list[randi() % species_list.size()]
                    if not other_species.members.is_empty():
                        parent_b = other_species.members[randi() % other_species.members.size()]
                    else:
                        parent_b = pool[randi() % pool.size()]
                else:
                    parent_b = pool[randi() % pool.size()]
                child = NeatGenome.crossover(parent_a, parent_b)
            else:
                var parent: NeatGenome = pool[randi() % pool.size()]
                child = parent.copy()

            child.mutate(config)
            new_population.append(child)

    var avg_fitness: float = 0.0
    var min_fitness: float = INF
    for genome in population:
        avg_fitness += genome.fitness
        min_fitness = minf(min_fitness, genome.fitness)
    avg_fitness /= population.size() if not population.is_empty() else 1.0
    if min_fitness == INF:
        min_fitness = 0.0

    while new_population.size() > config.population_size:
        new_population.pop_back()
    while new_population.size() < config.population_size:
        var src_idx: int = randi() % population.size()
        var filler = population[src_idx].copy()
        filler.mutate(config)
        new_population.append(filler)

    population = new_population
    population_size = config.population_size
    generation += 1

    innovation_tracker.reset_generation_cache()
    NeatSpecies.adjust_compatibility_threshold(species_list, config)

    generation_complete.emit(generation, best_fitness, avg_fitness, min_fitness)


func _cull_stagnant_species() -> void:
    if species_list.size() <= config.min_species_protected:
        return

    var sorted_species := species_list.duplicate()
    sorted_species.sort_custom(func(a, b): return a.best_fitness_ever > b.best_fitness_ever)

    var surviving: Array = []
    for i in sorted_species.size():
        if i < config.min_species_protected:
            surviving.append(sorted_species[i])
        elif sorted_species[i].is_stagnant(config.stagnation_kill_threshold):
            continue
        else:
            surviving.append(sorted_species[i])

    species_list = surviving


func _get_next_species_id() -> int:
    return _next_species_id


func _set_next_species_id(value: int) -> void:
    _next_species_id = value
