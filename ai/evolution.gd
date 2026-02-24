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

# Elo rating configuration for Hall of Fame
const ELO_DEFAULT: float = 1200.0    # Starting Elo for new HoF members
const ELO_K_FACTOR: float = 32.0   # K-factor for Elo updates (higher = faster changes)
const ELO_ELO_WEIGHTED: bool = true  # Whether to weight selection by Elo


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
    ## Keeps only the best HALL_OF_FAME_SIZE individuals, sorted by Elo (then fitness).
    var hof := hall_of_fame if is_white else black_hall_of_fame

    # Create entry with fitness score and Elo rating for sorting
    var entry := {
        "network": individual.clone(),  # Clone to freeze the weights
        "fitness": fitness,
        "elo": ELO_DEFAULT,
        "generation": generation,
        "games_played": 0,
        "games_won": 0,
        "games_drawn": 0,
        "games_lost": 0
    }

    # Add to hall of fame
    hof.append(entry)

    # Sort by Elo descending (then fitness as tiebreaker) and keep only the best
    hof.sort_custom(func(a, b):
        if a.elo != b.elo:
            return a.elo > b.elo
        return a.fitness > b.fitness
    )
    if hof.size() > HALL_OF_FAME_SIZE:
        hof.resize(HALL_OF_FAME_SIZE)


func update_hall_of_fame_elo(is_white: bool, opponent_idx: int, result: int) -> void:
    ## Update Elo ratings for a HoF member after a game.
    ## result: 1 = win, 0.5 = draw, 0 = loss (from the HoF member's perspective)
    var hof := hall_of_fame if is_white else black_hall_of_fame
    if opponent_idx < 0 or opponent_idx >= hof.size():
        return

    var entry: Dictionary = hof[opponent_idx]
    var expected := 1.0  # Assume playing against current generation ( Elo doesn't change)

    # Update Elo: new_elo = old_elo + K * (actual - expected)
    # For now, we just track stats; full Elo system would need opponent ratings
    var actual: float = result  # 1, 0.5, or 0

    # Update stats
    entry.games_played += 1
    if result >= 0.75:
        entry.games_won += 1
    elif result >= 0.25:
        entry.games_drawn += 1
    else:
        entry.games_lost += 1

    # Re-sort after stats update (Elo stays same for now since we're not tracking opponent rating)
    hof.sort_custom(func(a, b):
        if a.elo != b.elo:
            return a.elo > b.elo
        return a.fitness > b.fitness
    )


static func _elo_expected_score(rating_a: float, rating_b: float) -> float:
    ## Calculate expected score for player A against player B
    return 1.0 / (1.0 + pow(10.0, (rating_b - rating_a) / 400.0))


func get_hall_of_fame_opponent(color: int, use_elo_weight: bool = ELO_ELO_WEIGHTED):
    ## Get an opponent from the Hall of Fame.
    ## If use_elo_weight is true, weights selection by Elo (higher = more likely).
    ## Returns null if Hall of Fame is empty.
    var hof := hall_of_fame if color == 0 else black_hall_of_fame
    if hof.is_empty():
        return null

    if use_elo_weight and hof.size() > 1:
        # Weight by Elo using softmax-like approach
        var weights: Array = []
        var min_elo: float = INF
        for entry in hof:
            min_elo = minf(min_elo, entry.elo)

        # Convert Elo to weights (shift so lowest is ~1, then exponential)
        var total_weight: float = 0.0
        for entry in hof:
            var weight: float = exp((entry.elo - min_elo) / 200.0)  # Temperature = 200
            weights.append(weight)
            total_weight += weight

        # Weighted random selection
        var r: float = randf() * total_weight
        var cumsum: float = 0.0
        for i in range(hof.size()):
            cumsum += weights[i]
            if r <= cumsum:
                return hof[i].network

        # Fallback
        return hof[hof.size() - 1].network
    else:
        var idx := randi() % hof.size()
        return hof[idx].network


func get_hall_of_fame_stats(color: int) -> Dictionary:
    ## Get statistics about the Hall of Fame for reporting
    var hof := hall_of_fame if color == 0 else black_hall_of_fame
    if hof.is_empty():
        return {"size": 0, "avg_elo": 0.0, "top_elo": 0.0, "total_games": 0}

    var total_elo: float = 0.0
    var total_games: int = 0
    for entry in hof:
        total_elo += entry.elo
        total_games += entry.games_played

    return {
        "size": hof.size(),
        "avg_elo": total_elo / hof.size(),
        "top_elo": hof[0].elo,
        "total_games": total_games,
        "top_fitness": hof[0].fitness
    }


func has_hall_of_fame(color: int) -> bool:
    ## Check if Hall of Fame has any members for the given color.
    var hof := hall_of_fame if color == 0 else black_hall_of_fame
    return not hof.is_empty()


func seed_from_global_elite(elite_pool: Array) -> int:
    ## Seed the Hall of Fame with genomes from the global elite pool.
    ##
    ## Each entry in elite_pool must be a Dictionary with keys:
    ##   weights_ih, bias_h, weights_ho, bias_o  — Array[float]
    ##   fitness     — float
    ##   color       — "white" | "black"  (optional, defaults to both)
    ##   elo         — float              (optional, defaults to ELO_DEFAULT)
    ##
    ## Returns the number of genomes successfully seeded.
    var seeded := 0
    for entry in elite_pool:
        if not _is_valid_elite_entry(entry):
            continue

        # Build the network
        var e_input := int(entry.get("input_size", input_size))
        var e_hidden := int(entry.get("hidden_size", hidden_size))
        var e_output := int(entry.get("output_size", output_size))

        # Skip if architecture doesn't match (can't play in this run)
        if e_input != input_size or e_hidden != hidden_size or e_output != output_size:
            push_warning("GlobalElite: skipping genome with mismatched architecture (%d/%d/%d vs %d/%d/%d)" % [
                e_input, e_hidden, e_output, input_size, hidden_size, output_size
            ])
            continue

        var nn = NeuralNetworkScript.new(input_size, hidden_size, output_size, false)
        nn.load_from_dict(entry)

        var fitness := float(entry.get("fitness", 0.0))
        var elo := float(entry.get("elo", ELO_DEFAULT))
        var color: String = entry.get("color", "both")

        if color == "black":
            _seed_hall_of_fame(nn, fitness, elo, false)
        elif color == "white":
            _seed_hall_of_fame(nn, fitness, elo, true)
        else:
            # Unknown or "both": add to both sides
            _seed_hall_of_fame(nn, fitness, elo, true)
            _seed_hall_of_fame(nn.clone(), fitness, elo, false)

        seeded += 1

    if seeded > 0:
        print("GlobalElite: seeded %d genome(s) into hall of fame (white=%d, black=%d)" % [
            seeded, hall_of_fame.size(), black_hall_of_fame.size()
        ])
    return seeded


func _seed_hall_of_fame(network, fitness: float, elo: float, is_white: bool) -> void:
    ## Insert a pre-built network into the hall of fame.
    var hof := hall_of_fame if is_white else black_hall_of_fame
    hof.append({
        "network": network,
        "fitness": fitness,
        "elo": elo,
        "generation": -1,
        "games_played": 0,
        "games_won": 0,
        "games_drawn": 0,
        "games_lost": 0,
    })
    hof.sort_custom(func(a, b):
        if a.elo != b.elo:
            return a.elo > b.elo
        return a.fitness > b.fitness
    )
    if hof.size() > HALL_OF_FAME_SIZE:
        hof.resize(HALL_OF_FAME_SIZE)


static func _is_valid_elite_entry(entry) -> bool:
    if not entry is Dictionary:
        return false
    for key in ["weights_ih", "bias_h", "weights_ho", "bias_o", "fitness"]:
        if not entry.has(key):
            return false
    return true


func write_elite_contrib(worker_id: String) -> void:
    ## Serialize top hall-of-fame members to user://elite_contrib_{worker_id}.json
    ## so the Python layer can merge them into the global elite pool.
    var path := "user://elite_contrib_%s.json" % worker_id
    if worker_id.is_empty():
        path = "user://elite_contrib_default.json"

    var entries: Array = []

    for color_flag in [true, false]:
        var hof := hall_of_fame if color_flag else black_hall_of_fame
        var color_str := "white" if color_flag else "black"
        # Contribute up to 5 entries per color
        for i in mini(5, hof.size()):
            var hof_entry: Dictionary = hof[i]
            var genome: Dictionary = hof_entry["network"].to_dict()
            genome["fitness"] = hof_entry["fitness"]
            genome["elo"] = hof_entry["elo"]
            genome["color"] = color_str
            genome["generation"] = hof_entry["generation"]
            genome["source_run"] = worker_id
            entries.append(genome)

    var file := FileAccess.open(path, FileAccess.WRITE)
    if file == null:
        push_warning("GlobalElite: failed to write contrib file %s" % path)
        return

    var payload := {
        "elites": entries,
        "worker_id": worker_id,
        "updated_at": Time.get_unix_time_from_system(),
    }
    file.store_string(JSON.new().stringify(payload, "\t"))
    file.close()
    print("GlobalElite: wrote %d elite genome(s) to %s" % [entries.size(), path])
