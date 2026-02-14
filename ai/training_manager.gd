extends RefCounted
class_name TrainingManager

## Orchestrates coevolutionary training: runs games between white and black populations,
## evaluates fitness, and triggers evolution.

signal training_step_complete(generation: int)
signal game_complete(white_idx: int, black_idx: int, result: int)

var evolution: ChessEvolution
var games_per_individual: int = 3  # Each individual plays N games per generation
var max_moves_per_game: int = 150
var current_games: Array = []  # Array of active GameState dicts


func _init(p_evolution: ChessEvolution = null, p_games_per: int = 3, p_max_moves: int = 150) -> void:
	if p_evolution:
		evolution = p_evolution
	else:
		evolution = ChessEvolution.new()
	games_per_individual = p_games_per
	max_moves_per_game = p_max_moves


func run_generation() -> void:
	## Run all games for one generation, evaluate fitness, and evolve.
	_run_all_games()
	evolution.evolve()
	training_step_complete.emit(evolution.generation)


func _run_all_games() -> void:
	## Each white individual plays against `games_per_individual` random black opponents.
	for w_idx in evolution.population_size:
		for _g in games_per_individual:
			var b_idx := randi() % evolution.population_size
			var result := _play_game(w_idx, b_idx)
			game_complete.emit(w_idx, b_idx, result.result)


func _play_game(white_idx: int, black_idx: int) -> BoardState:
	## Play a single game between two networks, return final board state.
	var state := BoardState.new()
	state.setup_initial()

	var white_net = evolution.get_network(0, white_idx)
	var black_net = evolution.get_network(1, black_idx)

	var move_count := 0
	while not state.is_game_over and move_count < max_moves_per_game:
		var net = white_net if state.side_to_move == 0 else black_net
		var inputs := ChessEncoder.encode_board(state)
		var outputs := net.forward(inputs)
		var legal_moves := state.generate_legal_moves()

		if legal_moves.is_empty():
			break

		var chosen := ChessEncoder.decode_move(outputs, legal_moves)
		state.make_move(chosen)
		move_count += 1

	# Force draw if max moves reached
	if not state.is_game_over:
		state.is_game_over = true
		state.result = 2

	# Evaluate fitness for both players
	var w_fitness := ChessFitness.evaluate(state, 0, move_count)
	var b_fitness := ChessFitness.evaluate(state, 1, move_count)

	# Accumulate fitness (averaged over games)
	var w_prev: float = evolution.white_fitness[white_idx]
	var b_prev: float = evolution.black_fitness[black_idx]
	evolution.set_fitness(0, white_idx, w_prev + w_fitness / games_per_individual)
	evolution.set_fitness(1, black_idx, b_prev + b_fitness / games_per_individual)

	return state


func get_stats() -> Dictionary:
	return {
		"generation": evolution.generation,
		"white_best": evolution.best_white_fitness,
		"black_best": evolution.best_black_fitness,
		"white_avg": evolution.get_avg_fitness(0),
		"black_avg": evolution.get_avg_fitness(1),
		"population_size": evolution.population_size,
	}
