class_name TrainingManager
extends RefCounted

## Orchestrates coevolutionary training: runs games between white and black populations,
## evaluates fitness, and triggers evolution.

signal training_step_complete(generation: int, stats: Dictionary)
signal game_complete(white_idx: int, black_idx: int, result: int)

const MetricsLogger = preload("res://ai/metrics_logger.gd")
const ChessEvolutionScript = preload("res://ai/evolution.gd")
const BoardStateScript = preload("res://chess/board_state.gd")
const ChessEncoderScript = preload("res://chess/encoder.gd")
const ChessFitnessScript = preload("res://ai/fitness.gd")
const GameRecorderScript = preload("res://ai/game_recorder.gd")
const MinimaxPlayerScript = preload("res://ai/minimax_player.gd")

var evolution
var games_per_individual: int = 3  # Each individual plays N games per generation
var max_moves_per_game: int = 150
var current_games: Array = []  # Array of active GameState dicts
var total_games_played: int = 0
var metrics_logger: MetricsLogger
var last_game_state = null  # Store most recent game for visualization
var last_game_history: Array = []  # Array of board states showing each move
var game_recorder: GameRecorderScript = null  # Optional: records full games for replay
var record_replays: bool = false  # Enable to save every game as a replay file

# Minimax search configuration
var use_minimax: bool = true  # Use minimax search instead of direct network output
var minimax_depth: int = 2    # Search depth for minimax (2-3 recommended)

# Hall of Fame configuration
var hall_of_fame_ratio: float = 0.5  # Ratio of games against Hall of Fame opponents (0.0-1.0)

var _current_white_idx: int = 0
var _current_game_idx: int = 0
var _generation_in_progress: bool = false

func _init(p_evolution = null, p_games_per: int = 3, p_max_moves: int = 150) -> void:
	if p_evolution:
		evolution = p_evolution
	else:
		evolution = ChessEvolutionScript.new()
	games_per_individual = p_games_per
	max_moves_per_game = p_max_moves
	metrics_logger = MetricsLogger.new()


func run_generation() -> void:
	## Run all games for one generation, evaluate fitness, and evolve.
	var games_this_gen := _run_all_games()
	evolution.evolve()
	total_games_played += games_this_gen
	var stats := get_stats()
	metrics_logger.write_metrics(stats)
	training_step_complete.emit(evolution.generation, stats)


func run_one_game_step() -> bool:
	## Run one game incrementally. Returns true if generation is complete.
	if not _generation_in_progress:
		# Start new generation
		_current_white_idx = 0
		_current_game_idx = 0
		_generation_in_progress = true
		# Reset fitness
		evolution.white_fitness.fill(0.0)
		evolution.black_fitness.fill(0.0)

	# Play one game (select opponent from population or Hall of Fame)
	var use_hof := false
	var b_idx: int = -1
	
	# Decide whether to use Hall of Fame opponent
	if evolution.has_hall_of_fame(1) and randf() < hall_of_fame_ratio:
		use_hof = true
		b_idx = -1  # Special index for Hall of Fame
	else:
		b_idx = randi() % int(evolution.population_size)
	
	var result = _play_game_with_hof(_current_white_idx, b_idx, use_hof)
	game_complete.emit(_current_white_idx, b_idx, result.result)
	total_games_played += 1

	# Advance to next game
	_current_game_idx += 1
	if _current_game_idx >= games_per_individual:
		_current_game_idx = 0
		_current_white_idx += 1

	# Check if generation complete
	if _current_white_idx >= evolution.population_size:
		evolution.evolve()
		var stats := get_stats()
		metrics_logger.write_metrics(stats)
		training_step_complete.emit(evolution.generation, stats)
		_generation_in_progress = false
		return true

	return false


func _run_all_games() -> int:
	## Each white individual plays against `games_per_individual` opponents.
	## Opponents are selected from current population or Hall of Fame based on ratio.
	var games_played := 0
	for w_idx in evolution.population_size:
		for _g in games_per_individual:
			var use_hof := false
			var b_idx: int = -1
			
			# Decide whether to use Hall of Fame opponent
			if evolution.has_hall_of_fame(1) and randf() < hall_of_fame_ratio:
				use_hof = true
				b_idx = -1  # Special index for Hall of Fame
			else:
				b_idx = randi() % int(evolution.population_size)
			
			var result = _play_game_with_hof(w_idx, b_idx, use_hof)
			game_complete.emit(w_idx, b_idx, result.result)
			games_played += 1
	
	# Also evaluate black population as primary players against white opponents
	# This ensures both populations benefit from Hall of Fame diversity
	for b_idx in evolution.population_size:
		for _g in games_per_individual:
			var use_hof := false
			var w_idx: int = -1
			
			# Decide whether to use Hall of Fame opponent
			if evolution.has_hall_of_fame(0) and randf() < hall_of_fame_ratio:
				use_hof = true
				w_idx = -1  # Special index for Hall of Fame
			else:
				w_idx = randi() % int(evolution.population_size)
			
			# Note: we pass true for first parameter to indicate white is from HoF
			var result = _play_game_with_hof(w_idx, b_idx, use_hof)
			game_complete.emit(w_idx, b_idx, result.result)
			games_played += 1
	
	return games_played


func _play_game_with_hof(white_idx: int, black_idx: int, idx_is_hof: bool = false):
	## Play a game where one player might be from Hall of Fame.
	## idx_is_hof indicates whether the special index (-1) represents Hall of Fame.
	var white_is_hof := false
	var black_is_hof := false
	
	if white_idx == -1 and idx_is_hof:
		white_is_hof = true
	elif black_idx == -1 and idx_is_hof:
		black_is_hof = true
	
	return _play_game(white_idx, black_idx, white_is_hof, black_is_hof)

func _play_game(white_idx: int, black_idx: int, white_is_hof: bool = false, black_is_hof: bool = false):
	## Play a single game between two networks, return final board state.
	var state := BoardStateScript.new()
	state.setup_initial()

	# Track game history for visualization (every 5th move to reduce memory)
	var game_history: Array = []
	game_history.append(state.clone())

	# Get networks (either from current population or Hall of Fame)
	var white_net = null
	var black_net = null
	
	if white_is_hof:
		white_net = evolution.get_hall_of_fame_opponent(0)
		if white_net == null:
			push_error("Hall of Fame requested but empty for white")
			return {"result": 2}  # Return draw on error
	else:
		white_net = evolution.get_network(0, white_idx)
	
	if black_is_hof:
		black_net = evolution.get_hall_of_fame_opponent(1)
		if black_net == null:
			push_error("Hall of Fame requested but empty for black")
			return {"result": 2}  # Return draw on error
	else:
		black_net = evolution.get_network(1, black_idx)
	
	# Create minimax players if enabled
	var white_player = null
	var black_player = null
	if use_minimax:
		white_player = MinimaxPlayerScript.new(white_net, minimax_depth)
		black_player = MinimaxPlayerScript.new(black_net, minimax_depth)

	# Set up game recorder if enabled
	var recorder: GameRecorderScript = null
	if record_replays:
		recorder = GameRecorderScript.new()
		recorder.start_recording({
			"generation": evolution.generation,
			"white_id": white_idx,
			"black_id": black_idx,
		})

	var move_count := 0
	while not state.is_game_over and move_count < max_moves_per_game:
		var chosen: Vector2i
		
		var legal_moves := state.generate_legal_moves()
		if legal_moves.is_empty():
			break
		
		if use_minimax:
			# Use minimax search to choose move
			var player = white_player if state.side_to_move == 0 else black_player
			chosen = player.choose_move(state)
		else:
			# Use direct network output (original behavior)
			var net = white_net if state.side_to_move == 0 else black_net
			var inputs: PackedFloat32Array = ChessEncoderScript.encode_board(state)
			var outputs: PackedFloat32Array = net.forward(inputs)
			chosen = ChessEncoderScript.decode_move(outputs, legal_moves)
		
		if chosen.x == -1 or chosen.y == -1:
			break  # Invalid move
		
		state.make_move(chosen)
		move_count += 1

		# Record every move for replay
		if recorder:
			recorder.record_move(chosen, state)

		# Save every 5th move for visualization
		if move_count % 5 == 0:
			game_history.append(state.clone())

	# Force draw if max moves reached
	if not state.is_game_over:
		state.is_game_over = true
		state.result = 2

	# Evaluate fitness for both players
	var w_fitness: float = ChessFitnessScript.evaluate(state, 0, move_count)
	var b_fitness: float = ChessFitnessScript.evaluate(state, 1, move_count)

	# Accumulate fitness (averaged over games)
	# Only update fitness for current population members, not Hall of Fame
	if not white_is_hof:
		var w_prev: float = evolution.white_fitness[white_idx]
		evolution.set_fitness(0, white_idx, w_prev + w_fitness / games_per_individual)
	
	if not black_is_hof:
		var b_prev: float = evolution.black_fitness[black_idx]
		evolution.set_fitness(1, black_idx, b_prev + b_fitness / games_per_individual)

	# Add final state to history
	game_history.append(state.clone())

	# Save replay file if recording
	if recorder:
		recorder.stop_recording(state.result)
		recorder.save_to_file()
		game_recorder = recorder

	# Store for visualization
	last_game_state = state
	last_game_history = game_history

	return state


func get_stats() -> Dictionary:
	var white_best: float = evolution.best_white_fitness
	var black_best: float = evolution.best_black_fitness
	var white_avg: float = evolution.get_avg_fitness(0)
	var black_avg: float = evolution.get_avg_fitness(1)
	return {
		"generation": evolution.generation,
		"white_best": white_best,
		"black_best": black_best,
		"white_avg": white_avg,
		"black_avg": black_avg,
		"best_fitness": max(white_best, black_best),
		"avg_fitness": (white_avg + black_avg) * 0.5,
		"population_size": evolution.population_size,
		"games_played": total_games_played,
		"games_per_generation": evolution.population_size * games_per_individual,
	}
