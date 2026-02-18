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

# Tournament configuration
var use_tournament: bool = true  # Use tournament system instead of random opponents
var tournament_mode: String = "round_robin"  # "round_robin" or "swiss"
var tournament_opponents: int = 4  # Number of opponents each individual plays against
var tournament_results: Dictionary = {}  # Track wins/losses/draws for tournament scoring

var _current_white_idx: int = 0
var _current_game_idx: int = 0
var _generation_in_progress: bool = false
var _tournament_pairings: Dictionary = {}  # Pre-computed pairings for current generation

func _init(p_evolution = null, p_games_per: int = 3, p_max_moves: int = 150) -> void:
	if p_evolution:
		evolution = p_evolution
	else:
		evolution = ChessEvolutionScript.new()
	games_per_individual = p_games_per
	max_moves_per_game = p_max_moves
	metrics_logger = MetricsLogger.new()
	
	# If using tournament mode, games_per_individual is ignored in favor of tournament_opponents
	if use_tournament:
		games_per_individual = tournament_opponents


func _generate_round_robin_pairings(population_size: int) -> Dictionary:
	## Generate round-robin pairings where each individual plays against opponents
	## from different fitness quintiles to ensure diverse matchups.
	var pairings := {}
	
	# Initialize pairings for each individual
	for i in population_size:
		pairings[i] = []
	
	# Create fitness-based groups (quintiles)
	var sorted_indices := []
	for i in population_size:
		sorted_indices.append(i)
	
	# Sort by fitness if we have previous generation data
	if evolution.generation > 0:
		var fitness_arr = evolution.white_fitness
		sorted_indices.sort_custom(func(a, b): return fitness_arr[a] > fitness_arr[b])
	else:
		# Random shuffle for first generation
		sorted_indices.shuffle()
	
	# Divide into quintiles
	var group_size := max(1, population_size / 5)
	var groups := []
	for i in range(5):
		var group := []
		var start := int(i * group_size)
		var end := int(min((i + 1) * group_size, population_size))
		for j in range(start, end):
			if j < sorted_indices.size():
				group.append(sorted_indices[j])
		if not group.is_empty():
			groups.append(group)
	
	# For each individual, select opponents from different quintiles
	for i in population_size:
		var opponents_needed := tournament_opponents
		var selected_opponents := []
		
		# Try to get one opponent from each quintile
		var quintile_idx := 0
		while selected_opponents.size() < opponents_needed and quintile_idx < groups.size():
			var group = groups[quintile_idx]
			# Find an opponent in this quintile that isn't self
			var candidates := group.filter(func(idx): return idx != i and idx not in selected_opponents)
			if not candidates.is_empty():
				selected_opponents.append(candidates[randi() % candidates.size()])
			quintile_idx += 1
		
		# If we need more opponents, select randomly from remaining population
		while selected_opponents.size() < opponents_needed:
			var opp := randi() % population_size
			if opp != i and opp not in selected_opponents:
				selected_opponents.append(opp)
		
		pairings[i] = selected_opponents
	
	return pairings


func _generate_swiss_pairings(population_size: int, round_num: int) -> Dictionary:
	## Generate Swiss-system pairings based on current tournament standings.
	## In round 1, pair randomly. In subsequent rounds, pair by score.
	var pairings := {}
	
	# Initialize pairings
	for i in population_size:
		pairings[i] = []
	
	if round_num == 0 or tournament_results.is_empty():
		# First round: random pairings
		var indices := []
		for i in population_size:
			indices.append(i)
		indices.shuffle()
		
		# Pair adjacent individuals
		for i in range(0, population_size - 1, 2):
			pairings[indices[i]].append(indices[i + 1])
			pairings[indices[i + 1]].append(indices[i])
	else:
		# Subsequent rounds: pair by score
		var scores := _calculate_tournament_scores()
		var sorted_indices := []
		for i in population_size:
			sorted_indices.append(i)
		sorted_indices.sort_custom(func(a, b): return scores[a] > scores[b])
		
		# Pair individuals with similar scores
		var paired := {}
		for i in sorted_indices:
			if i in paired:
				continue
			
			# Find best unpaired opponent with similar score
			var best_opp := -1
			for j in sorted_indices:
				if j != i and j not in paired and j not in pairings.get(i, []):
					best_opp = j
					break
			
			if best_opp != -1:
				pairings[i].append(best_opp)
				pairings[best_opp].append(i)
				paired[i] = true
				paired[best_opp] = true
	
	return pairings


func _calculate_tournament_scores() -> Dictionary:
	## Calculate tournament scores from results (1 point for win, 0.5 for draw, 0 for loss).
	var scores := {}
	
	for i in evolution.population_size:
		scores[i] = 0.0
	
	for key in tournament_results:
		var parts := key.split("_")
		if parts.size() == 2:
			var idx := int(parts[0])
			var result := tournament_results[key]
			
			if result == 1:  # Win
				scores[idx] += 1.0
			elif result == 0:  # Draw
				scores[idx] += 0.5
			# Loss gives 0 points
	
	return scores


func _update_fitness_from_tournament() -> void:
	## Update fitness based on tournament results instead of accumulated game fitness.
	var white_scores := _calculate_tournament_scores()
	var black_scores := {}
	
	# Calculate black tournament scores
	for i in evolution.population_size:
		black_scores[i] = 0.0
	
	for key in tournament_results:
		var parts := key.split("_")
		if parts.size() == 2 and parts[1] == "black":
			var idx := int(parts[0])
			var result := tournament_results[key]
			
			if result == 1:  # Win
				black_scores[idx] += 1.0
			elif result == 0:  # Draw  
				black_scores[idx] += 0.5
	
	# Update fitness arrays with tournament scores
	for i in evolution.population_size:
		# Tournament score becomes base fitness
		var white_tournament_score := white_scores.get(i, 0.0)
		var black_tournament_score := black_scores.get(i, 0.0)
		
		# Add small bonus based on material/position from accumulated fitness
		var white_bonus := evolution.white_fitness[i] * 0.1  # 10% weight for material/position
		var black_bonus := evolution.black_fitness[i] * 0.1
		
		evolution.set_fitness(0, i, white_tournament_score + white_bonus)
		evolution.set_fitness(1, i, black_tournament_score + black_bonus)


func run_generation() -> void:
	## Run all games for one generation, evaluate fitness, and evolve.
	if use_tournament:
		# Clear tournament results for new generation
		tournament_results.clear()
		_tournament_pairings.clear()
		
		# Generate pairings based on tournament mode
		if tournament_mode == "swiss":
			# Run multiple Swiss rounds
			var swiss_rounds := mini(tournament_opponents, evolution.population_size - 1)
			for round_idx in swiss_rounds:
				var round_pairings := _generate_swiss_pairings(evolution.population_size, round_idx)
				# Play all games in this round
				for w_idx in round_pairings:
					for b_idx in round_pairings[w_idx]:
						if w_idx < b_idx:  # Avoid duplicate games
							var result = _play_game(w_idx, b_idx)
							_record_tournament_result(w_idx, b_idx, result)
							game_complete.emit(w_idx, b_idx, result.result)
							total_games_played += 1
		else:
			# Round-robin tournament
			_tournament_pairings = _generate_round_robin_pairings(evolution.population_size)
			# Play all tournament games
			for w_idx in _tournament_pairings:
				for b_idx in _tournament_pairings[w_idx]:
					var result = _play_game(w_idx, b_idx)
					_record_tournament_result(w_idx, b_idx, result)
					game_complete.emit(w_idx, b_idx, result.result)
					total_games_played += 1
		
		# Update fitness based on tournament results
		_update_fitness_from_tournament()
	else:
		# Original random opponent selection
		var games_this_gen := _run_all_games()
		total_games_played += games_this_gen
	
	evolution.evolve()
	var stats := get_stats()
	metrics_logger.write_metrics(stats)
	training_step_complete.emit(evolution.generation, stats)


func _record_tournament_result(white_idx: int, black_idx: int, game_result) -> void:
	## Record tournament result for scoring.
	if game_result.result == 2:  # Draw
		tournament_results[str(white_idx) + "_white"] = 0
		tournament_results[str(black_idx) + "_black"] = 0
	elif game_result.result == 1:  # White wins
		tournament_results[str(white_idx) + "_white"] = 1
		tournament_results[str(black_idx) + "_black"] = -1
	else:  # Black wins
		tournament_results[str(white_idx) + "_white"] = -1
		tournament_results[str(black_idx) + "_black"] = 1


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
		
		if use_tournament:
			# Clear tournament data and generate pairings
			tournament_results.clear()
			_tournament_pairings.clear()
			if tournament_mode == "round_robin":
				_tournament_pairings = _generate_round_robin_pairings(evolution.population_size)

	if use_tournament:
		# Tournament mode: play pre-determined pairings
		if tournament_mode == "swiss":
			# Swiss system needs special handling for incremental play
			# For simplicity, we'll fall back to batch processing in run_generation
			push_warning("Swiss tournament mode not supported in incremental mode")
			return false
		else:
			# Round-robin: play next game from pairings
			var found_game := false
			var w_idx := _current_white_idx
			var b_idx := -1
			
			while w_idx < evolution.population_size and not found_game:
				var opponents = _tournament_pairings.get(w_idx, [])
				if _current_game_idx < opponents.size():
					b_idx = opponents[_current_game_idx]
					found_game = true
				else:
					w_idx += 1
					_current_game_idx = 0
					_current_white_idx = w_idx
			
			if found_game:
				var result = _play_game(w_idx, b_idx)
				_record_tournament_result(w_idx, b_idx, result)
				game_complete.emit(w_idx, b_idx, result.result)
				total_games_played += 1
				
				# Advance to next game
				_current_game_idx += 1
			else:
				# All games played, update fitness and evolve
				_update_fitness_from_tournament()
				evolution.evolve()
				var stats := get_stats()
				metrics_logger.write_metrics(stats)
				training_step_complete.emit(evolution.generation, stats)
				_generation_in_progress = false
				return true
	else:
		# Original random opponent selection
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
		if use_tournament:
			# In tournament mode, accumulate raw fitness for bonus calculation
			evolution.set_fitness(0, white_idx, w_prev + w_fitness)
		else:
			evolution.set_fitness(0, white_idx, w_prev + w_fitness / games_per_individual)
	
	if not black_is_hof:
		var b_prev: float = evolution.black_fitness[black_idx]
		if use_tournament:
			# In tournament mode, accumulate raw fitness for bonus calculation
			evolution.set_fitness(1, black_idx, b_prev + b_fitness)
		else:
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
