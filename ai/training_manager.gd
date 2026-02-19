class_name TrainingManager
extends RefCounted

## Orchestrates coevolutionary training: runs games between white and black populations,
## evaluates fitness, and triggers evolution.

signal training_step_complete(generation: int, stats: Dictionary)
signal game_complete(white_idx: int, black_idx: int, result: int)

const MetricsLogger = preload("res://ai/metrics_logger.gd")
const ChessEvolutionScript = preload("res://ai/evolution.gd")
const ChessNeatEvolutionScript = preload("res://ai/chess_neat_evolution.gd")
const BoardStateScript = preload("res://chess/board_state.gd")
const ChessEncoderScript = preload("res://chess/encoder.gd")
const ChessFitnessScript = preload("res://ai/fitness.gd")
const GameRecorderScript = preload("res://ai/game_recorder.gd")
const MinimaxPlayerScript = preload("res://ai/minimax_player.gd")

var evolution
var use_neat: bool = false
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
var use_minimax: bool = false  # Use minimax search instead of direct network output
var minimax_depth: int = 2    # Search depth for minimax (2-3 recommended)

# Curriculum configuration
var use_curriculum: bool = true
var curriculum_stages: Array = [
	{
		"min_gen": 0,
		"max_gen": 2,
		"max_moves": 60,
		"weights": {
			"win_bonus": 6.0,
			"draw_bonus": 1.0,
			"material_weight": 1.2,
			"mobility_weight": 0.1,
			"king_safety_weight": 0.2,
			"move_count_bonus": 0.02
		}
	},
	{
		"min_gen": 3,
		"max_gen": 7,
		"max_moves": 100,
		"weights": {
			"win_bonus": 8.0,
			"draw_bonus": 2.0,
			"material_weight": 1.0,
			"mobility_weight": 0.08,
			"king_safety_weight": 0.35,
			"move_count_bonus": 0.015
		}
	},
	{
		"min_gen": 8,
		"max_gen": 999999,
		"max_moves": 150,
		"weights": {
			"win_bonus": 10.0,
			"draw_bonus": 3.0,
			"material_weight": 1.0,
			"mobility_weight": 0.05,
			"king_safety_weight": 0.5,
			"move_count_bonus": 0.01
		}
	}
]

# Move generation configuration
var use_bitboard_movegen: bool = false  # Use BitboardState for fast legal move generation

# Hall of Fame configuration
var hall_of_fame_ratio: float = 0.5  # Ratio of games against Hall of Fame opponents (0.0-1.0)

# Opponent sampling bias configuration
var use_fitness_weighted_sampling: bool = true
var opponent_sampling_temperature: float = 1.0  # Higher => flatter distribution

# Early termination (mercy rule)
var mercy_min_moves: int = 20
var mercy_material_threshold: float = 10.0  # End game if abs material diff exceeds this

# Encoder output cache (position -> outputs)
var use_encoder_cache: bool = true
var encoder_cache_max: int = 5000
var _encoder_cache: Dictionary = {}

# Rust batch simulation (optional)
var use_rust_batch_sim: bool = true
var _rust_batch_sim = null

# Tournament configuration
var use_tournament: bool = true  # Use tournament system instead of random opponents
var tournament_mode: String = "round_robin"  # "round_robin" or "swiss"
var tournament_opponents: int = 4  # Number of opponents each individual plays against
var tournament_results: Dictionary = {}  # Track wins/losses/draws for tournament scoring

var _current_white_idx: int = 0
var _current_game_idx: int = 0
var _generation_in_progress: bool = false
var _tournament_pairings: Dictionary = {}  # Pre-computed pairings for current generation

# New metrics tracking
var _generation_start_time: float = 0.0
var _white_wins: int = 0
var _white_draws: int = 0
var _white_losses: int = 0
var _black_wins: int = 0
var _black_draws: int = 0
var _black_losses: int = 0
var _total_game_moves: int = 0
var _games_this_generation: int = 0
var _white_material_total: float = 0.0
var _black_material_total: float = 0.0
var _white_tournament_scores: Array = []
var _black_tournament_scores: Array = []

func _init(p_evolution = null, p_games_per: int = 3, p_max_moves: int = 150, p_metrics_path: String = "", p_use_neat: bool = false) -> void:
	if p_evolution:
		evolution = p_evolution
		use_neat = p_use_neat or p_evolution is ChessNeatEvolutionScript
	else:
		use_neat = p_use_neat
		evolution = ChessNeatEvolutionScript.new() if use_neat else ChessEvolutionScript.new()
	games_per_individual = p_games_per
	max_moves_per_game = p_max_moves
	metrics_logger = MetricsLogger.new(p_metrics_path)
	
	if use_neat:
		use_rust_batch_sim = false
	
	# If using tournament mode, games_per_individual is ignored in favor of tournament_opponents


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
	var group_size: int = max(1, population_size / 5)
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
		var opponents_needed: int = mini(tournament_opponents, population_size - 1)
		var selected_opponents := []
		
		# Try to get one opponent from each quintile
		var quintile_idx := 0
		while selected_opponents.size() < opponents_needed and quintile_idx < groups.size():
			var group = groups[quintile_idx]
			# Find an opponent in this quintile that isn't self
			var candidates: Array = group.filter(func(idx): return idx != i and idx not in selected_opponents)
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
		var parts: PackedStringArray = key.split("_")
		if parts.size() == 2:
			var idx := int(parts[0])
			var result: int = tournament_results[key]
			
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
		var parts: PackedStringArray = key.split("_")
		if parts.size() == 2 and parts[1] == "black":
			var idx := int(parts[0])
			var result: int = tournament_results[key]
			
			if result == 1:  # Win
				black_scores[idx] += 1.0
			elif result == 0:  # Draw  
				black_scores[idx] += 0.5
	
	# Store tournament scores for metrics
	_white_tournament_scores.clear()
	_black_tournament_scores.clear()
	
	# Update fitness arrays with tournament scores
	for i in evolution.population_size:
		# Tournament score becomes base fitness
		var white_tournament_score: float = white_scores.get(i, 0.0)
		var black_tournament_score: float = black_scores.get(i, 0.0)
		
		# Store for metrics
		_white_tournament_scores.append(white_tournament_score)
		_black_tournament_scores.append(black_tournament_score)
		
		# Add small bonus based on material/position from accumulated fitness
		var white_bonus: float = evolution.white_fitness[i] * 0.1  # 10% weight for material/position
		var black_bonus: float = evolution.black_fitness[i] * 0.1
		
		evolution.set_fitness(0, i, white_tournament_score + white_bonus)
		evolution.set_fitness(1, i, black_tournament_score + black_bonus)


func _reset_generation_metrics() -> void:
	_generation_start_time = Time.get_unix_time_from_system()
	_white_wins = 0
	_white_draws = 0
	_white_losses = 0
	_black_wins = 0
	_black_draws = 0
	_black_losses = 0
	_total_game_moves = 0
	_games_this_generation = 0
	_white_material_total = 0.0
	_black_material_total = 0.0
	_white_tournament_scores.clear()
	_black_tournament_scores.clear()


func run_generation() -> void:
	## Run all games for one generation, evaluate fitness, and evolve.
	print("TrainingManager: Starting generation %d" % evolution.generation)
	_apply_curriculum()
	_reset_generation_metrics()
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
							print("Playing game: white_%d vs black_%d" % [w_idx, b_idx])
							var result = _play_game(w_idx, b_idx)
							print("Game result: %s" % result)
							_record_tournament_result(w_idx, b_idx, result)
							game_complete.emit(w_idx, b_idx, result["result"])
							total_games_played += 1
		else:
			# Round-robin tournament
			_tournament_pairings = _generate_round_robin_pairings(evolution.population_size)
			print("Generated %d tournament pairings" % _tournament_pairings.size())
			# Play all tournament games
			for w_idx in _tournament_pairings:
				for b_idx in _tournament_pairings[w_idx]:
					var result = _play_game(w_idx, b_idx)
					_record_tournament_result(w_idx, b_idx, result)
					game_complete.emit(w_idx, b_idx, result["result"])
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


func _track_game_metrics(game_result: Dictionary) -> void:
	## Track metrics for any completed game.
	_games_this_generation += 1
	_total_game_moves += game_result["move_count"]
	
	# Track material scores
	var state = game_result["state"]
	if state:
		_white_material_total += state.material_score(0)
		_black_material_total += state.material_score(1)
	
	# Track wins/draws/losses
	if game_result["result"] == 2:  # Draw
		_white_draws += 1
		_black_draws += 1
	elif game_result["result"] == 1:  # White wins
		_white_wins += 1
		_black_losses += 1
	else:  # Black wins
		_white_losses += 1
		_black_wins += 1


func _record_tournament_result(white_idx: int, black_idx: int, game_result) -> void:
	## Record tournament result for scoring.
	_track_game_metrics(game_result)
	
	if game_result["result"] == 2:  # Draw
		tournament_results[str(white_idx) + "_white"] = 0
		tournament_results[str(black_idx) + "_black"] = 0
	elif game_result["result"] == 1:  # White wins
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
		
		_apply_curriculum()
		# Reset generation metrics
		_reset_generation_metrics()
		
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
				game_complete.emit(w_idx, b_idx, result["result"])
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
		var opp := _select_black_opponent(_current_white_idx)
		var b_idx: int = opp["idx"]
		var use_hof: bool = opp["use_hof"]
		
		var result = _play_game_with_hof(_current_white_idx, b_idx, use_hof)
		_track_game_metrics(result)
		game_complete.emit(_current_white_idx, b_idx, result["result"])
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
			var opp := _select_black_opponent(w_idx)
			var b_idx: int = opp["idx"]
			var use_hof: bool = opp["use_hof"]
			
			var result = _play_game_with_hof(w_idx, b_idx, use_hof)
			_track_game_metrics(result)
			game_complete.emit(w_idx, b_idx, result["result"])
			games_played += 1
		
	return games_played


func _weighted_sample_index(fitness: PackedFloat32Array, target: float, temperature: float) -> int:
	if fitness.is_empty():
		return 0
	var denom: float = maxf(0.001, temperature)
	var total: float = 0.0
	for i in fitness.size():
		var diff: float = absf(fitness[i] - target)
		var w: float = 1.0 / (1.0 + (diff / denom))
		total += w
	if total <= 0.0:
		return randi() % fitness.size()
	var r: float = randf() * total
	var acc: float = 0.0
	for i in fitness.size():
		var diff: float = absf(fitness[i] - target)
		var w: float = 1.0 / (1.0 + (diff / denom))
		acc += w
		if r <= acc:
			return i
	return fitness.size() - 1


func _apply_curriculum() -> void:
	if not use_curriculum:
		return
	var gen: int = evolution.generation
	for stage in curriculum_stages:
		if gen >= stage["min_gen"] and gen <= stage["max_gen"]:
			max_moves_per_game = stage["max_moves"]
			ChessFitnessScript.set_weights(stage["weights"])
			return


func _select_black_opponent(white_idx: int) -> Dictionary:
	var use_hof := false
	var b_idx: int = -1
	
	if evolution.has_hall_of_fame(1) and randf() < hall_of_fame_ratio:
		use_hof = true
		b_idx = -1
	else:
		if use_fitness_weighted_sampling and evolution.black_fitness.size() == evolution.population_size:
			var target := 0.0
			if evolution.white_fitness.size() == evolution.population_size:
				target = evolution.white_fitness[white_idx]
			b_idx = _weighted_sample_index(evolution.black_fitness, target, opponent_sampling_temperature)
		else:
			b_idx = randi() % int(evolution.population_size)
	
	return {"idx": b_idx, "use_hof": use_hof}


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
	print("_play_game: white_idx=%d, black_idx=%d" % [white_idx, black_idx])
	var state := BoardStateScript.new()
	state.use_bitboard_movegen = use_bitboard_movegen
	state.setup_initial()
	print("Board initialized")

	# Track game history for visualization (every 5th move to reduce memory)
	var game_history: Array = []
	game_history.append(state.clone())

	# Get networks (either from current population or Hall of Fame)
	var white_net = null
	var black_net = null
	
	print("Getting networks...")
	if white_is_hof:
		white_net = evolution.get_hall_of_fame_opponent(0)
		if white_net == null:
			push_error("Hall of Fame requested but empty for white")
			return {"result": 2, "move_count": 0, "state": null}  # Return draw on error
	else:
		white_net = evolution.get_network(0, white_idx)
	
	if black_is_hof:
		black_net = evolution.get_hall_of_fame_opponent(1)
		if black_net == null:
			push_error("Hall of Fame requested but empty for black")
			return {"result": 2, "move_count": 0, "state": null}  # Return draw on error
	else:
		black_net = evolution.get_network(1, black_idx)
	
	print("Networks retrieved: white=%s, black=%s" % [white_net != null, black_net != null])

	if use_rust_batch_sim and not use_neat and not use_minimax and ClassDB.class_exists(&"RustBatchSimulator"):
		if _rust_batch_sim == null:
			_rust_batch_sim = ClassDB.instantiate("RustBatchSimulator")
		var w_weights: PackedFloat32Array = white_net.get_weights()
		var b_weights: PackedFloat32Array = black_net.get_weights()
		var stats: Dictionary = _rust_batch_sim.simulate_game(
			w_weights,
			b_weights,
			white_net.input_size,
			white_net.hidden_size,
			white_net.output_size,
			max_moves_per_game
		)
		var result_val: int = stats.get("result", 2)
		var moves_val: int = stats.get("move_count", 0)
		var w_fitness: float = ChessFitnessScript.evaluate_from_metrics(
			result_val,
			0,
			moves_val,
			stats.get("white_material", 0.0),
			stats.get("black_material", 0.0),
			stats.get("white_mobility", 0.0),
			stats.get("white_king_safety", 0.0),
			true
		)
		var b_fitness: float = ChessFitnessScript.evaluate_from_metrics(
			result_val,
			1,
			moves_val,
			stats.get("black_material", 0.0),
			stats.get("white_material", 0.0),
			stats.get("black_mobility", 0.0),
			stats.get("black_king_safety", 0.0),
			true
		)
		if not white_is_hof:
			var w_prev: float = evolution.white_fitness[white_idx]
			if use_tournament:
				evolution.set_fitness(0, white_idx, w_prev + w_fitness)
			else:
				evolution.set_fitness(0, white_idx, w_prev + w_fitness / games_per_individual)
		if not black_is_hof:
			var b_prev: float = evolution.black_fitness[black_idx]
			if use_tournament:
				evolution.set_fitness(1, black_idx, b_prev + b_fitness)
			else:
				evolution.set_fitness(1, black_idx, b_prev + b_fitness / games_per_individual)
		return {
			"state": null,
			"result": result_val,
			"move_count": moves_val,
			"material_score": null,
			"king_safety_score": null,
			"is_game_over": true
		}
	
	# Create minimax players if enabled
	var white_player = null
	var black_player = null
	if use_minimax:
		print("Creating minimax players with depth=%d" % minimax_depth)
		white_player = MinimaxPlayerScript.new(white_net, minimax_depth)
		black_player = MinimaxPlayerScript.new(black_net, minimax_depth)
		print("Minimax players created")

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
	print("Starting game loop, max_moves=%d" % max_moves_per_game)
	while not state.is_game_over and move_count < max_moves_per_game:
		var chosen := -1
		
		var legal_moves := state.generate_legal_moves()
		if legal_moves.is_empty():
			break
		
		if use_minimax:
			# Use minimax search to choose move
			var player = white_player if state.side_to_move == 0 else black_player
			print("Using minimax to choose move for %s" % ("white" if state.side_to_move == 0 else "black"))
			chosen = player.choose_move(state)
			print("Minimax chose: %s" % chosen)
		else:
			# Use direct network output (original behavior)
			var net = white_net if state.side_to_move == 0 else black_net
			var outputs: PackedFloat32Array
			if use_encoder_cache:
				var cache_key := state.get_position_key()
				if _encoder_cache.has(cache_key):
					outputs = _encoder_cache[cache_key]
				else:
					var inputs: PackedFloat32Array = ChessEncoderScript.encode_board(state)
					outputs = net.forward(inputs)
					_encoder_cache[cache_key] = outputs
					if _encoder_cache.size() > encoder_cache_max:
						_encoder_cache.clear()
			else:
				var inputs2: PackedFloat32Array = ChessEncoderScript.encode_board(state)
				outputs = net.forward(inputs2)
			chosen = ChessEncoderScript.decode_move(outputs, legal_moves)
		
		if chosen == -1:
			break  # Invalid move
		
		state.make_move(chosen)
		move_count += 1

		# Early termination (mercy rule) to save compute
		if move_count >= mercy_min_moves:
			var material_diff := state.material_score(0) - state.material_score(1)
			if absf(material_diff) >= mercy_material_threshold:
				state.is_game_over = true
				state.result = 1 if material_diff > 0.0 else -1
				break

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

	# Return game result with all needed data
	return {
		"state": state,
		"result": state.result,
		"move_count": move_count,
		"material_score": state.material_score.bind(state),  # Bind the method for later use
		"king_safety_score": state.king_safety_score.bind(state),
		"is_game_over": state.is_game_over
	}


func get_stats() -> Dictionary:
	var white_best: float = evolution.best_white_fitness
	var black_best: float = evolution.best_black_fitness
	var white_avg: float = evolution.get_avg_fitness(0)
	var black_avg: float = evolution.get_avg_fitness(1)
	
	# Calculate rates
	var games_count: float = max(1.0, float(_games_this_generation))
	var white_win_rate := float(_white_wins) / games_count
	var white_draw_rate := float(_white_draws) / games_count
	var white_loss_rate := float(_white_losses) / games_count
	var black_win_rate := float(_black_wins) / games_count
	var black_draw_rate := float(_black_draws) / games_count
	var black_loss_rate := float(_black_losses) / games_count
	
	# Calculate averages
	var avg_game_length := float(_total_game_moves) / games_count if games_count > 0 else 0.0
	var white_material_avg := _white_material_total / games_count if games_count > 0 else 0.0
	var black_material_avg := _black_material_total / games_count if games_count > 0 else 0.0
	
	# Calculate tournament scores
	var white_tournament_best := 0.0
	var white_tournament_avg := 0.0
	var black_tournament_best := 0.0
	var black_tournament_avg := 0.0
	
	if not _white_tournament_scores.is_empty():
		white_tournament_best = _white_tournament_scores.max()
		var white_sum := 0.0
		for score in _white_tournament_scores:
			white_sum += score
		white_tournament_avg = white_sum / float(_white_tournament_scores.size())
	
	if not _black_tournament_scores.is_empty():
		black_tournament_best = _black_tournament_scores.max()
		var black_sum := 0.0
		for score in _black_tournament_scores:
			black_sum += score
		black_tournament_avg = black_sum / float(_black_tournament_scores.size())
	
	# Calculate generation time
	var generation_time_sec := 0.0
	if _generation_start_time > 0:
		generation_time_sec = Time.get_unix_time_from_system() - _generation_start_time
	
	# Calculate total games this generation (different for tournament vs regular)
	var total_games_gen := _games_this_generation
	if total_games_gen == 0:  # Fallback for compatibility
		total_games_gen = evolution.population_size * games_per_individual
	
	# Derived throughput metrics
	var games_per_sec := 0.0
	var moves_per_sec := 0.0
	if generation_time_sec > 0.0:
		games_per_sec = float(total_games_gen) / generation_time_sec
		moves_per_sec = float(_total_game_moves) / generation_time_sec
	
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
		
		# New metrics
		"white_win_rate": white_win_rate,
		"white_draw_rate": white_draw_rate,
		"white_loss_rate": white_loss_rate,
		"black_win_rate": black_win_rate,
		"black_draw_rate": black_draw_rate,
		"black_loss_rate": black_loss_rate,
		"white_hof_size": evolution.hall_of_fame.size(),
		"black_hof_size": evolution.black_hall_of_fame.size(),
		"white_tournament_score_best": white_tournament_best,
		"white_tournament_score_avg": white_tournament_avg,
		"black_tournament_score_best": black_tournament_best,
		"black_tournament_score_avg": black_tournament_avg,
		"total_games_this_gen": total_games_gen,
		"avg_game_length": avg_game_length,
		"white_material_avg": white_material_avg,
		"black_material_avg": black_material_avg,
		"generation_time_sec": generation_time_sec,
		"games_per_sec": games_per_sec,
		"moves_per_sec": moves_per_sec,
	}
