extends "res://test/test_base.gd"

## End-to-end integration tests for the coevolution training loop.
## Tests the full pipeline: population init → game play → fitness eval → evolution.

const TrainingManagerScript = preload("res://ai/training_manager.gd")
const ChessEvolutionScript = preload("res://ai/evolution.gd")
const BoardStateScript = preload("res://chess/board_state.gd")
const ChessEncoderScript = preload("res://chess/encoder.gd")
const ChessFitnessScript = preload("res://ai/fitness.gd")
const NeuralNetworkScript = preload("res://ai/neural_network.gd")


func _run_tests() -> void:
	_test("full generation runs without errors", func():
		var evo = ChessEvolutionScript.new(6, 389, 16, 128, 2, 0.15, 0.2, 0.7)
		var tm = TrainingManagerScript.new(evo, 2, 50)
		tm.run_generation()
		assert_eq(evo.generation, 1, "Generation should be 1 after one run")
		assert_gt(tm.total_games_played, 0.0, "Should have played some games")
	)

	_test("two generations evolve populations", func():
		var evo = ChessEvolutionScript.new(6, 389, 16, 128, 2, 0.15, 0.2, 0.7)
		var tm = TrainingManagerScript.new(evo, 2, 50)
		tm.run_generation()
		tm.run_generation()
		assert_eq(evo.generation, 2)
		# Population size should be preserved
		assert_eq(evo.white_pop.size(), 6)
		assert_eq(evo.black_pop.size(), 6)
	)

	_test("fitness values are non-negative after generation", func():
		var evo = ChessEvolutionScript.new(6, 389, 16, 128, 2, 0.15, 0.2, 0.7)
		var tm = TrainingManagerScript.new(evo, 2, 50)

		# Manually run games without evolving to check fitness
		for w_idx in evo.population_size:
			for _g in 2:
				var b_idx: int = randi() % int(evo.population_size)
				_play_test_game(evo, w_idx, b_idx, 50)

		# Check all fitness values are non-negative
		for i in evo.population_size:
			assert_true(evo.white_fitness[i] >= 0.0, "White fitness[%d] should be >= 0" % i)
			assert_true(evo.black_fitness[i] >= 0.0, "Black fitness[%d] should be >= 0" % i)
	)

	_test("best fitness tracked across generations", func():
		var evo = ChessEvolutionScript.new(6, 389, 16, 128, 2, 0.15, 0.2, 0.7)
		var tm = TrainingManagerScript.new(evo, 2, 50)
		tm.run_generation()
		# all_time_best should be set after first generation
		assert_not_null(evo.all_time_best_white, "Should track all-time best white")
		assert_not_null(evo.all_time_best_black, "Should track all-time best black")
		assert_gt(evo.all_time_best_white_fitness, -INF, "Best white fitness should be set")
		assert_gt(evo.all_time_best_black_fitness, -INF, "Best black fitness should be set")
	)

	_test("incremental game step completes a generation", func():
		var evo = ChessEvolutionScript.new(4, 389, 16, 128, 1, 0.15, 0.2, 0.7)
		var tm = TrainingManagerScript.new(evo, 1, 30)

		var gen_complete := false
		var steps := 0
		while not gen_complete and steps < 100:
			gen_complete = tm.run_one_game_step()
			steps += 1

		assert_true(gen_complete, "Generation should complete within expected steps")
		assert_eq(evo.generation, 1)
		# Should have played exactly pop_size * games_per_individual games
		assert_eq(tm.total_games_played, 4)
	)

	_test("stats dictionary has required keys", func():
		var evo = ChessEvolutionScript.new(4, 389, 16, 128, 1, 0.15, 0.2, 0.7)
		var tm = TrainingManagerScript.new(evo, 1, 30)
		tm.run_generation()
		var stats := tm.get_stats()
		assert_true(stats.has("generation"), "Stats should have generation")
		assert_true(stats.has("white_best"), "Stats should have white_best")
		assert_true(stats.has("black_best"), "Stats should have black_best")
		assert_true(stats.has("white_avg"), "Stats should have white_avg")
		assert_true(stats.has("black_avg"), "Stats should have black_avg")
		assert_true(stats.has("games_played"), "Stats should have games_played")
		assert_true(stats.has("population_size"), "Stats should have population_size")
	)

	_test("training_step_complete signal fires", func():
		var evo = ChessEvolutionScript.new(4, 389, 16, 128, 1, 0.15, 0.2, 0.7)
		var tm = TrainingManagerScript.new(evo, 1, 30)
		var counter := [0, -1]  # [received, gen]
		tm.training_step_complete.connect(_make_step_cb(counter))
		tm.run_generation()
		assert_true(counter[0] > 0, "training_step_complete should fire")
		assert_eq(counter[1], 1, "Signal should report generation 1")
	)

	_test("game_complete signal fires for each game", func():
		var evo = ChessEvolutionScript.new(4, 389, 16, 128, 1, 0.15, 0.2, 0.7)
		var tm = TrainingManagerScript.new(evo, 1, 30)
		var counter := [0]
		tm.game_complete.connect(_make_game_cb(counter))
		tm.run_generation()
		assert_eq(counter[0], 4, "Should fire once per game (pop_size * games_per)")
	)

	_test("neural network outputs valid moves via encoder", func():
		var net = NeuralNetworkScript.new(389, 16, 128)
		var state = BoardStateScript.new()
		state.setup_initial()

		var inputs: PackedFloat32Array = ChessEncoderScript.encode_board(state)
		assert_eq(inputs.size(), 389, "Encoder should produce 389 inputs")

		var outputs: PackedFloat32Array = net.forward(inputs)
		assert_eq(outputs.size(), 128, "Network should produce 128 outputs")

		var legal_moves := state.generate_legal_moves()
		assert_gt(legal_moves.size(), 0.0, "Initial position should have legal moves")

		var chosen: Vector2i = ChessEncoderScript.decode_move(outputs, legal_moves)
		assert_true(legal_moves.has(chosen), "Decoded move should be in legal moves")
	)

	_test("complete game produces valid board state", func():
		var net_w = NeuralNetworkScript.new(389, 16, 128)
		var net_b = NeuralNetworkScript.new(389, 16, 128)
		var state = BoardStateScript.new()
		state.setup_initial()

		var moves := 0
		while not state.is_game_over and moves < 50:
			var net = net_w if state.side_to_move == 0 else net_b
			var inputs := ChessEncoderScript.encode_board(state)
			var outputs: PackedFloat32Array = net.forward(inputs)
			var legal := state.generate_legal_moves()
			if legal.is_empty():
				break
			var chosen: Vector2i = ChessEncoderScript.decode_move(outputs, legal)
			state.make_move(chosen)
			moves += 1

		assert_gt(moves, 0.0, "Should play at least one move")
		# Result should be valid (0=ongoing, 1=white wins, -1=black wins, 2=draw)
		if state.is_game_over:
			assert_true(state.result in [1, -1, 2], "Game result should be valid")
	)

	_test("fitness evaluation produces consistent results", func():
		var state = BoardStateScript.new()
		state.setup_initial()

		var fitness_w := ChessFitnessScript.evaluate(state, 0, 10)
		var fitness_b := ChessFitnessScript.evaluate(state, 1, 10)

		# Initial symmetric position should give roughly equal fitness
		assert_gt(fitness_w, 0.0, "White fitness should be positive")
		assert_gt(fitness_b, 0.0, "Black fitness should be positive")
		# Both should be equal for symmetric starting position
		assert_eq(fitness_w, fitness_b, "Fitness should be equal for starting position")
	)

	_test("evolution improves or maintains all-time best over 3 gens", func():
		var evo = ChessEvolutionScript.new(8, 389, 16, 128, 2, 0.15, 0.2, 0.7)
		var tm = TrainingManagerScript.new(evo, 2, 50)

		tm.run_generation()
		var first_best_w: float = evo.all_time_best_white_fitness
		var first_best_b: float = evo.all_time_best_black_fitness

		tm.run_generation()
		tm.run_generation()

		# All-time best should never decrease
		assert_true(evo.all_time_best_white_fitness >= first_best_w,
			"All-time best white should not decrease")
		assert_true(evo.all_time_best_black_fitness >= first_best_b,
			"All-time best black should not decrease")
	)

	_test("last_game_state is populated after training", func():
		var evo = ChessEvolutionScript.new(4, 389, 16, 128, 1, 0.15, 0.2, 0.7)
		var tm = TrainingManagerScript.new(evo, 1, 30)
		tm.run_generation()
		assert_not_null(tm.last_game_state, "last_game_state should be set")
		assert_gt(tm.last_game_history.size(), 0.0, "last_game_history should have entries")
	)


func _make_step_cb(counter: Array) -> Callable:
	return func(gen: int, _stats: Dictionary):
		counter[0] += 1
		counter[1] = gen

func _make_game_cb(counter: Array) -> Callable:
	return func(_w: int, _b: int, _r: int):
		counter[0] += 1

func _play_test_game(evo, w_idx: int, b_idx: int, max_moves: int) -> void:
	## Helper: play a game and accumulate fitness (mirrors TrainingManager._play_game).
	var state := BoardStateScript.new()
	state.setup_initial()
	var white_net = evo.get_network(0, w_idx)
	var black_net = evo.get_network(1, b_idx)
	var move_count := 0

	while not state.is_game_over and move_count < max_moves:
		var net = white_net if state.side_to_move == 0 else black_net
		var inputs := ChessEncoderScript.encode_board(state)
		var outputs: PackedFloat32Array = net.forward(inputs)
		var legal := state.generate_legal_moves()
		if legal.is_empty():
			break
		state.make_move(ChessEncoderScript.decode_move(outputs, legal))
		move_count += 1

	if not state.is_game_over:
		state.is_game_over = true
		state.result = 2

	var w_fit := ChessFitnessScript.evaluate(state, 0, move_count)
	var b_fit := ChessFitnessScript.evaluate(state, 1, move_count)
	evo.set_fitness(0, w_idx, evo.white_fitness[w_idx] + w_fit / 2.0)
	evo.set_fitness(1, b_idx, evo.black_fitness[b_idx] + b_fit / 2.0)
