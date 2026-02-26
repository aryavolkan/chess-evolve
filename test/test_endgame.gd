extends "res://test/test_base.gd"

const EndgameHintsScript = preload("res://chess/endgame_hints.gd")
const BoardStateScript = preload("res://chess/board_state.gd")


func _run_tests() -> void:
	_test("K vs K is known draw", func():
		var state := BoardStateScript.new()
		state.board.fill(0)
		state.board[4] = 6   # White king on e1
		state.board[60] = -6  # Black king on e8
		state.white_king_sq = 4
		state.black_king_sq = 60

		assert_true(EndgameHintsScript.is_known_draw(state), "K vs K should be a known draw")
		var result := EndgameHintsScript.evaluate_endgame(state)
		assert_eq(result.type, "known_draw", "K vs K should have type known_draw")
	)

	_test("K+Q vs K detected", func():
		var state := BoardStateScript.new()
		state.board.fill(0)
		state.board[4] = 6    # White king on e1
		state.board[27] = 5   # White queen on d4
		state.board[60] = -6  # Black king on e8
		state.white_king_sq = 4
		state.black_king_sq = 60

		var result := EndgameHintsScript.evaluate_endgame(state)
		assert_eq(result.type, "KQ_vs_K", "Should detect K+Q vs K")
		assert_eq(result.advantage, 0, "White should have advantage")
		assert_gt(result.eval_bonus, 0.0, "Eval bonus should be positive for white")
	)

	_test("K+R vs K detected", func():
		var state := BoardStateScript.new()
		state.board.fill(0)
		state.board[4] = 6    # White king on e1
		state.board[27] = 4   # White rook on d4
		state.board[60] = -6  # Black king on e8
		state.white_king_sq = 4
		state.black_king_sq = 60

		var result := EndgameHintsScript.evaluate_endgame(state)
		assert_eq(result.type, "KR_vs_K", "Should detect K+R vs K")
		assert_eq(result.advantage, 0, "White should have advantage")
		assert_gt(result.eval_bonus, 0.0, "Eval bonus should be positive for white")
	)

	_test("K+B vs K is known draw", func():
		var state := BoardStateScript.new()
		state.board.fill(0)
		state.board[4] = 6    # White king on e1
		state.board[27] = 3   # White bishop on d4
		state.board[60] = -6  # Black king on e8
		state.white_king_sq = 4
		state.black_king_sq = 60

		assert_true(EndgameHintsScript.is_known_draw(state), "K+B vs K should be a known draw")
	)

	_test("K+N vs K is known draw", func():
		var state := BoardStateScript.new()
		state.board.fill(0)
		state.board[4] = 6    # White king on e1
		state.board[27] = 2   # White knight on d4
		state.board[60] = -6  # Black king on e8
		state.white_king_sq = 4
		state.black_king_sq = 60

		assert_true(EndgameHintsScript.is_known_draw(state), "K+N vs K should be a known draw")
	)

	_test("corner bonus higher than center", func():
		# K+R vs K with losing king in corner (a1 = sq 0)
		var state_corner := BoardStateScript.new()
		state_corner.board.fill(0)
		state_corner.board[4] = 6    # White king on e1
		state_corner.board[27] = 4   # White rook on d4
		state_corner.board[0] = -6   # Black king in corner a1
		state_corner.white_king_sq = 4
		state_corner.black_king_sq = 0

		var result_corner := EndgameHintsScript.evaluate_endgame(state_corner)

		# K+R vs K with losing king in center (e4 = sq 28)
		var state_center := BoardStateScript.new()
		state_center.board.fill(0)
		state_center.board[4] = 6    # White king on e1
		state_center.board[35] = 4   # White rook on d5 (moved to avoid overlap)
		state_center.board[28] = -6  # Black king in center e4
		state_center.white_king_sq = 4
		state_center.black_king_sq = 28

		var result_center := EndgameHintsScript.evaluate_endgame(state_center)

		assert_gt(result_corner.eval_bonus, result_center.eval_bonus,
			"Corner king should give higher bonus than center king")
	)

	_test("general endgame amplifies material", func():
		# White: king + rook + 3 pawns. Black: king + 3 pawns.
		var state := BoardStateScript.new()
		state.board.fill(0)
		state.board[4] = 6    # White king on e1
		state.board[60] = -6  # Black king on e8
		state.board[0] = 4    # White rook on a1
		state.board[8] = 1    # White pawn on a2
		state.board[9] = 1    # White pawn on b2
		state.board[10] = 1   # White pawn on c2
		state.board[48] = -1  # Black pawn on a7
		state.board[49] = -1  # Black pawn on b7
		state.board[50] = -1  # Black pawn on c7
		state.white_king_sq = 4
		state.black_king_sq = 60

		var result := EndgameHintsScript.evaluate_endgame(state)
		assert_eq(result.type, "general_endgame", "Should detect general endgame")
		assert_eq(result.advantage, 0, "White should have advantage")
		assert_gt(result.eval_bonus, 0.0, "Eval bonus should be positive (white has extra rook)")
	)

	_test("full board not an endgame", func():
		var state := BoardStateScript.new()
		state.setup_initial()

		var result := EndgameHintsScript.evaluate_endgame(state)
		assert_true(result.is_empty(), "Initial position should not be an endgame")
	)
