extends "res://test/test_base.gd"

const BoardStateScript = preload("res://chess/board_state.gd")


func _run_tests() -> void:
	_test("initial position has equal fitness", func():
		var b := BoardStateScript.new()
		b.setup_initial()
		var w := ChessFitness.evaluate(b, 0, 0)
		var bl := ChessFitness.evaluate(b, 1, 0)
		# Should be roughly equal (material equal, mobility might differ since white moves first)
		assert_true(absf(w - bl) < 2.0, "Fitness should be roughly equal at start")
	)

	_test("winning position has higher fitness", func():
		var b := BoardStateScript.new()
		b.board.fill(0)
		b.board[4] = ChessConstants.Piece.KING
		b.board[60] = -ChessConstants.Piece.KING
		b.board[3] = ChessConstants.Piece.QUEEN  # White has extra queen
		b.is_game_over = false
		var w := ChessFitness.evaluate(b, 0, 10)
		var bl := ChessFitness.evaluate(b, 1, 10)
		assert_gt(w, bl, "White with queen should have higher fitness")
	)

	_test("checkmate gives high fitness to winner", func():
		var b := BoardStateScript.new()
		b.board.fill(0)
		b.board[4] = ChessConstants.Piece.KING
		b.board[60] = -ChessConstants.Piece.KING
		b.is_game_over = true
		b.result = 1  # White wins
		var w := ChessFitness.evaluate(b, 0, 20)
		assert_gt(w, 10.0, "Winner should have high fitness")
	)

	_test("longer games give more fitness", func():
		var b := BoardStateScript.new()
		b.setup_initial()
		var short := ChessFitness.evaluate(b, 0, 5)
		var long_game := ChessFitness.evaluate(b, 0, 50)
		assert_gt(long_game, short, "Longer games should give more fitness")
	)
