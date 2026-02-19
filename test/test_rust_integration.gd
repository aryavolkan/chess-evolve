extends "res://test/test_base.gd"

const BoardStateScript = preload("res://chess/board_state.gd")


func _run_tests() -> void:
	_test("rust availability check does not crash", func():
		var available := ChessRustIntegration.is_rust_available()
		assert_eq(typeof(available), TYPE_BOOL)
	)

	_test("gds fallback board has 20 legal moves at start", func():
		var state := BoardStateScript.new()
		state.setup_initial()
		assert_eq(state.generate_legal_moves().size(), 20)
	)

	_test("rust legal move count matches GDScript", func():
		if ChessRustIntegration.is_rust_available():
			var state := BoardStateScript.new()
			state.setup_initial()
			var rust_board := RustChessBoard.new()
			var rust_moves := rust_board.get_legal_moves()
			var gd_moves := state.generate_legal_moves()
			assert_eq(rust_moves.size(), gd_moves.size())
		else:
			assert_true(true)
	)

	_test("rust make_move flips side_to_move", func():
		if ChessRustIntegration.is_rust_available():
			var rust_board := RustChessBoard.new()
			var start_side := rust_board.side_to_move()
			var moves := rust_board.get_legal_moves()
			assert_gt(moves.size(), 0)
			rust_board.make_move(moves[0])
			assert_ne(rust_board.side_to_move(), start_side)
		else:
			assert_true(true)
	)

	_test("rust dense network forward returns correct output size", func():
		if ClassDB.class_exists(&"RustDenseNetwork"):
			var net := RustDenseNetwork.new()
			net.setup(4, 8, 218)
			var inputs := PackedFloat32Array()
			inputs.resize(4)
			inputs.fill(0.0)
			var out := net.forward(inputs)
			assert_eq(out.size(), 218)
		else:
			assert_true(true)
	)

	_test("rust dense network zero weights returns zeros", func():
		if ClassDB.class_exists(&"RustDenseNetwork"):
			var net := RustDenseNetwork.new()
			net.setup(4, 8, 218)
			var weights_ih := PackedFloat32Array()
			weights_ih.resize(4 * 8)
			weights_ih.fill(0.0)
			var biases_h := PackedFloat32Array()
			biases_h.resize(8)
			biases_h.fill(0.0)
			var weights_ho := PackedFloat32Array()
			weights_ho.resize(8 * 218)
			weights_ho.fill(0.0)
			var biases_o := PackedFloat32Array()
			biases_o.resize(218)
			biases_o.fill(0.0)
			net.set_weights(weights_ih, biases_h, weights_ho, biases_o)
			var inputs := PackedFloat32Array()
			inputs.resize(4)
			inputs.fill(0.0)
			var out := net.forward(inputs)
			for v in out:
				assert_true(absf(v) < 0.0001)
		else:
			assert_true(true)
	)

	_test("rust availability check stays safe on repeat", func():
		var available := ChessRustIntegration.is_rust_available()
		assert_eq(typeof(available), TYPE_BOOL)
	)
