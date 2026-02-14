extends "res://test/test_base.gd"

const BoardStateScript = preload("res://chess/board_state.gd")


func _run_tests() -> void:
	_test("encode produces correct size", func():
		var b := BoardStateScript.new()
		b.setup_initial()
		var encoded := ChessEncoder.encode_board(b)
		assert_eq(encoded.size(), ChessEncoder.INPUT_SIZE)
	)

	_test("encoding has non-zero values", func():
		var b := BoardStateScript.new()
		b.setup_initial()
		var encoded := ChessEncoder.encode_board(b)
		var non_zero := 0
		for v in encoded:
			if v != 0.0: non_zero += 1
		assert_gt(non_zero, 30, "Should have many non-zero inputs")
	)

	_test("decode selects legal move", func():
		var b := BoardStateScript.new()
		b.setup_initial()
		var legal := b.generate_legal_moves()
		# Random output
		var outputs := PackedFloat32Array()
		outputs.resize(ChessEncoder.MOVE_OUTPUT_SIZE)
		for i in outputs.size():
			outputs[i] = randf_range(-1.0, 1.0)
		var move := ChessEncoder.decode_move(outputs, legal)
		assert_true(legal.has(move), "Decoded move should be legal")
	)

	_test("decode with empty moves returns -1,-1", func():
		var outputs := PackedFloat32Array()
		outputs.resize(128)
		var empty: Array[Vector2i] = []
		var move := ChessEncoder.decode_move(outputs, empty)
		assert_eq(move, Vector2i(-1, -1))
	)

	_test("encoding differs for different positions", func():
		var b1 := BoardStateScript.new()
		b1.setup_initial()
		var b2 := BoardStateScript.new()
		b2.setup_initial()
		b2.make_move(Vector2i(12, 28))  # e2-e4
		var e1 := ChessEncoder.encode_board(b1)
		var e2 := ChessEncoder.encode_board(b2)
		var diffs := 0
		for i in e1.size():
			if e1[i] != e2[i]: diffs += 1
		assert_gt(diffs, 0, "Different positions should encode differently")
	)
