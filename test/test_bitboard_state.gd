extends "res://test/test_base.gd"


func _run_tests() -> void:
	_test("attack_tables_knight_a1", func() -> void:
		var sq := BitboardState.square(0, 0)
		var attacks := BitboardState.KNIGHT_ATTACKS[sq]
		assert_eq(BitboardState._popcount(attacks), 2)
		assert_true((attacks & (1 << BitboardState.square(1, 2))) != 0)
		assert_true((attacks & (1 << BitboardState.square(2, 1))) != 0)
	)

	_test("attack_tables_king_e1", func() -> void:
		var sq := BitboardState.square(4, 0)
		var attacks := BitboardState.KING_ATTACKS[sq]
		assert_eq(BitboardState._popcount(attacks), 5)
		assert_true((attacks & (1 << BitboardState.square(3, 0))) != 0)
		assert_true((attacks & (1 << BitboardState.square(5, 0))) != 0)
		assert_true((attacks & (1 << BitboardState.square(3, 1))) != 0)
		assert_true((attacks & (1 << BitboardState.square(4, 1))) != 0)
		assert_true((attacks & (1 << BitboardState.square(5, 1))) != 0)
	)

	_test("attack_tables_pawn_e4", func() -> void:
		var sq := BitboardState.square(4, 3)
		var w_attacks := BitboardState.PAWN_ATTACKS_W[sq]
		var b_attacks := BitboardState.PAWN_ATTACKS_B[sq]
		assert_true((w_attacks & (1 << BitboardState.square(3, 4))) != 0)
		assert_true((w_attacks & (1 << BitboardState.square(5, 4))) != 0)
		assert_true((b_attacks & (1 << BitboardState.square(3, 2))) != 0)
		assert_true((b_attacks & (1 << BitboardState.square(5, 2))) != 0)
	)

	_test("attack_tables_ray_d4", func() -> void:
		var sq := BitboardState.square(3, 3)
		var ray_n := BitboardState.RAY_N[sq]
		assert_eq(BitboardState._popcount(ray_n), 4)
		assert_true((ray_n & (1 << BitboardState.square(3, 4))) != 0)
		assert_true((ray_n & (1 << BitboardState.square(3, 7))) != 0)
	)

	_test("from_board_state_matches_legal_moves", func() -> void:
		var bs := BoardState.new()
		bs.setup_initial()
		var bb := BitboardState.from_board_state(bs)
		var board_moves := _sorted_moves(bs.generate_legal_moves())
		var bb_moves := _sorted_moves(bb.get_legal_moves())
		assert_eq(bb_moves, board_moves)
	)

	_test("apply_move_matches_board_state", func() -> void:
		var bs := BoardState.new()
		bs.setup_initial()
		var move := BoardState.encode_move(12, 28) # e2 -> e4
		bs.make_move(move)
		var expected := BitboardState.from_board_state(bs)

		var start := BoardState.new()
		start.setup_initial()
		var bb := BitboardState.from_board_state(start)
		var actual := bb.apply_move(BitboardState.encode_move(12, 28))

		_assert_bitboards_equal(actual, expected)
		assert_eq(actual.en_passant_square, expected.en_passant_square)
		assert_eq(actual.side_to_move, expected.side_to_move)
		assert_eq(actual.castling_rights, expected.castling_rights)
	)


func _sorted_moves(moves: PackedInt32Array) -> Array:
	var arr: Array = []
	for m in moves:
		arr.append(m)
	arr.sort()
	return arr


func _assert_bitboards_equal(actual: BitboardState, expected: BitboardState) -> void:
	assert_eq(actual.bb_w_pawns, expected.bb_w_pawns)
	assert_eq(actual.bb_w_knights, expected.bb_w_knights)
	assert_eq(actual.bb_w_bishops, expected.bb_w_bishops)
	assert_eq(actual.bb_w_rooks, expected.bb_w_rooks)
	assert_eq(actual.bb_w_queens, expected.bb_w_queens)
	assert_eq(actual.bb_w_king, expected.bb_w_king)
	assert_eq(actual.bb_b_pawns, expected.bb_b_pawns)
	assert_eq(actual.bb_b_knights, expected.bb_b_knights)
	assert_eq(actual.bb_b_bishops, expected.bb_b_bishops)
	assert_eq(actual.bb_b_rooks, expected.bb_b_rooks)
	assert_eq(actual.bb_b_queens, expected.bb_b_queens)
	assert_eq(actual.bb_b_king, expected.bb_b_king)
