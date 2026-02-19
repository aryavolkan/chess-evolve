extends "res://test/test_base.gd"


func _run_tests() -> void:
	_test("starting_position_piece_counts", func() -> void:
		var bs := BoardState.new()
		bs.setup_initial()
		var bb := BitboardState.from_board_state(bs)
		assert_eq(BitboardState._popcount(bb.bb_w_pawns), 8)
		assert_eq(BitboardState._popcount(bb.bb_w_knights), 2)
		assert_eq(BitboardState._popcount(bb.bb_w_bishops), 2)
		assert_eq(BitboardState._popcount(bb.bb_w_rooks), 2)
		assert_eq(BitboardState._popcount(bb.bb_w_queens), 1)
		assert_eq(BitboardState._popcount(bb.bb_w_king), 1)
		assert_eq(BitboardState._popcount(bb.bb_b_pawns), 8)
		assert_eq(BitboardState._popcount(bb.bb_b_knights), 2)
		assert_eq(BitboardState._popcount(bb.bb_b_bishops), 2)
		assert_eq(BitboardState._popcount(bb.bb_b_rooks), 2)
		assert_eq(BitboardState._popcount(bb.bb_b_queens), 1)
		assert_eq(BitboardState._popcount(bb.bb_b_king), 1)
	)

	_test("attack_tables_knight_a1", func() -> void:
		var sq := BitboardState.square(0, 0)
		var attacks := BitboardState.KNIGHT_ATTACKS[sq]
		assert_eq(BitboardState._popcount(attacks), 2)
		assert_true((attacks & (1 << BitboardState.square(1, 2))) != 0)
		assert_true((attacks & (1 << BitboardState.square(2, 1))) != 0)
	)

	_test("attack_tables_knight_e4", func() -> void:
		var sq := BitboardState.square(4, 3)
		var attacks := BitboardState.KNIGHT_ATTACKS[sq]
		assert_eq(BitboardState._popcount(attacks), 8)
		var expected := [
			BitboardState.square(2, 2),
			BitboardState.square(2, 4),
			BitboardState.square(3, 1),
			BitboardState.square(3, 5),
			BitboardState.square(5, 1),
			BitboardState.square(5, 5),
			BitboardState.square(6, 2),
			BitboardState.square(6, 4),
		]
		for target in expected:
			assert_true((attacks & (1 << target)) != 0)
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

	_test("from_board_state_matches_legal_moves_multiple_positions", func() -> void:
		var positions := [
			[],
			[_mv(4, 1, 4, 3)],  # e2-e4
			[_mv(4, 1, 4, 3), _mv(4, 6, 4, 4), _mv(6, 0, 5, 2), _mv(1, 7, 2, 5)],
			[_mv(3, 1, 3, 3), _mv(3, 6, 3, 4), _mv(2, 0, 6, 4)],
			[_mv(4, 1, 4, 3), _mv(2, 6, 2, 4), _mv(6, 0, 5, 2), _mv(3, 6, 3, 5), _mv(3, 1, 3, 3), _mv(2, 4, 3, 3)],
		]

		for sequence in positions:
			var bs := BoardState.new()
			bs.setup_initial()
			for mv in sequence:
				bs.make_move(mv)
			var bb := BitboardState.from_board_state(bs)
			var board_moves := _sorted_moves(bs.generate_legal_moves())
			var bb_moves := _sorted_moves(bb.get_legal_moves())
			assert_eq(bb_moves, board_moves)
	)

	_test("apply_move_matches_board_state_pawn_push", func() -> void:
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

	_test("apply_move_matches_board_state_capture", func() -> void:
		var moves := [
			BoardState.encode_move(12, 28), # e2-e4
			BoardState.encode_move(51, 35), # d7-d5
			BoardState.encode_move(28, 35), # e4xd5
		]
		var bs := BoardState.new()
		bs.setup_initial()
		for mv in moves:
			bs.make_move(mv)
		var expected := BitboardState.from_board_state(bs)

		var start := BoardState.new()
		start.setup_initial()
		var bb := BitboardState.from_board_state(start)
		for mv in moves:
			bb = bb.apply_move(mv)

		_assert_bitboards_equal(bb, expected)
	)

	_test("apply_move_matches_board_state_castling", func() -> void:
		var bs := BoardState.new()
		bs.board.fill(0)
		bs.board[4] = ChessConstants.Piece.KING
		bs.board[7] = ChessConstants.Piece.ROOK
		bs.board[60] = -ChessConstants.Piece.KING
		bs.castling_rights = 0b0001
		bs.side_to_move = 0
		bs._rebuild_piece_lists()
		var castle := BoardState.encode_move(4, 6)
		bs.make_move(castle)
		var expected := BitboardState.from_board_state(bs)

		var start := BoardState.new()
		start.board.fill(0)
		start.board[4] = ChessConstants.Piece.KING
		start.board[7] = ChessConstants.Piece.ROOK
		start.board[60] = -ChessConstants.Piece.KING
		start.castling_rights = 0b0001
		start.side_to_move = 0
		start._rebuild_piece_lists()
		var bb := BitboardState.from_board_state(start)
		var actual := bb.apply_move(BitboardState.encode_move(4, 6))

		_assert_bitboards_equal(actual, expected)
		assert_eq(actual.castling_rights, expected.castling_rights)
	)


func _mv(f1: int, r1: int, f2: int, r2: int) -> int:
	return BoardState.encode_move(BoardState.square(f1, r1), BoardState.square(f2, r2))


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
