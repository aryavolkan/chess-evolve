extends "res://test/test_base.gd"

const ChessPGNScript = preload("res://chess/pgn.gd")
const BoardStateScript = preload("res://chess/board_state.gd")
const Piece = ChessConstants.Piece  # gdlint:ignore = constant-name


func _run_tests() -> void:
	_test("square_name converts correctly", func():
		assert_eq(ChessPGNScript.square_name(0), "a1")
		assert_eq(ChessPGNScript.square_name(7), "h1")
		assert_eq(ChessPGNScript.square_name(56), "a8")
		assert_eq(ChessPGNScript.square_name(63), "h8")
		assert_eq(ChessPGNScript.square_name(12), "e2")
		assert_eq(ChessPGNScript.square_name(28), "e4")
	)

	_test("pawn move to SAN", func():
		var state := BoardStateScript.new()
		state.setup_initial()
		# e2-e4 = encode(12, 28)
		var move := BoardStateScript.encode_move(12, 28)
		var san := ChessPGNScript.move_to_san(state, move)
		assert_eq(san, "e4")
	)

	_test("knight move to SAN", func():
		var state := BoardStateScript.new()
		state.setup_initial()
		# Nb1-c3 = encode(1, 18)
		var move := BoardStateScript.encode_move(1, 18)
		var san := ChessPGNScript.move_to_san(state, move)
		assert_eq(san, "Nc3")
	)

	_test("pawn capture to SAN", func():
		# Set up: white pawn on e4, black pawn on d5
		var state := BoardStateScript.new()
		state.setup_initial()
		# Play e4, d5 to set up capture
		state.make_move(BoardStateScript.encode_move(12, 28))  # e4
		state.make_move(BoardStateScript.encode_move(51, 35))  # d5
		# Now exd5
		var move := BoardStateScript.encode_move(28, 35)
		var san := ChessPGNScript.move_to_san(state, move)
		assert_eq(san, "exd5")
	)

	_test("kingside castling to SAN", func():
		# Set up a position where white can castle kingside
		var state := BoardStateScript.new()
		state.board.fill(0)
		state.board[4] = Piece.KING    # White king on e1
		state.board[7] = Piece.ROOK    # White rook on h1
		state.board[60] = -Piece.KING  # Black king on e8
		state.white_king_sq = 4
		state.black_king_sq = 60
		state.side_to_move = 0
		state.castling_rights = 0b0001  # White kingside only
		state._mark_dirty()
		# O-O = king e1->g1 = encode(4, 6)
		var move := BoardStateScript.encode_move(4, 6)
		var san := ChessPGNScript.move_to_san(state, move)
		assert_eq(san, "O-O")
	)

	_test("queenside castling to SAN", func():
		var state := BoardStateScript.new()
		state.board.fill(0)
		state.board[4] = Piece.KING    # White king on e1
		state.board[0] = Piece.ROOK    # White rook on a1
		state.board[60] = -Piece.KING  # Black king on e8
		state.white_king_sq = 4
		state.black_king_sq = 60
		state.side_to_move = 0
		state.castling_rights = 0b0010  # White queenside only
		state._mark_dirty()
		# O-O-O = king e1->c1 = encode(4, 2)
		var move := BoardStateScript.encode_move(4, 2)
		var san := ChessPGNScript.move_to_san(state, move)
		assert_eq(san, "O-O-O")
	)

	_test("check suffix", func():
		# Set up: white queen can give check
		var state := BoardStateScript.new()
		state.board.fill(0)
		state.board[4] = Piece.KING    # White king on e1
		state.board[27] = Piece.QUEEN  # White queen on d4
		state.board[60] = -Piece.KING  # Black king on e8
		state.white_king_sq = 4
		state.black_king_sq = 60
		state.side_to_move = 0
		state._mark_dirty()
		# Qd4-d8 gives check: encode(27, 59)
		# d8=59. Is d4-d8 legal? Queen moves straight up file d.
		# d4=27 (rank 3, file 3), d8=59 (rank 7, file 3). Yes, straight line.
		var move := BoardStateScript.encode_move(27, 59)
		var san := ChessPGNScript.move_to_san(state, move)
		assert_true(san.ends_with("+") or san.ends_with("#"), "Expected check suffix, got: " + san)
	)

	_test("fool's mate checkmate suffix", func():
		# Play fool's mate: 1.f3 e5 2.g4 Qh4#
		var state := BoardStateScript.new()
		state.setup_initial()
		state.make_move(BoardStateScript.encode_move(13, 21))  # f2-f3
		state.make_move(BoardStateScript.encode_move(52, 36))  # e7-e5
		state.make_move(BoardStateScript.encode_move(14, 30))  # g2-g4
		# Qh4# = Qd8-h4 = encode(59, 31)
		var move := BoardStateScript.encode_move(59, 31)
		var san := ChessPGNScript.move_to_san(state, move)
		assert_true(san.ends_with("#"), "Expected checkmate suffix (#), got: " + san)
	)

	_test("full game PGN format", func():
		# Fool's mate as a complete game
		var moves := PackedInt32Array([
			BoardStateScript.encode_move(13, 21),  # f3
			BoardStateScript.encode_move(52, 36),  # e5
			BoardStateScript.encode_move(14, 30),  # g4
			BoardStateScript.encode_move(59, 31),  # Qh4#
		])
		var pgn := ChessPGNScript.game_to_pgn(moves, {"White": "Net-W", "Black": "Net-B"})
		assert_true(pgn.contains('[White "Net-W"]'), "Missing White header")
		assert_true(pgn.contains('[Black "Net-B"]'), "Missing Black header")
		assert_true(pgn.contains("1. f3"), "Missing 1. f3")
		assert_true(pgn.contains("0-1"), "Missing 0-1 result for black win")
	)

	_test("promotion to SAN", func():
		# White pawn on e7 (sq 52), empty e8 (sq 60)
		var state := BoardStateScript.new()
		state.board.fill(0)
		state.board[4] = Piece.KING     # White king on e1
		state.board[52] = Piece.PAWN    # White pawn on e7
		state.board[56] = -Piece.KING   # Black king on a8
		state.white_king_sq = 4
		state.black_king_sq = 56
		state.side_to_move = 0
		state._mark_dirty()
		# e7-e8=Q = encode(52, 60)
		var move := BoardStateScript.encode_move(52, 60)
		var san := ChessPGNScript.move_to_san(state, move)
		assert_true(san.contains("=Q"), "Expected promotion suffix =Q, got: " + san)
		assert_true(san.begins_with("e8"), "Expected e8=Q, got: " + san)
	)

	_test("piece_letter returns correct letters", func():
		assert_eq(ChessPGNScript.piece_letter(Piece.PAWN), "")
		assert_eq(ChessPGNScript.piece_letter(Piece.KNIGHT), "N")
		assert_eq(ChessPGNScript.piece_letter(Piece.BISHOP), "B")
		assert_eq(ChessPGNScript.piece_letter(Piece.ROOK), "R")
		assert_eq(ChessPGNScript.piece_letter(Piece.QUEEN), "Q")
		assert_eq(ChessPGNScript.piece_letter(Piece.KING), "K")
	)
