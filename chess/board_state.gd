extends RefCounted
class_name BoardState

## Represents a chess board state with full game logic.
## Board is an 8x8 array: positive = white, negative = black.
## Values correspond to ChessConstants.Piece enum.

const Piece = ChessConstants.Piece

var board: Array[int] = []  # 64 squares, row-major (0=a1, 63=h8)
var side_to_move: int = 0  # 0=white, 1=black
var castling_rights: int = 0b1111  # KQkq bits
var en_passant_square: int = -1  # target square or -1
var halfmove_clock: int = 0
var fullmove_number: int = 1
var is_game_over: bool = false
var result: int = 0  # 0=ongoing, 1=white wins, -1=black wins, 2=draw


func _init() -> void:
	board.resize(64)
	board.fill(0)


func setup_initial() -> void:
	## Set up the standard starting position.
	board.fill(0)
	# White pieces (rank 1 = indices 0-7)
	board[0] = Piece.ROOK; board[1] = Piece.KNIGHT; board[2] = Piece.BISHOP
	board[3] = Piece.QUEEN; board[4] = Piece.KING
	board[5] = Piece.BISHOP; board[6] = Piece.KNIGHT; board[7] = Piece.ROOK
	for i in range(8, 16):
		board[i] = Piece.PAWN
	# Black pieces (rank 8 = indices 56-63)
	board[56] = -Piece.ROOK; board[57] = -Piece.KNIGHT; board[58] = -Piece.BISHOP
	board[59] = -Piece.QUEEN; board[60] = -Piece.KING
	board[61] = -Piece.BISHOP; board[62] = -Piece.KNIGHT; board[63] = -Piece.ROOK
	for i in range(48, 56):
		board[i] = -Piece.PAWN
	side_to_move = 0
	castling_rights = 0b1111
	en_passant_square = -1
	halfmove_clock = 0
	fullmove_number = 1
	is_game_over = false
	result = 0


static func file_of(sq: int) -> int:
	return sq % 8

static func rank_of(sq: int) -> int:
	return sq / 8

static func square(file: int, rank: int) -> int:
	return rank * 8 + file

static func is_valid_square(sq: int) -> bool:
	return sq >= 0 and sq < 64


func piece_at(sq: int) -> int:
	return board[sq]

func is_white_piece(sq: int) -> bool:
	return board[sq] > 0

func is_black_piece(sq: int) -> bool:
	return board[sq] < 0

func piece_color(sq: int) -> int:
	if board[sq] > 0: return 0
	if board[sq] < 0: return 1
	return -1

func is_friendly(sq: int) -> bool:
	if side_to_move == 0: return board[sq] > 0
	return board[sq] < 0

func is_enemy(sq: int) -> bool:
	if side_to_move == 0: return board[sq] < 0
	return board[sq] > 0

func abs_piece(sq: int) -> int:
	return absi(board[sq])


func clone() -> BoardState:
	var copy: BoardState = get_script().new()
	copy.board = board.duplicate()
	copy.side_to_move = side_to_move
	copy.castling_rights = castling_rights
	copy.en_passant_square = en_passant_square
	copy.halfmove_clock = halfmove_clock
	copy.fullmove_number = fullmove_number
	copy.is_game_over = is_game_over
	copy.result = result
	return copy


func generate_legal_moves() -> Array[Vector2i]:
	## Returns array of Vector2i(from, to) for all legal moves.
	var moves: Array[Vector2i] = []
	var pseudo := _generate_pseudo_legal_moves()
	for m in pseudo:
		if _is_legal(m):
			moves.append(m)
	return moves


func _generate_pseudo_legal_moves() -> Array[Vector2i]:
	var moves: Array[Vector2i] = []
	for sq in range(64):
		if (side_to_move == 0 and board[sq] > 0) or (side_to_move == 1 and board[sq] < 0):
			_add_piece_moves(sq, moves)
	return moves


func _add_piece_moves(sq: int, moves: Array[Vector2i]) -> void:
	var piece := abs_piece(sq)
	match piece:
		Piece.PAWN: _add_pawn_moves(sq, moves)
		Piece.KNIGHT: _add_knight_moves(sq, moves)
		Piece.BISHOP: _add_slider_moves(sq, moves, [Vector2i(-1,-1), Vector2i(-1,1), Vector2i(1,-1), Vector2i(1,1)])
		Piece.ROOK: _add_slider_moves(sq, moves, [Vector2i(0,-1), Vector2i(0,1), Vector2i(-1,0), Vector2i(1,0)])
		Piece.QUEEN: _add_slider_moves(sq, moves, [Vector2i(-1,-1), Vector2i(-1,1), Vector2i(1,-1), Vector2i(1,1), Vector2i(0,-1), Vector2i(0,1), Vector2i(-1,0), Vector2i(1,0)])
		Piece.KING: _add_king_moves(sq, moves)


func _add_pawn_moves(sq: int, moves: Array[Vector2i]) -> void:
	var dir := 1 if side_to_move == 0 else -1
	var start_rank := 1 if side_to_move == 0 else 6
	var f := file_of(sq)
	var r := rank_of(sq)

	# Forward one
	var fwd := square(f, r + dir)
	if is_valid_square(fwd) and board[fwd] == 0:
		moves.append(Vector2i(sq, fwd))
		# Forward two from starting rank
		if r == start_rank:
			var fwd2 := square(f, r + 2 * dir)
			if board[fwd2] == 0:
				moves.append(Vector2i(sq, fwd2))

	# Captures
	for df in [-1, 1]:
		var nf: int = f + df
		if nf < 0 or nf > 7: continue
		var cap_sq := square(nf, r + dir)
		if not is_valid_square(cap_sq): continue
		if is_enemy(cap_sq):
			moves.append(Vector2i(sq, cap_sq))
		elif cap_sq == en_passant_square:
			moves.append(Vector2i(sq, cap_sq))


func _add_knight_moves(sq: int, moves: Array[Vector2i]) -> void:
	var f := file_of(sq)
	var r := rank_of(sq)
	var offsets := [Vector2i(-2,-1), Vector2i(-2,1), Vector2i(-1,-2), Vector2i(-1,2),
					Vector2i(1,-2), Vector2i(1,2), Vector2i(2,-1), Vector2i(2,1)]
	for off in offsets:
		var nf: int = f + off.x
		var nr: int = r + off.y
		if nf < 0 or nf > 7 or nr < 0 or nr > 7: continue
		var target := square(nf, nr)
		if not is_friendly(target):
			moves.append(Vector2i(sq, target))


func _add_slider_moves(sq: int, moves: Array[Vector2i], directions: Array) -> void:
	var f := file_of(sq)
	var r := rank_of(sq)
	for dir: Vector2i in directions:
		var nf := f + dir.x
		var nr := r + dir.y
		while nf >= 0 and nf <= 7 and nr >= 0 and nr <= 7:
			var target := square(nf, nr)
			if is_friendly(target): break
			moves.append(Vector2i(sq, target))
			if is_enemy(target): break
			nf += dir.x
			nr += dir.y


func _add_king_moves(sq: int, moves: Array[Vector2i]) -> void:
	var f := file_of(sq)
	var r := rank_of(sq)
	for df in [-1, 0, 1]:
		for dr in [-1, 0, 1]:
			if df == 0 and dr == 0: continue
			var nf: int = f + df
			var nr: int = r + dr
			if nf < 0 or nf > 7 or nr < 0 or nr > 7: continue
			var target := square(nf, nr)
			if not is_friendly(target):
				moves.append(Vector2i(sq, target))

	# Castling
	if side_to_move == 0 and sq == 4:
		if castling_rights & 0b0001 and board[5] == 0 and board[6] == 0 and board[7] == Piece.ROOK:
			if not _is_square_attacked(4, 1) and not _is_square_attacked(5, 1):
				moves.append(Vector2i(4, 6))
		if castling_rights & 0b0010 and board[3] == 0 and board[2] == 0 and board[1] == 0 and board[0] == Piece.ROOK:
			if not _is_square_attacked(4, 1) and not _is_square_attacked(3, 1):
				moves.append(Vector2i(4, 2))
	elif side_to_move == 1 and sq == 60:
		if castling_rights & 0b0100 and board[61] == 0 and board[62] == 0 and board[63] == -Piece.ROOK:
			if not _is_square_attacked(60, 0) and not _is_square_attacked(61, 0):
				moves.append(Vector2i(60, 62))
		if castling_rights & 0b1000 and board[59] == 0 and board[58] == 0 and board[57] == 0 and board[56] == -Piece.ROOK:
			if not _is_square_attacked(60, 0) and not _is_square_attacked(59, 0):
				moves.append(Vector2i(60, 58))


func _is_legal(move: Vector2i) -> bool:
	## Check if a move leaves our king safe.
	var test := clone()
	test._apply_move_unchecked(move)
	var king_sq := test._find_king(side_to_move)
	if king_sq == -1: return false
	return not test._is_square_attacked(king_sq, 1 - side_to_move)


func _find_king(color: int) -> int:
	var king_val := Piece.KING if color == 0 else -Piece.KING
	for sq in range(64):
		if board[sq] == king_val: return sq
	return -1


func _is_square_attacked(sq: int, by_color: int) -> bool:
	## Check if 'sq' is attacked by any piece of 'by_color'.
	var sign := 1 if by_color == 0 else -1
	var f := file_of(sq)
	var r := rank_of(sq)

	# Knight attacks
	var knight_offsets := [Vector2i(-2,-1), Vector2i(-2,1), Vector2i(-1,-2), Vector2i(-1,2),
						   Vector2i(1,-2), Vector2i(1,2), Vector2i(2,-1), Vector2i(2,1)]
	for off in knight_offsets:
		var nf: int = f + off.x
		var nr: int = r + off.y
		if nf >= 0 and nf <= 7 and nr >= 0 and nr <= 7:
			if board[square(nf, nr)] == sign * Piece.KNIGHT: return true

	# Pawn attacks
	var pawn_dir := -1 if by_color == 0 else 1  # Pawns attack from behind
	for df in [-1, 1]:
		var nf: int = f + df
		var nr: int = r + pawn_dir
		if nf >= 0 and nf <= 7 and nr >= 0 and nr <= 7:
			if board[square(nf, nr)] == sign * Piece.PAWN: return true

	# King attacks
	for df in [-1, 0, 1]:
		for dr in [-1, 0, 1]:
			if df == 0 and dr == 0: continue
			var nf: int = f + df
			var nr: int = r + dr
			if nf >= 0 and nf <= 7 and nr >= 0 and nr <= 7:
				if board[square(nf, nr)] == sign * Piece.KING: return true

	# Slider attacks (bishop/queen diagonals, rook/queen straights)
	var diag_dirs := [Vector2i(-1,-1), Vector2i(-1,1), Vector2i(1,-1), Vector2i(1,1)]
	for dir in diag_dirs:
		var nf: int = f + dir.x
		var nr: int = r + dir.y
		while nf >= 0 and nf <= 7 and nr >= 0 and nr <= 7:
			var p := board[square(nf, nr)]
			if p != 0:
				if p == sign * Piece.BISHOP or p == sign * Piece.QUEEN: return true
				break
			nf += dir.x; nr += dir.y

	var straight_dirs := [Vector2i(0,-1), Vector2i(0,1), Vector2i(-1,0), Vector2i(1,0)]
	for dir in straight_dirs:
		var nf: int = f + dir.x
		var nr: int = r + dir.y
		while nf >= 0 and nf <= 7 and nr >= 0 and nr <= 7:
			var p := board[square(nf, nr)]
			if p != 0:
				if p == sign * Piece.ROOK or p == sign * Piece.QUEEN: return true
				break
			nf += dir.x; nr += dir.y

	return false


func make_move(move: Vector2i) -> void:
	## Apply a legal move and update game state.
	_apply_move_unchecked(move)
	side_to_move = 1 - side_to_move
	if side_to_move == 0:
		fullmove_number += 1
	_check_game_over()


func _apply_move_unchecked(move: Vector2i) -> void:
	var from := move.x
	var to := move.y
	var piece := board[from]
	var abs_p := absi(piece)
	var captured := board[to]

	# En passant capture
	if abs_p == Piece.PAWN and to == en_passant_square:
		var ep_captured_sq := to + (-8 if side_to_move == 0 else 8)
		board[ep_captured_sq] = 0

	# Update halfmove clock
	if abs_p == Piece.PAWN or captured != 0:
		halfmove_clock = 0
	else:
		halfmove_clock += 1

	# En passant square
	en_passant_square = -1
	if abs_p == Piece.PAWN and absi(rank_of(to) - rank_of(from)) == 2:
		en_passant_square = square(file_of(from), (rank_of(from) + rank_of(to)) / 2)

	# Castling: move rook
	if abs_p == Piece.KING:
		if to - from == 2:  # Kingside
			board[to - 1] = board[to + 1]; board[to + 1] = 0
		elif from - to == 2:  # Queenside
			board[to + 1] = board[to - 2]; board[to - 2] = 0

	# Update castling rights
	if abs_p == Piece.KING:
		if side_to_move == 0: castling_rights &= 0b1100
		else: castling_rights &= 0b0011
	if from == 0 or to == 0: castling_rights &= ~0b0010
	if from == 7 or to == 7: castling_rights &= ~0b0001
	if from == 56 or to == 56: castling_rights &= ~0b1000
	if from == 63 or to == 63: castling_rights &= ~0b0100

	# Move piece
	board[to] = piece
	board[from] = 0

	# Pawn promotion (auto-queen)
	if abs_p == Piece.PAWN:
		if (side_to_move == 0 and rank_of(to) == 7) or (side_to_move == 1 and rank_of(to) == 0):
			board[to] = Piece.QUEEN if side_to_move == 0 else -Piece.QUEEN


func _check_game_over() -> void:
	var moves := generate_legal_moves()
	if moves.size() == 0:
		is_game_over = true
		var king_sq := _find_king(side_to_move)
		if king_sq != -1 and _is_square_attacked(king_sq, 1 - side_to_move):
			result = 1 if side_to_move == 1 else -1  # Checkmate
		else:
			result = 2  # Stalemate
	elif halfmove_clock >= 100:
		is_game_over = true
		result = 2  # 50-move rule


func is_in_check() -> bool:
	var king_sq := _find_king(side_to_move)
	if king_sq == -1: return false
	return _is_square_attacked(king_sq, 1 - side_to_move)


func material_score(color: int) -> float:
	## Sum material value for given color.
	var total := 0.0
	var sign := 1 if color == 0 else -1
	for sq in range(64):
		if (color == 0 and board[sq] > 0) or (color == 1 and board[sq] < 0):
			total += ChessConstants.PIECE_VALUES.get(absi(board[sq]), 0.0)
	return total


func mobility_score(color: int) -> int:
	## Count number of legal moves for a color (approximation: uses pseudo-legal).
	var saved_side := side_to_move
	side_to_move = color
	var moves := generate_legal_moves()
	side_to_move = saved_side
	return moves.size()


func king_safety_score(color: int) -> float:
	## Simple king safety: count friendly pawns near king.
	var king_sq := _find_king(color)
	if king_sq == -1: return 0.0
	var kf := file_of(king_sq)
	var kr := rank_of(king_sq)
	var safety := 0.0
	var sign := 1 if color == 0 else -1
	for df in [-1, 0, 1]:
		for dr in [-1, 0, 1]:
			var nf: int = kf + df
			var nr: int = kr + dr
			if nf >= 0 and nf <= 7 and nr >= 0 and nr <= 7:
				var p := board[square(nf, nr)]
				if p == sign * Piece.PAWN:
					safety += 1.0
	return safety


func to_string_board() -> String:
	var s := ""
	for r in range(7, -1, -1):
		s += str(r + 1) + " "
		for f in range(8):
			var p := board[square(f, r)]
			if p == 0:
				s += ". "
			else:
				var sym: String = ChessConstants.PIECE_SYMBOLS.get(absi(p), "?")
				s += sym.to_lower() + " " if p < 0 else sym + " "
		s += "\n"
	s += "  a b c d e f g h"
	return s
