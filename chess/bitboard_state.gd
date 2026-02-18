class_name BitboardState
extends RefCounted

const Piece = ChessConstants.Piece  # gdlint:ignore = constant-name

const MOVE_FLAG_PROMOTE := 1 << 12
const MOVE_FLAG_EN_PASSANT := 1 << 13

static var KNIGHT_ATTACKS: Array[int] = []
static var KING_ATTACKS: Array[int] = []
static var PAWN_ATTACKS_W: Array[int] = []
static var PAWN_ATTACKS_B: Array[int] = []

static var RAY_N: Array[int] = []
static var RAY_S: Array[int] = []
static var RAY_E: Array[int] = []
static var RAY_W: Array[int] = []
static var RAY_NE: Array[int] = []
static var RAY_NW: Array[int] = []
static var RAY_SE: Array[int] = []
static var RAY_SW: Array[int] = []

var bb_w_pawns: int = 0
var bb_w_knights: int = 0
var bb_w_bishops: int = 0
var bb_w_rooks: int = 0
var bb_w_queens: int = 0
var bb_w_king: int = 0

var bb_b_pawns: int = 0
var bb_b_knights: int = 0
var bb_b_bishops: int = 0
var bb_b_rooks: int = 0
var bb_b_queens: int = 0
var bb_b_king: int = 0

var side_to_move: int = 0
var castling_rights: int = 0b1111
var en_passant_square: int = -1
var halfmove_clock: int = 0
var fullmove_number: int = 1
var is_game_over: bool = false
var result: int = 0

var _white_king_sq: int = -1
var _black_king_sq: int = -1


static func _static_init() -> void:
	KNIGHT_ATTACKS.resize(64)
	KING_ATTACKS.resize(64)
	PAWN_ATTACKS_W.resize(64)
	PAWN_ATTACKS_B.resize(64)
	RAY_N.resize(64)
	RAY_S.resize(64)
	RAY_E.resize(64)
	RAY_W.resize(64)
	RAY_NE.resize(64)
	RAY_NW.resize(64)
	RAY_SE.resize(64)
	RAY_SW.resize(64)

	for sq in range(64):
		KNIGHT_ATTACKS[sq] = _compute_knight_attacks(sq)
		KING_ATTACKS[sq] = _compute_king_attacks(sq)
		PAWN_ATTACKS_W[sq] = _compute_pawn_attacks(sq, 0)
		PAWN_ATTACKS_B[sq] = _compute_pawn_attacks(sq, 1)
		RAY_N[sq] = _compute_ray(sq, 0, 1)
		RAY_S[sq] = _compute_ray(sq, 0, -1)
		RAY_E[sq] = _compute_ray(sq, 1, 0)
		RAY_W[sq] = _compute_ray(sq, -1, 0)
		RAY_NE[sq] = _compute_ray(sq, 1, 1)
		RAY_NW[sq] = _compute_ray(sq, -1, 1)
		RAY_SE[sq] = _compute_ray(sq, 1, -1)
		RAY_SW[sq] = _compute_ray(sq, -1, -1)


static func file_of(sq: int) -> int:
	return sq % 8

static func rank_of(sq: int) -> int:
	return sq / 8

static func square(file: int, rank: int) -> int:
	return rank * 8 + file

static func is_valid_square(sq: int) -> bool:
	return sq >= 0 and sq < 64

static func encode_move(from_sq: int, to_sq: int, flags: int = 0) -> int:
	return (from_sq << 6) | to_sq | flags

static func move_from(encoded_move: int) -> int:
	return (encoded_move >> 6) & 0x3f

static func move_to(encoded_move: int) -> int:
	return encoded_move & 0x3f

static func move_flags(encoded_move: int) -> int:
	return encoded_move & ~0xfff


static func from_board_state(bs: BoardState) -> BitboardState:
	var state: BitboardState = BitboardState.new()
	state.side_to_move = bs.side_to_move
	state.castling_rights = bs.castling_rights
	state.en_passant_square = bs.en_passant_square
	state.halfmove_clock = bs.halfmove_clock
	state.fullmove_number = bs.fullmove_number
	state.is_game_over = bs.is_game_over
	state.result = bs.result

	for sq in range(64):
		var p: int = bs.board[sq]
		if p == 0:
			continue
			
		state._set_piece_at(sq, p)

	state._white_king_sq = state._find_king(0)
	state._black_king_sq = state._find_king(1)
	return state


func clone() -> BitboardState:
	var copy: BitboardState = BitboardState.new()
	copy.bb_w_pawns = bb_w_pawns
	copy.bb_w_knights = bb_w_knights
	copy.bb_w_bishops = bb_w_bishops
	copy.bb_w_rooks = bb_w_rooks
	copy.bb_w_queens = bb_w_queens
	copy.bb_w_king = bb_w_king
	copy.bb_b_pawns = bb_b_pawns
	copy.bb_b_knights = bb_b_knights
	copy.bb_b_bishops = bb_b_bishops
	copy.bb_b_rooks = bb_b_rooks
	copy.bb_b_queens = bb_b_queens
	copy.bb_b_king = bb_b_king
	copy.side_to_move = side_to_move
	copy.castling_rights = castling_rights
	copy.en_passant_square = en_passant_square
	copy.halfmove_clock = halfmove_clock
	copy.fullmove_number = fullmove_number
	copy.is_game_over = is_game_over
	copy.result = result
	copy._white_king_sq = _white_king_sq
	copy._black_king_sq = _black_king_sq
	return copy


func get_legal_moves() -> PackedInt32Array:
	var moves := PackedInt32Array()
	var pseudo := _generate_pseudo_legal_moves()
	for m in pseudo:
		var next := apply_move(m)
		var moved_color := 1 - next.side_to_move
		if not next.is_in_check(moved_color):
			moves.append(m)
	return moves


func apply_move(move: int) -> BitboardState:
	var next := clone()
	next._apply_move_in_place(move)
	return next


func is_in_check(color: int) -> bool:
	var king_sq := _find_king(color)
	if king_sq == -1:
		return false
	return _is_square_attacked(king_sq, 1 - color)


func _generate_pseudo_legal_moves() -> PackedInt32Array:
	var moves := PackedInt32Array()
	var white_occ := _white_occ()
	var black_occ := _black_occ()
	var all_occ := white_occ | black_occ
	var friendly_occ := white_occ if side_to_move == 0 else black_occ
	var enemy_occ := black_occ if side_to_move == 0 else white_occ

	if side_to_move == 0:
		_add_pawn_moves(bb_w_pawns, 0, friendly_occ, enemy_occ, moves)
		_add_knight_moves(bb_w_knights, friendly_occ, moves)
		_add_slider_moves(bb_w_bishops, all_occ, friendly_occ, moves, true)
		_add_slider_moves(bb_w_rooks, all_occ, friendly_occ, moves, false)
		_add_slider_moves(bb_w_queens, all_occ, friendly_occ, moves, true)
		_add_slider_moves(bb_w_queens, all_occ, friendly_occ, moves, false)
		_add_king_moves(bb_w_king, friendly_occ, enemy_occ, all_occ, moves)
	else:
		_add_pawn_moves(bb_b_pawns, 1, friendly_occ, enemy_occ, moves)
		_add_knight_moves(bb_b_knights, friendly_occ, moves)
		_add_slider_moves(bb_b_bishops, all_occ, friendly_occ, moves, true)
		_add_slider_moves(bb_b_rooks, all_occ, friendly_occ, moves, false)
		_add_slider_moves(bb_b_queens, all_occ, friendly_occ, moves, true)
		_add_slider_moves(bb_b_queens, all_occ, friendly_occ, moves, false)
		_add_king_moves(bb_b_king, friendly_occ, enemy_occ, all_occ, moves)

	return moves


func _add_pawn_moves(pawns: int, color: int, friendly_occ: int, enemy_occ: int, moves: PackedInt32Array) -> void:
	var dir := 8 if color == 0 else -8
	var start_rank := 1 if color == 0 else 6
	var promotion_rank := 7 if color == 0 else 0
	var pawn_attacks := PAWN_ATTACKS_W if color == 0 else PAWN_ATTACKS_B

	var bb := pawns
	while bb != 0:
		var sq := _bit_scan_forward(bb)
		bb &= bb - 1
		var r := rank_of(sq)

		var one_sq := sq + dir
		if is_valid_square(one_sq) and ((_bit_at(friendly_occ | enemy_occ, one_sq)) == 0):
			var flags := MOVE_FLAG_PROMOTE if rank_of(one_sq) == promotion_rank else 0
			moves.append(encode_move(sq, one_sq, flags))
			if r == start_rank:
				var two_sq := sq + dir * 2
				if is_valid_square(two_sq) and ((_bit_at(friendly_occ | enemy_occ, two_sq)) == 0):
					moves.append(encode_move(sq, two_sq))

		var attacks := pawn_attacks[sq] & enemy_occ
		while attacks != 0:
			var to_sq := _bit_scan_forward(attacks)
			attacks &= attacks - 1
			var cap_flags := MOVE_FLAG_PROMOTE if rank_of(to_sq) == promotion_rank else 0
			moves.append(encode_move(sq, to_sq, cap_flags))

		if en_passant_square != -1:
			var ep_mask := 1 << en_passant_square
			if pawn_attacks[sq] & ep_mask:
				moves.append(encode_move(sq, en_passant_square, MOVE_FLAG_EN_PASSANT))


func _add_knight_moves(knights: int, friendly_occ: int, moves: PackedInt32Array) -> void:
	var bb := knights
	while bb != 0:
		var sq := _bit_scan_forward(bb)
		bb &= bb - 1
		var attacks := KNIGHT_ATTACKS[sq] & ~friendly_occ
		while attacks != 0:
			var to_sq := _bit_scan_forward(attacks)
			attacks &= attacks - 1
			moves.append(encode_move(sq, to_sq))


func _add_slider_moves(sliders: int, occupancy: int, friendly_occ: int, moves: PackedInt32Array, is_diagonal: bool) -> void:
	var bb := sliders
	while bb != 0:
		var sq := _bit_scan_forward(bb)
		bb &= bb - 1
		var attacks := _slider_attacks(sq, occupancy, is_diagonal)
		attacks &= ~friendly_occ
		while attacks != 0:
			var to_sq := _bit_scan_forward(attacks)
			attacks &= attacks - 1
			moves.append(encode_move(sq, to_sq))


func _add_king_moves(king: int, friendly_occ: int, enemy_occ: int, occupancy: int, moves: PackedInt32Array) -> void:
	if king == 0:
		return
	var sq := _bit_scan_forward(king)
	var attacks := KING_ATTACKS[sq] & ~friendly_occ
	while attacks != 0:
		var to_sq := _bit_scan_forward(attacks)
		attacks &= attacks - 1
		moves.append(encode_move(sq, to_sq))

	# Castling
	if side_to_move == 0 and sq == 4:
		if (
			castling_rights & 0b0001
			and _bit_at(occupancy, 5) == 0
			and _bit_at(occupancy, 6) == 0
			and _bit_at(bb_w_rooks, 7) == 1
		):
			if not _is_square_attacked(4, 1) and not _is_square_attacked(5, 1):
				moves.append(encode_move(4, 6))
		if (
			castling_rights & 0b0010
			and _bit_at(occupancy, 3) == 0
			and _bit_at(occupancy, 2) == 0
			and _bit_at(occupancy, 1) == 0
			and _bit_at(bb_w_rooks, 0) == 1
		):
			if not _is_square_attacked(4, 1) and not _is_square_attacked(3, 1):
				moves.append(encode_move(4, 2))
	elif side_to_move == 1 and sq == 60:
		if (
			castling_rights & 0b0100
			and _bit_at(occupancy, 61) == 0
			and _bit_at(occupancy, 62) == 0
			and _bit_at(bb_b_rooks, 63) == 1
		):
			if not _is_square_attacked(60, 0) and not _is_square_attacked(61, 0):
				moves.append(encode_move(60, 62))
		if (
			castling_rights & 0b1000
			and _bit_at(occupancy, 59) == 0
			and _bit_at(occupancy, 58) == 0
			and _bit_at(occupancy, 57) == 0
			and _bit_at(bb_b_rooks, 56) == 1
		):
			if not _is_square_attacked(60, 0) and not _is_square_attacked(59, 0):
				moves.append(encode_move(60, 58))


func _apply_move_in_place(move: int) -> void:
	var from := move_from(move)
	var to := move_to(move)
	var flags := move_flags(move)
	var piece := _piece_at(from)
	var abs_p := absi(piece)
	var captured := _piece_at(to)

	# En passant capture
	if abs_p == Piece.PAWN and (flags & MOVE_FLAG_EN_PASSANT) != 0:
		var ep_sq := to + (-8 if side_to_move == 0 else 8)
		captured = _piece_at(ep_sq)
		_remove_piece_at(ep_sq, captured)

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
			var rook_from := to + 1
			var rook_to := to - 1
			var rook_piece := _piece_at(rook_from)
			_remove_piece_at(rook_from, rook_piece)
			_set_piece_at(rook_to, rook_piece)
		elif from - to == 2:  # Queenside
			var rook_from_q := to - 2
			var rook_to_q := to + 1
			var rook_piece_q := _piece_at(rook_from_q)
			_remove_piece_at(rook_from_q, rook_piece_q)
			_set_piece_at(rook_to_q, rook_piece_q)

	# Update castling rights
	if abs_p == Piece.KING:
		if side_to_move == 0: castling_rights &= 0b1100
		else: castling_rights &= 0b0011
	if from == 0 or to == 0: castling_rights &= ~0b0010
	if from == 7 or to == 7: castling_rights &= ~0b0001
	if from == 56 or to == 56: castling_rights &= ~0b1000
	if from == 63 or to == 63: castling_rights &= ~0b0100

	# Move piece
	_remove_piece_at(from, piece)
	if captured != 0:
		_remove_piece_at(to, captured)
	_set_piece_at(to, piece)

	# Update king cache
	if abs_p == Piece.KING:
		if side_to_move == 0:
			_white_king_sq = to
		else:
			_black_king_sq = to

	# Pawn promotion (auto-queen)
	if abs_p == Piece.PAWN:
		if (side_to_move == 0 and rank_of(to) == 7) or (side_to_move == 1 and rank_of(to) == 0):
			_remove_piece_at(to, piece)
			_set_piece_at(to, Piece.QUEEN if side_to_move == 0 else -Piece.QUEEN)

	# Switch side
	side_to_move = 1 - side_to_move
	if side_to_move == 0:
		fullmove_number += 1


func _is_square_attacked(sq: int, by_color: int) -> bool:
	var enemy_occ := _black_occ() if by_color == 1 else _white_occ()
	var friendly_occ := _white_occ() if by_color == 1 else _black_occ()
	var occupancy := enemy_occ | friendly_occ

	var attackers := 0
	if by_color == 0:
		attackers = PAWN_ATTACKS_B[sq] & bb_w_pawns
		if attackers != 0: return true
		if KNIGHT_ATTACKS[sq] & bb_w_knights: return true
		if KING_ATTACKS[sq] & bb_w_king: return true
		if _slider_attacks(sq, occupancy, true) & (bb_w_bishops | bb_w_queens): return true
		if _slider_attacks(sq, occupancy, false) & (bb_w_rooks | bb_w_queens): return true
	else:
		attackers = PAWN_ATTACKS_W[sq] & bb_b_pawns
		if attackers != 0: return true
		if KNIGHT_ATTACKS[sq] & bb_b_knights: return true
		if KING_ATTACKS[sq] & bb_b_king: return true
		if _slider_attacks(sq, occupancy, true) & (bb_b_bishops | bb_b_queens): return true
		if _slider_attacks(sq, occupancy, false) & (bb_b_rooks | bb_b_queens): return true

	return false


func _slider_attacks(sq: int, occupancy: int, diagonal: bool) -> int:
	var attacks := 0
	if diagonal:
		attacks |= _ray_attacks(sq, occupancy, RAY_NE, true)
		attacks |= _ray_attacks(sq, occupancy, RAY_NW, true)
		attacks |= _ray_attacks(sq, occupancy, RAY_SE, false)
		attacks |= _ray_attacks(sq, occupancy, RAY_SW, false)
	else:
		attacks |= _ray_attacks(sq, occupancy, RAY_N, true)
		attacks |= _ray_attacks(sq, occupancy, RAY_E, true)
		attacks |= _ray_attacks(sq, occupancy, RAY_S, false)
		attacks |= _ray_attacks(sq, occupancy, RAY_W, false)
	return attacks


func _ray_attacks(sq: int, occupancy: int, ray_table: Array[int], positive: bool) -> int:
	var attacks := ray_table[sq]
	var blockers := attacks & occupancy
	if blockers != 0:
		var blocker_sq := _bit_scan_forward(blockers) if positive else _bit_scan_reverse(blockers)
		attacks &= ~ray_table[blocker_sq]
	return attacks


func _white_occ() -> int:
	return bb_w_pawns | bb_w_knights | bb_w_bishops | bb_w_rooks | bb_w_queens | bb_w_king

func _black_occ() -> int:
	return bb_b_pawns | bb_b_knights | bb_b_bishops | bb_b_rooks | bb_b_queens | bb_b_king


func _piece_at(sq: int) -> int:
	var mask := 1 << sq
	if bb_w_pawns & mask: return Piece.PAWN
	if bb_w_knights & mask: return Piece.KNIGHT
	if bb_w_bishops & mask: return Piece.BISHOP
	if bb_w_rooks & mask: return Piece.ROOK
	if bb_w_queens & mask: return Piece.QUEEN
	if bb_w_king & mask: return Piece.KING
	if bb_b_pawns & mask: return -Piece.PAWN
	if bb_b_knights & mask: return -Piece.KNIGHT
	if bb_b_bishops & mask: return -Piece.BISHOP
	if bb_b_rooks & mask: return -Piece.ROOK
	if bb_b_queens & mask: return -Piece.QUEEN
	if bb_b_king & mask: return -Piece.KING
	return 0


func _set_piece_at(sq: int, piece: int) -> void:
	var mask := 1 << sq
	match piece:
		Piece.PAWN: bb_w_pawns |= mask
		Piece.KNIGHT: bb_w_knights |= mask
		Piece.BISHOP: bb_w_bishops |= mask
		Piece.ROOK: bb_w_rooks |= mask
		Piece.QUEEN: bb_w_queens |= mask
		Piece.KING:
			bb_w_king |= mask
			_white_king_sq = sq
			
		-Piece.PAWN: bb_b_pawns |= mask
		-Piece.KNIGHT: bb_b_knights |= mask
		-Piece.BISHOP: bb_b_bishops |= mask
		-Piece.ROOK: bb_b_rooks |= mask
		-Piece.QUEEN: bb_b_queens |= mask
		-Piece.KING:
			bb_b_king |= mask
			_black_king_sq = sq


func _remove_piece_at(sq: int, piece: int) -> void:
	if piece == 0:
		return
	var mask := ~(1 << sq)
	match piece:
		Piece.PAWN: bb_w_pawns &= mask
		Piece.KNIGHT: bb_w_knights &= mask
		Piece.BISHOP: bb_w_bishops &= mask
		Piece.ROOK: bb_w_rooks &= mask
		Piece.QUEEN: bb_w_queens &= mask
		Piece.KING: bb_w_king &= mask
		-Piece.PAWN: bb_b_pawns &= mask
		-Piece.KNIGHT: bb_b_knights &= mask
		-Piece.BISHOP: bb_b_bishops &= mask
		-Piece.ROOK: bb_b_rooks &= mask
		-Piece.QUEEN: bb_b_queens &= mask
		-Piece.KING: bb_b_king &= mask


func _find_king(color: int) -> int:
	return _white_king_sq if color == 0 else _black_king_sq


static func _bit_scan_forward(bb: int) -> int:
	for i in range(64):
		if bb & (1 << i):
			return i
	return -1


static func _bit_scan_reverse(bb: int) -> int:
	for i in range(63, -1, -1):
		if bb & (1 << i):
			return i
	return -1


static func _popcount(bb: int) -> int:
	var count := 0
	for i in range(64):
		if (bb >> i) & 1:
			count += 1
	return count


static func _bit_at(bb: int, sq: int) -> int:
	return 1 if (bb & (1 << sq)) != 0 else 0


static func _compute_knight_attacks(sq: int) -> int:
	var f := file_of(sq)
	var r := rank_of(sq)
	var attacks := 0
	var offsets := [
		Vector2i(-2, -1), Vector2i(-2, 1), Vector2i(-1, -2), Vector2i(-1, 2),
		Vector2i(1, -2), Vector2i(1, 2), Vector2i(2, -1), Vector2i(2, 1)
	]
	for off in offsets:
		var nf: int = f + off.x
		var nr: int = r + off.y
		if nf < 0 or nf > 7 or nr < 0 or nr > 7:
			continue
		attacks |= 1 << square(nf, nr)
	return attacks


static func _compute_king_attacks(sq: int) -> int:
	var f := file_of(sq)
	var r := rank_of(sq)
	var attacks := 0
	for df in [-1, 0, 1]:
		for dr in [-1, 0, 1]:
			if df == 0 and dr == 0:
				continue
			var nf: int = f + df
			var nr: int = r + dr
			if nf < 0 or nf > 7 or nr < 0 or nr > 7:
				continue
			attacks |= 1 << square(nf, nr)
	return attacks


static func _compute_pawn_attacks(sq: int, color: int) -> int:
	var f := file_of(sq)
	var r := rank_of(sq)
	var attacks := 0
	var dir := 1 if color == 0 else -1
	var nr := r + dir
	if nr < 0 or nr > 7:
		return attacks
	for df: int in [-1, 1]:
		var nf: int = f + df
		if nf < 0 or nf > 7:
			continue
		attacks |= 1 << square(nf, nr)
	return attacks


static func _compute_ray(sq: int, df: int, dr: int) -> int:
	var f := file_of(sq)
	var r := rank_of(sq)
	var attacks := 0
	var nf := f + df
	var nr := r + dr
	while nf >= 0 and nf <= 7 and nr >= 0 and nr <= 7:
		attacks |= 1 << square(nf, nr)
		nf += df
		nr += dr
	return attacks
