class_name EndgameHints
extends RefCounted

## Endgame evaluation hints for chess positions.
## Only activates when total non-king pieces <= 6.
## Provides bonuses to guide the AI in known endgame patterns.

const Piece = ChessConstants.Piece


## Returns {type: String, advantage: int (0=white, 1=black, -1=none), eval_bonus: float}
## or empty dict if not a recognized endgame.
static func evaluate_endgame(state) -> Dictionary:
	var pieces := count_pieces(state)
	var total: int = pieces.white_total + pieces.black_total
	if total > 6:
		return {}

	var w: Array = pieces.white
	var b: Array = pieces.black

	# K vs K
	if w.is_empty() and b.is_empty():
		return {type = "known_draw", advantage = -1, eval_bonus = 0.0}

	# One side has pieces, other has none
	if b.is_empty():
		# White has pieces vs bare black king
		return _evaluate_lone_king(state, w, 0)
	if w.is_empty():
		# Black has pieces vs bare white king
		var result := _evaluate_lone_king(state, b, 1)
		# Negate eval_bonus since it's computed for the side with pieces
		result.eval_bonus = -result.eval_bonus
		return result

	# General endgame: amplify material difference
	var white_mat: float = state.material_score(0)
	var black_mat: float = state.material_score(1)
	var mat_diff: float = white_mat - black_mat
	var bonus: float = mat_diff * 0.5
	var adv: int = -1
	if mat_diff > 0.0:
		adv = 0
	elif mat_diff < 0.0:
		adv = 1
	return {type = "general_endgame", advantage = adv, eval_bonus = bonus}


static func _evaluate_lone_king(state, pieces_list: Array, color: int) -> Dictionary:
	# color is the side WITH pieces (0=white, 1=black)
	var winning_king_sq: int = state.white_king_sq if color == 0 else state.black_king_sq
	var losing_king_sq: int = state.black_king_sq if color == 0 else state.white_king_sq

	# Check for known draws first
	if pieces_list.size() == 1:
		var p: int = pieces_list[0]
		if p == Piece.BISHOP or p == Piece.KNIGHT:
			return {type = "known_draw", advantage = -1, eval_bonus = 0.0}

	# K+Q vs K
	if pieces_list.size() == 1 and pieces_list[0] == Piece.QUEEN:
		var corner_dist := king_corner_distance(losing_king_sq)
		var king_prox := 7.0 - float(square_distance(winning_king_sq, losing_king_sq))
		var bonus := (7.0 - corner_dist) * 1.0 + king_prox * 0.5 + 5.0
		return {type = "KQ_vs_K", advantage = color, eval_bonus = bonus}

	# K+R vs K
	if pieces_list.size() == 1 and pieces_list[0] == Piece.ROOK:
		var corner_dist := king_corner_distance(losing_king_sq)
		var king_prox := 7.0 - float(square_distance(winning_king_sq, losing_king_sq))
		var bonus := (7.0 - corner_dist) * 1.0 + king_prox * 0.5 + 5.0
		return {type = "KR_vs_K", advantage = color, eval_bonus = bonus}

	# K+B+N vs K
	if pieces_list.size() == 2:
		var sorted := pieces_list.duplicate()
		sorted.sort()
		if sorted[0] == Piece.KNIGHT and sorted[1] == Piece.BISHOP:
			var corner_dist := king_corner_distance(losing_king_sq)
			var king_prox := 7.0 - float(square_distance(winning_king_sq, losing_king_sq))
			var bonus := (7.0 - corner_dist) * 0.5 + king_prox * 0.25 + 2.0
			return {type = "KBN_vs_K", advantage = color, eval_bonus = bonus}

	# K+P vs K
	if pieces_list.size() == 1 and pieces_list[0] == Piece.PAWN:
		# Find the pawn square
		var pawn_sq := -1
		var sign := 1 if color == 0 else -1
		for sq in range(64):
			if state.board[sq] == sign * Piece.PAWN:
				pawn_sq = sq
				break
		var pawn_rank: int = pawn_sq / 8
		var advancement: float
		if color == 0:
			advancement = float(pawn_rank) / 7.0
		else:
			advancement = float(7 - pawn_rank) / 7.0
		var bonus := advancement * 5.0 + 2.0
		return {type = "KP_vs_K", advantage = color, eval_bonus = bonus}

	# Fallback for unrecognized piece combo with lone king opponent
	var corner_dist := king_corner_distance(losing_king_sq)
	var king_prox := 7.0 - float(square_distance(winning_king_sq, losing_king_sq))
	var bonus := (7.0 - corner_dist) * 0.5 + king_prox * 0.25 + 3.0
	return {type = "winning_endgame", advantage = color, eval_bonus = bonus}


## Count non-king pieces per color.
## Returns {white: Array[int], black: Array[int], white_total: int, black_total: int}
## where white/black arrays contain piece types present (e.g., [1, 4] = pawn + rook)
static func count_pieces(state) -> Dictionary:
	var white: Array[int] = []
	var black: Array[int] = []
	for sq in range(64):
		var p: int = state.board[sq]
		if p > 0 and p != Piece.KING:
			white.append(p)
		elif p < 0 and absi(p) != Piece.KING:
			black.append(absi(p))
	return {white = white, black = black, white_total = white.size(), black_total = black.size()}


## Check if position is a known theoretical draw.
static func is_known_draw(state) -> bool:
	var pieces := count_pieces(state)
	var total: int = pieces.white_total + pieces.black_total
	if total > 6:
		return false

	# K vs K
	if pieces.white.is_empty() and pieces.black.is_empty():
		return true

	# K+B vs K or K+N vs K (either side)
	if pieces.white.is_empty() and pieces.black.size() == 1:
		var p: int = pieces.black[0]
		if p == Piece.BISHOP or p == Piece.KNIGHT:
			return true
	if pieces.black.is_empty() and pieces.white.size() == 1:
		var p: int = pieces.white[0]
		if p == Piece.BISHOP or p == Piece.KNIGHT:
			return true

	return false


## Distance from square to nearest corner (0-3 scale, Chebyshev distance).
static func king_corner_distance(sq: int) -> float:
	var f := sq % 8
	var r := sq / 8
	var dist_to_corners := [
		maxi(f, r),           # distance to a1 corner
		maxi(7 - f, r),       # distance to h1 corner
		maxi(f, 7 - r),       # distance to a8 corner
		maxi(7 - f, 7 - r),   # distance to h8 corner
	]
	return float(dist_to_corners.min())


## Chebyshev distance between two squares.
static func square_distance(sq1: int, sq2: int) -> int:
	var f1 := sq1 % 8
	var r1 := sq1 / 8
	var f2 := sq2 % 8
	var r2 := sq2 / 8
	return maxi(absi(f1 - f2), absi(r1 - r2))
