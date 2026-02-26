class_name ChessPGN
extends RefCounted

## Generates PGN (Portable Game Notation) from recorded move lists.
## Pure chess utility — no AI dependencies.

const BoardStateScript = preload("res://chess/board_state.gd")
const PIECE = ChessConstants.Piece


static func file_char(f: int) -> String:
	return char("a".unicode_at(0) + f)


static func square_name(sq: int) -> String:
	return file_char(sq % 8) + str(sq / 8 + 1)


static func piece_letter(abs_piece: int) -> String:
	match abs_piece:
		PIECE.KNIGHT: return "N"
		PIECE.BISHOP: return "B"
		PIECE.ROOK: return "R"
		PIECE.QUEEN: return "Q"
		PIECE.KING: return "K"
	return ""  # Pawns have no letter


static func move_to_san(state, encoded_move: int) -> String:
	## Convert an encoded move to Standard Algebraic Notation given the board BEFORE the move.
	var from_sq: int = BoardStateScript.move_from(encoded_move)
	var to_sq: int = BoardStateScript.move_to(encoded_move)
	var piece: int = state.board[from_sq]
	var abs_p: int = absi(piece)
	var captured: int = state.board[to_sq]

	# Castling
	if abs_p == PIECE.KING and absi(to_sq - from_sq) == 2:
		var san := "O-O" if to_sq > from_sq else "O-O-O"
		# Check/checkmate suffix
		var after := state.clone()
		after.make_move(encoded_move)
		san += _check_suffix(after)
		return san

	var san := ""

	# Piece letter (empty for pawns)
	san += piece_letter(abs_p)

	# Disambiguation for non-pawn pieces
	if abs_p != PIECE.PAWN:
		var disambig := _disambiguation(state, from_sq, to_sq, abs_p)
		san += disambig

	# Capture
	var is_capture := captured != 0 or (abs_p == PIECE.PAWN and to_sq == state.en_passant_square)
	if is_capture:
		if abs_p == PIECE.PAWN:
			san += file_char(from_sq % 8)
		san += "x"

	# Destination square
	san += square_name(to_sq)

	# Promotion
	if abs_p == PIECE.PAWN:
		var to_rank := to_sq / 8
		if to_rank == 7 or to_rank == 0:
			san += "=Q"

	# Check/checkmate suffix
	var after := state.clone()
	after.make_move(encoded_move)
	san += _check_suffix(after)

	return san


static func game_to_pgn(moves: PackedInt32Array, headers: Dictionary = {}) -> String:
	## Convert a full game to PGN format.
	var lines := PackedStringArray()

	# Default headers
	var h := {
		"Event": headers.get("Event", "Chess-Evolve Training"),
		"Site": headers.get("Site", "Chess-Evolve"),
		"Date": headers.get("Date", "????.??.??"),
		"Round": headers.get("Round", "?"),
		"White": headers.get("White", "Network-W"),
		"Black": headers.get("Black", "Network-B"),
		"Result": headers.get("Result", "*"),
	}
	for key in h:
		lines.append('[%s "%s"]' % [key, h[key]])
	lines.append("")

	# Replay and convert moves
	var state := BoardStateScript.new()
	state.setup_initial()
	var move_text := PackedStringArray()

	for i in moves.size():
		var move: int = moves[i]
		if state.is_game_over:
			break
		var san := move_to_san(state, move)
		if state.side_to_move == 0:
			move_text.append("%d. %s" % [state.fullmove_number, san])
		else:
			move_text.append(san)
		state.make_move(move)

	# Determine result
	var result_str: String = h["Result"]
	if result_str == "*" and state.is_game_over:
		match state.result:
			1: result_str = "1-0"
			-1: result_str = "0-1"
			2: result_str = "1/2-1/2"

	# Wrap move text at ~80 chars
	var current_line := ""
	for token in move_text:
		if current_line.length() + token.length() + 1 > 80 and not current_line.is_empty():
			lines.append(current_line)
			current_line = token
		elif current_line.is_empty():
			current_line = token
		else:
			current_line += " " + token
	if not current_line.is_empty():
		lines.append(current_line + " " + result_str)
	else:
		lines.append(result_str)

	return "\n".join(lines) + "\n"


static func _disambiguation(state, from_sq: int, to_sq: int, abs_piece: int) -> String:
	## Determine disambiguation string when multiple pieces of the same type can reach to_sq.
	var legal_moves := state.generate_legal_moves()
	var ambiguous_from: Array[int] = []
	for move in legal_moves:
		var m_from: int = BoardStateScript.move_from(move)
		var m_to: int = BoardStateScript.move_to(move)
		if m_to == to_sq and m_from != from_sq and absi(state.board[m_from]) == abs_piece:
			ambiguous_from.append(m_from)

	if ambiguous_from.is_empty():
		return ""

	var from_file := from_sq % 8
	var from_rank := from_sq / 8
	var need_file := true
	var need_rank := false

	# Check if file alone is sufficient
	for other_from in ambiguous_from:
		if other_from % 8 == from_file:
			need_rank = true
		if other_from / 8 == from_rank:
			need_file = true

	# If both file and rank are ambiguous, use both
	var same_file := false
	var same_rank := false
	for other_from in ambiguous_from:
		if other_from % 8 == from_file:
			same_file = true
		if other_from / 8 == from_rank:
			same_rank = true

	if same_file and same_rank:
		return file_char(from_file) + str(from_rank + 1)
	if same_file:
		return str(from_rank + 1)
	return file_char(from_file)


static func _check_suffix(state_after) -> String:
	## Returns "+", "#", or "" based on check/checkmate status after a move.
	if state_after.is_in_check():
		if state_after.generate_legal_moves().is_empty():
			return "#"
		return "+"
	return ""
