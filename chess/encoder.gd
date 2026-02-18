class_name ChessEncoder
extends RefCounted

## Encodes board state into neural network inputs and decodes outputs to moves.
##
## Input encoding (808 floats):
##   - 768: 12 piece planes × 64 squares (one-hot per piece type per color)
##   - 1: side to move (0=white, 1=black)
##   - 4: castling rights
##   - 1: en passant file (0-7 normalized, or -1 if none)
##   - 34: piece counts and material balance (padding to nice number)
## Simplified encoding (389 floats) used for MVP:
##   - 384: 6 piece types × 64 squares, signed (+1 white, -1 black)
##   - 1: side to move
##   - 4: castling rights
##   - Extra features not needed for MVP

const INPUT_SIZE := 389
const MOVE_OUTPUT_SIZE := 128  # from_square(64) + to_square(64)

const PIECE = ChessConstants.Piece


static func encode_board(state) -> PackedFloat32Array:
	## Encode board into neural network input vector.
	var inputs := PackedFloat32Array()
	inputs.resize(INPUT_SIZE)
	inputs.fill(0.0)

	var idx := 0
	# 6 piece-type planes × 64 squares (signed: +1 white, -1 black)
	for piece_type in range(1, 7):  # PAWN through KING
		for sq in range(64):
			var p: int = state.board[sq]
			if absi(p) == piece_type:
				inputs[idx] = 1.0 if p > 0 else -1.0
			idx += 1

	# Side to move
	inputs[idx] = 0.0 if state.side_to_move == 0 else 1.0
	idx += 1

	# Castling rights
	inputs[idx] = 1.0 if state.castling_rights & 0b0001 else 0.0; idx += 1
	inputs[idx] = 1.0 if state.castling_rights & 0b0010 else 0.0; idx += 1
	inputs[idx] = 1.0 if state.castling_rights & 0b0100 else 0.0; idx += 1
	inputs[idx] = 1.0 if state.castling_rights & 0b1000 else 0.0; idx += 1

	return inputs


static func decode_move(outputs: PackedFloat32Array, legal_moves: PackedInt32Array) -> int:
	## Decode network output into a legal move.
	## Output is split: first 64 = from-square preferences, next 64 = to-square preferences.
	## We score each legal move as from_score + to_score and pick the best.
	if legal_moves.is_empty():
		return -1

	var best_move := legal_moves[0]
	var best_score := -INF

	for move in legal_moves:
		var from_sq := BoardState.move_from(move)
		var to_sq := BoardState.move_to(move)
		var from_score: float = outputs[from_sq] if from_sq < outputs.size() else 0.0
		var to_score: float = outputs[64 + to_sq] if (64 + to_sq) < outputs.size() else 0.0
		var score := from_score + to_score
		if score > best_score:
			best_score = score
			best_move = move

	return best_move
