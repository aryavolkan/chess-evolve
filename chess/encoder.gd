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
const MOVE_OUTPUT_SIZE := 4096  # 64 from-squares × 64 to-squares

const PIECE = ChessConstants.Piece


static func encode_board(state) -> PackedFloat32Array:
    ## Encode board into neural network input vector.
    if state is BoardState:
        if not state.encoder_dirty and state.encoder_cache.size() == INPUT_SIZE:
            return state.encoder_cache

    var inputs := PackedFloat32Array()
    inputs.resize(INPUT_SIZE)
    inputs.fill(0.0)

    # Single-pass encoding: iterate board squares once, write to the correct
    # piece-type plane offset. This is O(64) instead of O(6*64) for the
    # original nested-loop approach — 6x fewer iterations.
    var board: Array[int] = state.board
    for sq in range(64):
        var p: int = board[sq]
        if p != 0:
            var plane := (absi(p) - 1) * 64  # piece_type - 1 maps to plane index
            inputs[plane + sq] = 1.0 if p > 0 else -1.0

    # Side to move (index 384)
    inputs[384] = 0.0 if state.side_to_move == 0 else 1.0

    # Castling rights (indices 385-388)
    inputs[385] = 1.0 if state.castling_rights & 0b0001 else 0.0
    inputs[386] = 1.0 if state.castling_rights & 0b0010 else 0.0
    inputs[387] = 1.0 if state.castling_rights & 0b0100 else 0.0
    inputs[388] = 1.0 if state.castling_rights & 0b1000 else 0.0

    if state is BoardState:
        state.encoder_cache = inputs
        state.encoder_dirty = false

    return inputs


static func decode_move(outputs: PackedFloat32Array, legal_moves: PackedInt32Array, temperature: float = 0.0) -> int:
    ## Decode network output into a legal move.
    ## Each from-to pair has a unique output: index = from_sq * 64 + to_sq.
    ## When temperature <= 0.0, uses argmax (deterministic). When > 0.0, uses
    ## softmax sampling for stochastic exploration.
    if legal_moves.is_empty():
        return -1

    # Collect scores for legal moves
    var scores := PackedFloat32Array()
    scores.resize(legal_moves.size())
    for i in legal_moves.size():
        var move := legal_moves[i]
        var from_sq := BoardState.move_from(move)
        var to_sq := BoardState.move_to(move)
        var idx := from_sq * 64 + to_sq
        scores[i] = outputs[idx] if idx < outputs.size() else 0.0

    # Argmax (temperature=0 or single move)
    if temperature <= 0.0 or legal_moves.size() == 1:
        var best_i := 0
        for i in range(1, scores.size()):
            if scores[i] > scores[best_i]:
                best_i = i
        return legal_moves[best_i]

    # Softmax sampling with temperature
    var max_score := scores[0]
    for i in range(1, scores.size()):
        max_score = maxf(max_score, scores[i])

    var inv_temp := 1.0 / temperature
    var exp_sum := 0.0
    var exp_vals := PackedFloat32Array()
    exp_vals.resize(scores.size())
    for i in scores.size():
        var v := exp((scores[i] - max_score) * inv_temp)
        exp_vals[i] = v
        exp_sum += v

    # Weighted random sample
    var r := randf() * exp_sum
    var acc := 0.0
    for i in exp_vals.size():
        acc += exp_vals[i]
        if r <= acc:
            return legal_moves[i]
    return legal_moves[legal_moves.size() - 1]
