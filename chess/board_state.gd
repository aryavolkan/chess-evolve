class_name BoardState
extends RefCounted

## Represents a chess board state with full game logic.
## Board is an 8x8 array: positive = white, negative = black.
## Values correspond to ChessConstants.Piece enum.

const PIECE = ChessConstants.Piece

# Precomputed tables — static so they're allocated once at class load, never per-call.
static var _knight_offsets: Array[Vector2i] = [
    Vector2i(-2, -1), Vector2i(-2, 1), Vector2i(-1, -2), Vector2i(-1, 2),
    Vector2i(1, -2), Vector2i(1, 2), Vector2i(2, -1), Vector2i(2, 1)
]
static var _diag_dirs: Array[Vector2i] = [
    Vector2i(-1, -1), Vector2i(-1, 1), Vector2i(1, -1), Vector2i(1, 1)
]
static var _straight_dirs: Array[Vector2i] = [
    Vector2i(0, -1), Vector2i(0, 1), Vector2i(-1, 0), Vector2i(1, 0)
]
static var _all_dirs: Array[Vector2i] = [
    Vector2i(-1, -1), Vector2i(-1, 1), Vector2i(1, -1), Vector2i(1, 1),
    Vector2i(0, -1), Vector2i(0, 1), Vector2i(-1, 0), Vector2i(1, 0)
]

# Zobrist-style hash table: random numbers for each (piece, square) pair.
# Piece values range -6..+6 (0=empty), mapped to index 0..12 via piece+6.
# Table[piece_idx * 64 + square] gives the hash contribution.
# Uses deterministic seed so hash values are consistent across instances.
static var _zobrist_table: PackedInt64Array = _init_zobrist()
static var _zobrist_side: int = 0  # XOR'd when side_to_move == 1
static var _zobrist_castling: PackedInt64Array = PackedInt64Array()  # 16 entries
static var _zobrist_ep: PackedInt64Array = PackedInt64Array()  # 65 entries (-1..63 -> 0..64)

static func _init_zobrist() -> PackedInt64Array:
    var rng := RandomNumberGenerator.new()
    rng.seed = 0xDEADBEEFC4E55E70  # deterministic
    var table := PackedInt64Array()
    table.resize(13 * 64)  # 13 piece types (-6..+6) x 64 squares
    for i in table.size():
        table[i] = rng.randi() | (rng.randi() << 32)
    _zobrist_side = rng.randi() | (rng.randi() << 32)
    _zobrist_castling.resize(16)
    for i in 16:
        _zobrist_castling[i] = rng.randi() | (rng.randi() << 32)
    _zobrist_ep.resize(65)
    for i in 65:
        _zobrist_ep[i] = rng.randi() | (rng.randi() << 32)
    return table

var board: Array[int] = []  # 64 squares, row-major (0=a1, 63=h8)
var side_to_move: int = 0  # 0=white, 1=black
var castling_rights: int = 0b1111  # KQkq bits
var en_passant_square: int = -1  # target square or -1
var halfmove_clock: int = 0
var fullmove_number: int = 1
var is_game_over: bool = false
var result: int = 0  # 0=ongoing, 1=white wins, -1=black wins, 2=draw
var white_king_sq: int = -1
var black_king_sq: int = -1

var use_bitboard_movegen: bool = false
var _legal_moves_cache: PackedInt32Array = PackedInt32Array()
var legal_moves_dirty: bool = true
var encoder_cache: PackedFloat32Array = PackedFloat32Array()
var encoder_dirty: bool = true


func _init() -> void:
    board.resize(64)
    board.fill(0)
    mark_dirty()


func setup_initial() -> void:
    ## Set up the standard starting position.
    board.fill(0)
    # White pieces (rank 1 = indices 0-7)
    board[0] = PIECE.ROOK; board[1] = PIECE.KNIGHT; board[2] = PIECE.BISHOP
    board[3] = PIECE.QUEEN; board[4] = PIECE.KING
    board[5] = PIECE.BISHOP; board[6] = PIECE.KNIGHT; board[7] = PIECE.ROOK
    for i in range(8, 16):
        board[i] = PIECE.PAWN
    # Black pieces (rank 8 = indices 56-63)
    board[56] = -PIECE.ROOK; board[57] = -PIECE.KNIGHT; board[58] = -PIECE.BISHOP
    board[59] = -PIECE.QUEEN; board[60] = -PIECE.KING
    board[61] = -PIECE.BISHOP; board[62] = -PIECE.KNIGHT; board[63] = -PIECE.ROOK
    for i in range(48, 56):
        board[i] = -PIECE.PAWN
    side_to_move = 0
    castling_rights = 0b1111
    en_passant_square = -1
    halfmove_clock = 0
    fullmove_number = 1
    is_game_over = false
    result = 0
    white_king_sq = 4
    black_king_sq = 60
    mark_dirty()


func mark_dirty() -> void:
    legal_moves_dirty = true
    encoder_dirty = true


func get_position_key() -> int:
    ## Zobrist hash of the position — O(pieces) XOR instead of string allocation.
    ## Returns an int suitable as a Dictionary key or for repetition detection.
    var h: int = 0
    for sq in 64:
        var p: int = board[sq]
        if p != 0:
            h ^= _zobrist_table[(p + 6) * 64 + sq]
    if side_to_move == 1:
        h ^= _zobrist_side
    h ^= _zobrist_castling[castling_rights & 0xF]
    h ^= _zobrist_ep[en_passant_square + 1]  # -1 maps to index 0
    return h


func rebuild_piece_lists() -> void:
    white_king_sq = -1
    black_king_sq = -1
    for sq in range(64):
        var p := board[sq]
        if p == PIECE.KING:
            white_king_sq = sq
        elif p == -PIECE.KING:
            black_king_sq = sq
    mark_dirty()


static func file_of(sq: int) -> int:
    return sq % 8

static func rank_of(sq: int) -> int:
    return sq / 8

static func square(file: int, rank: int) -> int:
    return rank * 8 + file

static func is_valid_square(sq: int) -> bool:
    return sq >= 0 and sq < 64


static func encode_move(from_sq: int, to_sq: int) -> int:
    return (from_sq << 6) | to_sq


static func move_from(encoded_move: int) -> int:
    return encoded_move >> 6


static func move_to(encoded_move: int) -> int:
    return encoded_move & 0x3f


static func move_to_vec(encoded_move: int) -> Vector2i:
    return Vector2i(move_from(encoded_move), move_to(encoded_move))


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
    copy.white_king_sq = white_king_sq
    copy.black_king_sq = black_king_sq
    copy.use_bitboard_movegen = use_bitboard_movegen
    copy.legal_moves_dirty = true
    copy.encoder_dirty = true
    return copy


func generate_legal_moves() -> PackedInt32Array:
    ## Returns packed int moves: (from << 6) | to.
    if not legal_moves_dirty:
        return _legal_moves_cache

    var moves := PackedInt32Array()
    if use_bitboard_movegen:
        var bb := BitboardState.from_board_state(self)
        var bb_moves := bb.get_legal_moves()
        moves.resize(bb_moves.size())
        for i in bb_moves.size():
            moves[i] = bb_moves[i] & 0xFFF
    else:
        var pseudo := _generate_pseudo_legal_moves()
        var king_sq := _find_king(side_to_move)  # Find once; re-used by every _is_legal() call.
        for m in pseudo:
            if _is_legal(m, king_sq):
                moves.append(m)

    _legal_moves_cache = moves
    legal_moves_dirty = false
    return moves


func benchmark_generate_legal_moves(iterations: int = 1000) -> void:
    var state := BoardState.new()
    state.setup_initial()
    var start := Time.get_ticks_usec()
    for i in iterations:
        state.generate_legal_moves()
    var elapsed := Time.get_ticks_usec() - start
    var avg := float(elapsed) / float(iterations)
    print("generate_legal_moves %d %d usec total, %s usec/call" % [iterations, elapsed, avg])


func _generate_pseudo_legal_moves() -> PackedInt32Array:
    var moves := PackedInt32Array()
    for sq in range(64):
        if (side_to_move == 0 and board[sq] > 0) or (side_to_move == 1 and board[sq] < 0):
            add_piece_moves(sq, moves)
    return moves


func add_piece_moves(sq: int, moves: PackedInt32Array) -> void:
    var piece := abs_piece(sq)
    match piece:
        PIECE.PAWN: _add_pawn_moves(sq, moves)
        PIECE.KNIGHT: _add_knight_moves(sq, moves)
        PIECE.BISHOP: _add_slider_moves(sq, moves, _diag_dirs)
        PIECE.ROOK: _add_slider_moves(sq, moves, _straight_dirs)
        PIECE.QUEEN: _add_slider_moves(sq, moves, _all_dirs)
        PIECE.KING: _add_king_moves(sq, moves)


func _add_pawn_moves(sq: int, moves: PackedInt32Array) -> void:
    var dir := 1 if side_to_move == 0 else -1
    var start_rank := 1 if side_to_move == 0 else 6
    var f := file_of(sq)
    var r := rank_of(sq)

    # Forward one
    var fwd := square(f, r + dir)
    if is_valid_square(fwd) and board[fwd] == 0:
        moves.append(encode_move(sq, fwd))
        # Forward two from starting rank
        if r == start_rank:
            var fwd2 := square(f, r + 2 * dir)
            if board[fwd2] == 0:
                moves.append(encode_move(sq, fwd2))

    # Captures
    for df in [-1, 1]:
        var nf: int = f + df
        if nf < 0 or nf > 7: continue
        var cap_sq := square(nf, r + dir)
        if not is_valid_square(cap_sq): continue
        if is_enemy(cap_sq):
            moves.append(encode_move(sq, cap_sq))
        elif cap_sq == en_passant_square:
            moves.append(encode_move(sq, cap_sq))


func _add_knight_moves(sq: int, moves: PackedInt32Array) -> void:
    var f := file_of(sq)
    var r := rank_of(sq)
    for off in _knight_offsets:
        var nf: int = f + off.x
        var nr: int = r + off.y
        if nf < 0 or nf > 7 or nr < 0 or nr > 7: continue
        var target := square(nf, nr)
        if not is_friendly(target):
            moves.append(encode_move(sq, target))


func _add_slider_moves(sq: int, moves: PackedInt32Array, directions: Array) -> void:
    var f := file_of(sq)
    var r := rank_of(sq)
    var side_sign := 1 if side_to_move == 0 else -1
    for dir: Vector2i in directions:
        var nf := f + dir.x
        var nr := r + dir.y
        while nf >= 0 and nf <= 7 and nr >= 0 and nr <= 7:
            var target := square(nf, nr)
            var p := board[target]
            if p == 0:
                moves.append(encode_move(sq, target))
            elif p * side_sign > 0:
                break
            else:
                moves.append(encode_move(sq, target))
                break
            nf += dir.x
            nr += dir.y


func _add_king_moves(sq: int, moves: PackedInt32Array) -> void:
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
                moves.append(encode_move(sq, target))

    # Castling
    if side_to_move == 0 and sq == 4:
        if (
            castling_rights & 0b0001
            and board[5] == 0
            and board[6] == 0
            and board[7] == PIECE.ROOK
        ):
            if not _is_square_attacked(4, 1) and not _is_square_attacked(5, 1):
                moves.append(encode_move(4, 6))
        if (
            castling_rights & 0b0010
            and board[3] == 0
            and board[2] == 0
            and board[1] == 0
            and board[0] == PIECE.ROOK
        ):
            if not _is_square_attacked(4, 1) and not _is_square_attacked(3, 1):
                moves.append(encode_move(4, 2))
    elif side_to_move == 1 and sq == 60:
        if (
            castling_rights & 0b0100
            and board[61] == 0
            and board[62] == 0
            and board[63] == -PIECE.ROOK
        ):
            if not _is_square_attacked(60, 0) and not _is_square_attacked(61, 0):
                moves.append(encode_move(60, 62))
        if (
            castling_rights & 0b1000
            and board[59] == 0
            and board[58] == 0
            and board[57] == 0
            and board[56] == -PIECE.ROOK
        ):
            if not _is_square_attacked(60, 0) and not _is_square_attacked(59, 0):
                moves.append(encode_move(60, 58))


func _is_legal(move: int, king_sq: int) -> bool:
    ## Check legality via in-place make/unmake — avoids cloning the entire board.
    ## Equivalent to the original clone-based approach but allocates nothing.
    var from := move_from(move)
    var to := move_to(move)
    var piece := board[from]
    var abs_p := absi(piece)
    var captured := board[to]

    # Handle en passant capture.
    var ep_sq := -1
    var ep_captured := 0
    if abs_p == PIECE.PAWN and to == en_passant_square:
        ep_sq = to + (-8 if side_to_move == 0 else 8)
        ep_captured = board[ep_sq]
        board[ep_sq] = 0

    # Handle castling: temporarily move the rook.
    var rook_from := -1
    var rook_to_sq := -1
    var rook_piece := 0
    if abs_p == PIECE.KING:
        if to - from == 2:       # Kingside
            rook_from = to + 1; rook_to_sq = to - 1
        elif from - to == 2:     # Queenside
            rook_from = to - 2; rook_to_sq = to + 1
        if rook_from != -1:
            rook_piece = board[rook_from]
            board[rook_to_sq] = rook_piece
            board[rook_from] = 0

    # Apply piece move (with pawn promotion to queen for attack purposes).
    board[to] = piece
    board[from] = 0
    if abs_p == PIECE.PAWN:
        if (side_to_move == 0 and rank_of(to) == 7) or (side_to_move == 1 and rank_of(to) == 0):
            board[to] = PIECE.QUEEN if side_to_move == 0 else -PIECE.QUEEN

    # The king is on `to` if it just moved, otherwise it stays at king_sq.
    var actual_king_sq := to if abs_p == PIECE.KING else king_sq
    var legal := actual_king_sq != -1 and not _is_square_attacked(actual_king_sq, 1 - side_to_move)

    # Undo all board changes.
    board[from] = piece
    board[to] = captured
    if ep_sq != -1:
        board[ep_sq] = ep_captured
    if rook_from != -1:
        board[rook_from] = rook_piece
        board[rook_to_sq] = 0

    return legal


func _find_king(color: int) -> int:
    return white_king_sq if color == 0 else black_king_sq


func _is_square_attacked(sq: int, by_color: int) -> bool:
    ## Check if 'sq' is attacked by any piece of 'by_color'.
    var sign := 1 if by_color == 0 else -1
    var f := file_of(sq)
    var r := rank_of(sq)

    # Knight attacks
    for off in _knight_offsets:
        var nf: int = f + off.x
        var nr: int = r + off.y
        if nf >= 0 and nf <= 7 and nr >= 0 and nr <= 7:
            if board[square(nf, nr)] == sign * PIECE.KNIGHT: return true

    # Pawn attacks
    var pawn_dir := -1 if by_color == 0 else 1  # Pawns attack from behind
    for df in [-1, 1]:
        var nf: int = f + df
        var nr: int = r + pawn_dir
        if nf >= 0 and nf <= 7 and nr >= 0 and nr <= 7:
            if board[square(nf, nr)] == sign * PIECE.PAWN: return true

    # King attacks
    for df in [-1, 0, 1]:
        for dr in [-1, 0, 1]:
            if df == 0 and dr == 0: continue
            var nf: int = f + df
            var nr: int = r + dr
            if nf >= 0 and nf <= 7 and nr >= 0 and nr <= 7:
                if board[square(nf, nr)] == sign * PIECE.KING: return true

    # Slider attacks (bishop/queen diagonals, rook/queen straights)
    for dir in _diag_dirs:
        var nf: int = f + dir.x
        var nr: int = r + dir.y
        while nf >= 0 and nf <= 7 and nr >= 0 and nr <= 7:
            var p := board[square(nf, nr)]
            if p != 0:
                if p == sign * PIECE.BISHOP or p == sign * PIECE.QUEEN: return true
                break
            nf += dir.x; nr += dir.y

    for dir in _straight_dirs:
        var nf: int = f + dir.x
        var nr: int = r + dir.y
        while nf >= 0 and nf <= 7 and nr >= 0 and nr <= 7:
            var p := board[square(nf, nr)]
            if p != 0:
                if p == sign * PIECE.ROOK or p == sign * PIECE.QUEEN: return true
                break
            nf += dir.x; nr += dir.y

    return false


func make_move(move) -> void:
    ## Apply a legal move and update game state.
    _apply_move_unchecked(move)
    side_to_move = 1 - side_to_move
    if side_to_move == 0:
        fullmove_number += 1
    _check_game_over()


func _apply_move_unchecked(move) -> void:
    var from: int = move_from(move) if typeof(move) == TYPE_INT else move.x
    var to: int = move_to(move) if typeof(move) == TYPE_INT else move.y
    var piece := board[from]
    var abs_p := absi(piece)
    var captured := board[to]

    # En passant capture
    if abs_p == PIECE.PAWN and to == en_passant_square:
        var ep_captured_sq := to + (-8 if side_to_move == 0 else 8)
        board[ep_captured_sq] = 0

    # Update halfmove clock
    if abs_p == PIECE.PAWN or captured != 0:
        halfmove_clock = 0
    else:
        halfmove_clock += 1

    # En passant square
    en_passant_square = -1
    if abs_p == PIECE.PAWN and absi(rank_of(to) - rank_of(from)) == 2:
        en_passant_square = square(file_of(from), (rank_of(from) + rank_of(to)) / 2)

    # Castling: move rook
    if abs_p == PIECE.KING:
        if to - from == 2:  # Kingside
            board[to - 1] = board[to + 1]; board[to + 1] = 0
        elif from - to == 2:  # Queenside
            board[to + 1] = board[to - 2]; board[to - 2] = 0

    # Update castling rights
    if abs_p == PIECE.KING:
        if side_to_move == 0: castling_rights &= 0b1100
        else: castling_rights &= 0b0011
    if from == 0 or to == 0: castling_rights &= ~0b0010
    if from == 7 or to == 7: castling_rights &= ~0b0001
    if from == 56 or to == 56: castling_rights &= ~0b1000
    if from == 63 or to == 63: castling_rights &= ~0b0100

    # Move piece
    board[to] = piece
    board[from] = 0

    # Update king position cache
    if abs_p == PIECE.KING:
        if side_to_move == 0:
            white_king_sq = to
        else:
            black_king_sq = to

    # Pawn promotion (auto-queen)
    if abs_p == PIECE.PAWN:
        if (side_to_move == 0 and rank_of(to) == 7) or (side_to_move == 1 and rank_of(to) == 0):
            board[to] = PIECE.QUEEN if side_to_move == 0 else -PIECE.QUEEN

    mark_dirty()


func _check_game_over() -> void:
    # Check 50-move rule first — avoids the expensive generate_legal_moves() call.
    if halfmove_clock >= 100:
        is_game_over = true
        result = 2  # 50-move rule draw
        return
    var moves := generate_legal_moves()
    if moves.is_empty():
        is_game_over = true
        if is_in_check():
            result = 1 if side_to_move == 1 else -1  # Checkmate
        else:
            result = 2  # Stalemate


func is_in_check() -> bool:
    var king_sq := _find_king(side_to_move)
    if king_sq == -1: return false
    return _is_square_attacked(king_sq, 1 - side_to_move)


func material_score(color: int) -> float:
    ## Sum material value for given color.
    var total := 0.0
    for sq in range(64):
        var p: int = board[sq]
        if (color == 0 and p > 0) or (color == 1 and p < 0):
            total += ChessConstants.PIECE_VALUES_ARRAY[absi(p)]
    return total


func mobility_score(color: int) -> int:
    ## Count number of legal moves for a color.
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
                if p == sign * PIECE.PAWN:
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
