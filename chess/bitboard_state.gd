class_name BitboardState
extends RefCounted

const Piece = ChessConstants.Piece  # gdlint:ignore = constant-name

const MOVE_FLAG_PROMOTE := 1 << 12
const MOVE_FLAG_EN_PASSANT := 1 << 13

static var knight_attacks: Array[int] = []
static var king_attacks: Array[int] = []
static var pawn_attacks_w: Array[int] = []
static var pawn_attacks_b: Array[int] = []

static var ray_n: Array[int] = []
static var ray_s: Array[int] = []
static var ray_e: Array[int] = []
static var ray_w: Array[int] = []
static var ray_ne: Array[int] = []
static var ray_nw: Array[int] = []
static var ray_se: Array[int] = []
static var ray_sw: Array[int] = []

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

var white_king_sq: int = -1
var black_king_sq: int = -1


static func _static_init() -> void:
    knight_attacks.resize(64)
    king_attacks.resize(64)
    pawn_attacks_w.resize(64)
    pawn_attacks_b.resize(64)
    ray_n.resize(64)
    ray_s.resize(64)
    ray_e.resize(64)
    ray_w.resize(64)
    ray_ne.resize(64)
    ray_nw.resize(64)
    ray_se.resize(64)
    ray_sw.resize(64)

    for sq in range(64):
        knight_attacks[sq] = _compute_knight_attacks(sq)
        king_attacks[sq] = _compute_king_attacks(sq)
        pawn_attacks_w[sq] = _compute_pawn_attacks(sq, 0)
        pawn_attacks_b[sq] = _compute_pawn_attacks(sq, 1)
        ray_n[sq] = _compute_ray(sq, 0, 1)
        ray_s[sq] = _compute_ray(sq, 0, -1)
        ray_e[sq] = _compute_ray(sq, 1, 0)
        ray_w[sq] = _compute_ray(sq, -1, 0)
        ray_ne[sq] = _compute_ray(sq, 1, 1)
        ray_nw[sq] = _compute_ray(sq, -1, 1)
        ray_se[sq] = _compute_ray(sq, 1, -1)
        ray_sw[sq] = _compute_ray(sq, -1, -1)


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

        state.set_piece_at(sq, p)

    state.white_king_sq = state.find_king(0)
    state.black_king_sq = state.find_king(1)
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
    copy.white_king_sq = white_king_sq
    copy.black_king_sq = black_king_sq
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
    next.apply_move_in_place(move)
    return next


func is_in_check(color: int) -> bool:
    var king_sq := find_king(color)
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


func _add_pawn_moves(
    pawns: int, color: int, friendly_occ: int, enemy_occ: int,
    moves: PackedInt32Array
) -> void:
    var dir := 8 if color == 0 else -8
    var start_rank := 1 if color == 0 else 6
    var promotion_rank := 7 if color == 0 else 0
    var pawn_attacks := pawn_attacks_w if color == 0 else pawn_attacks_b

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
        var attacks := knight_attacks[sq] & ~friendly_occ
        while attacks != 0:
            var to_sq := _bit_scan_forward(attacks)
            attacks &= attacks - 1
            moves.append(encode_move(sq, to_sq))


func _add_slider_moves(
    sliders: int, occupancy: int, friendly_occ: int,
    moves: PackedInt32Array, is_diagonal: bool
) -> void:
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


func _add_king_moves(king: int, friendly_occ: int, _enemy_occ: int, occupancy: int, moves: PackedInt32Array) -> void:
    if king == 0:
        return
    var sq := _bit_scan_forward(king)
    var attacks := king_attacks[sq] & ~friendly_occ
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
            if (
                not _is_square_attacked(4, 1)
                and not _is_square_attacked(5, 1)
                and not _is_square_attacked(6, 1)
            ):
                moves.append(encode_move(4, 6))
        if (
            castling_rights & 0b0010
            and _bit_at(occupancy, 3) == 0
            and _bit_at(occupancy, 2) == 0
            and _bit_at(occupancy, 1) == 0
            and _bit_at(bb_w_rooks, 0) == 1
        ):
            if (
                not _is_square_attacked(4, 1)
                and not _is_square_attacked(3, 1)
                and not _is_square_attacked(2, 1)
            ):
                moves.append(encode_move(4, 2))
    elif side_to_move == 1 and sq == 60:
        if (
            castling_rights & 0b0100
            and _bit_at(occupancy, 61) == 0
            and _bit_at(occupancy, 62) == 0
            and _bit_at(bb_b_rooks, 63) == 1
        ):
            if (
                not _is_square_attacked(60, 0)
                and not _is_square_attacked(61, 0)
                and not _is_square_attacked(62, 0)
            ):
                moves.append(encode_move(60, 62))
        if (
            castling_rights & 0b1000
            and _bit_at(occupancy, 59) == 0
            and _bit_at(occupancy, 58) == 0
            and _bit_at(occupancy, 57) == 0
            and _bit_at(bb_b_rooks, 56) == 1
        ):
            if (
                not _is_square_attacked(60, 0)
                and not _is_square_attacked(59, 0)
                and not _is_square_attacked(58, 0)
            ):
                moves.append(encode_move(60, 58))


func apply_move_in_place(move: int) -> void:
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
            set_piece_at(rook_to, rook_piece)
        elif from - to == 2:  # Queenside
            var rook_from_q := to - 2
            var rook_to_q := to + 1
            var rook_piece_q := _piece_at(rook_from_q)
            _remove_piece_at(rook_from_q, rook_piece_q)
            set_piece_at(rook_to_q, rook_piece_q)

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
    set_piece_at(to, piece)

    # Update king cache
    if abs_p == Piece.KING:
        if side_to_move == 0:
            white_king_sq = to
        else:
            black_king_sq = to

    # Pawn promotion (auto-queen)
    if abs_p == Piece.PAWN:
        if (side_to_move == 0 and rank_of(to) == 7) or (side_to_move == 1 and rank_of(to) == 0):
            _remove_piece_at(to, piece)
            set_piece_at(to, Piece.QUEEN if side_to_move == 0 else -Piece.QUEEN)

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
        attackers = pawn_attacks_b[sq] & bb_w_pawns
        if attackers != 0: return true
        if knight_attacks[sq] & bb_w_knights: return true
        if king_attacks[sq] & bb_w_king: return true
        if _slider_attacks(sq, occupancy, true) & (bb_w_bishops | bb_w_queens): return true
        if _slider_attacks(sq, occupancy, false) & (bb_w_rooks | bb_w_queens): return true
    else:
        attackers = pawn_attacks_w[sq] & bb_b_pawns
        if attackers != 0: return true
        if knight_attacks[sq] & bb_b_knights: return true
        if king_attacks[sq] & bb_b_king: return true
        if _slider_attacks(sq, occupancy, true) & (bb_b_bishops | bb_b_queens): return true
        if _slider_attacks(sq, occupancy, false) & (bb_b_rooks | bb_b_queens): return true

    return false


func _slider_attacks(sq: int, occupancy: int, diagonal: bool) -> int:
    var attacks := 0
    if diagonal:
        attacks |= _ray_attacks(sq, occupancy, ray_ne, true)
        attacks |= _ray_attacks(sq, occupancy, ray_nw, true)
        attacks |= _ray_attacks(sq, occupancy, ray_se, false)
        attacks |= _ray_attacks(sq, occupancy, ray_sw, false)
    else:
        attacks |= _ray_attacks(sq, occupancy, ray_n, true)
        attacks |= _ray_attacks(sq, occupancy, ray_e, true)
        attacks |= _ray_attacks(sq, occupancy, ray_s, false)
        attacks |= _ray_attacks(sq, occupancy, ray_w, false)
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
    var piece := 0
    if bb_w_pawns & mask: piece = Piece.PAWN
    elif bb_w_knights & mask: piece = Piece.KNIGHT
    elif bb_w_bishops & mask: piece = Piece.BISHOP
    elif bb_w_rooks & mask: piece = Piece.ROOK
    elif bb_w_queens & mask: piece = Piece.QUEEN
    elif bb_w_king & mask: piece = Piece.KING
    elif bb_b_pawns & mask: piece = -Piece.PAWN
    elif bb_b_knights & mask: piece = -Piece.KNIGHT
    elif bb_b_bishops & mask: piece = -Piece.BISHOP
    elif bb_b_rooks & mask: piece = -Piece.ROOK
    elif bb_b_queens & mask: piece = -Piece.QUEEN
    elif bb_b_king & mask: piece = -Piece.KING
    return piece


func set_piece_at(sq: int, piece: int) -> void:
    var mask := 1 << sq
    match piece:
        Piece.PAWN: bb_w_pawns |= mask
        Piece.KNIGHT: bb_w_knights |= mask
        Piece.BISHOP: bb_w_bishops |= mask
        Piece.ROOK: bb_w_rooks |= mask
        Piece.QUEEN: bb_w_queens |= mask
        Piece.KING:
            bb_w_king |= mask
            white_king_sq = sq

        -Piece.PAWN: bb_b_pawns |= mask
        -Piece.KNIGHT: bb_b_knights |= mask
        -Piece.BISHOP: bb_b_bishops |= mask
        -Piece.ROOK: bb_b_rooks |= mask
        -Piece.QUEEN: bb_b_queens |= mask
        -Piece.KING:
            bb_b_king |= mask
            black_king_sq = sq


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


func find_king(color: int) -> int:
    return white_king_sq if color == 0 else black_king_sq


static func _bit_scan_forward(bb: int) -> int:
    ## O(log n) forward scan using binary search on the isolated lowest bit.
    if bb == 0:
        return -1
    bb = bb & (-bb)  # isolate lowest set bit
    var n := 0
    # Use ~complement to avoid hex_to_int overflow for values > 0x7FFFFFFFFFFFFFFF
    if bb & ~0xFFFFFFFF: n += 32
    if bb & ~0x0000FFFF0000FFFF: n += 16
    if bb & ~0x00FF00FF00FF00FF: n += 8
    if bb & ~0x0F0F0F0F0F0F0F0F: n += 4
    if bb & ~0x3333333333333333: n += 2
    if bb & ~0x5555555555555555: n += 1
    return n


static func _bit_scan_reverse(bb: int) -> int:
    ## O(log n) reverse scan using binary search.
    if bb == 0:
        return -1
    var n := 0
    if bb & ~0xFFFFFFFF: n += 32; bb >>= 32
    if bb & 0x00000000FFFF0000: n += 16; bb >>= 16
    if bb & 0x000000000000FF00: n += 8; bb >>= 8
    if bb & 0x00000000000000F0: n += 4; bb >>= 4
    if bb & 0x000000000000000C: n += 2; bb >>= 2
    if bb & 0x0000000000000002: n += 1
    return n


static func _popcount(bb: int) -> int:
    ## Kernighan's algorithm: O(k) where k = number of set bits.
    var count := 0
    while bb != 0:
        bb &= bb - 1
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
