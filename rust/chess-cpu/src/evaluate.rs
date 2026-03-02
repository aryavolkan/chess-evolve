use crate::board::ChessBoard;

/// Material score for one side using bitboard popcount.
pub fn material_score(board: &ChessBoard, color: u8) -> f32 {
    let values: [f32; 6] = [1.0, 3.0, 3.25, 5.0, 9.0, 0.0];
    let offset = if color == 0 { 0 } else { 6 };
    let mut total = 0.0f32;
    for i in 0..6 {
        total += board.bb[offset + i].0.count_ones() as f32 * values[i];
    }
    total
}

/// Mobility score: number of legal moves available to the given side.
pub fn mobility_score(board: &ChessBoard, color: u8) -> i32 {
    let mut copy = *board;
    copy.side_to_move = color;
    let mut buf = Vec::with_capacity(256);
    copy.get_legal_moves_with_buf(&mut buf).len() as i32
}

/// King safety score: count of friendly pawns adjacent to the king.
pub fn king_safety_score(board: &ChessBoard, color: u8) -> f32 {
    let king_bb = if color == 0 {
        board.bb[5].0
    } else {
        board.bb[11].0
    };
    if king_bb == 0 {
        return 0.0;
    }
    let king_sq = king_bb.trailing_zeros() as i32;
    if king_sq >= 64 {
        return 0.0;
    }
    let kf = king_sq % 8;
    let kr = king_sq / 8;
    let pawn_piece: i8 = if color == 0 { 1 } else { -1 };
    let mut safety = 0.0;
    for df in -1..=1 {
        for dr in -1..=1 {
            let nf = kf + df;
            let nr = kr + dr;
            if !(0..=7).contains(&nf) || !(0..=7).contains(&nr) {
                continue;
            }
            let sq = (nr * 8 + nf) as usize;
            if board.piece_at(sq) == pawn_piece {
                safety += 1.0;
            }
        }
    }
    safety
}
