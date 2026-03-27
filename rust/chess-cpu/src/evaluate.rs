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

/// King safety score: composite metric combining friendly pawn shelter and
/// enemy attack pressure near the king.
///
/// Positive = safer king (more pawn shelter, fewer attackers).
/// Negative = exposed king (little shelter, many nearby attackers).
///
/// Components:
///   +1.0 per friendly pawn adjacent to king (pawn shelter)
///   -0.5 per enemy knight/bishop in 2-square radius (minor piece pressure)
///   -1.0 per enemy rook/queen in 2-square radius (major piece pressure)
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

    let friendly_pawn: i8 = if color == 0 { 1 } else { -1 };
    // Enemy piece values: knight=2, bishop=3, rook=4, queen=5 (absolute)
    let enemy_sign: i8 = if color == 0 { -1 } else { 1 };

    let mut safety = 0.0f32;

    // Scan the 5x5 area centered on the king (radius 2)
    for df in -2..=2i32 {
        for dr in -2..=2i32 {
            let nf = kf + df;
            let nr = kr + dr;
            if !(0..=7).contains(&nf) || !(0..=7).contains(&nr) {
                continue;
            }
            if df == 0 && dr == 0 {
                continue; // skip king's own square
            }
            let sq = (nr * 8 + nf) as usize;
            let piece = board.piece_at(sq);

            // Friendly pawn shelter (adjacent only, radius 1)
            if df.abs() <= 1 && dr.abs() <= 1 && piece == friendly_pawn {
                safety += 1.0;
            }

            // Enemy attack pressure (full 5x5 radius 2)
            let enemy_piece = piece * enemy_sign; // positive if enemy
            match enemy_piece {
                2 | 3 => safety -= 0.5, // enemy knight or bishop nearby
                4 | 5 => safety -= 1.0, // enemy rook or queen nearby
                _ => {}
            }
        }
    }
    safety
}

/// King danger score: measures how vulnerable a king is to attack.
///
/// Higher = more danger. Used to reward attacking the opponent's king.
///
/// Components:
///   - King mobility restriction: (8 - king_legal_moves) * 0.5
///   - Open files toward king: +0.5 per file adjacent to king with no friendly pawns
///   - Open diagonals toward king: +0.5 per unblocked diagonal ray toward king
///     (lanes that enemy bishops/queens can exploit)
///   - Enemy attacker pressure: piece values near king (radius 2)
pub fn king_danger_score(board: &ChessBoard, color: u8) -> f32 {
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
    let friendly_pawn: i8 = if color == 0 { 1 } else { -1 };
    let enemy_sign: i8 = if color == 0 { -1 } else { 1 };

    let mut danger = 0.0f32;

    // 1. King mobility restriction — count king's legal escape squares
    let mut king_moves = 0;
    for df in -1..=1i32 {
        for dr in -1..=1i32 {
            if df == 0 && dr == 0 {
                continue;
            }
            let nf = kf + df;
            let nr = kr + dr;
            if !(0..=7).contains(&nf) || !(0..=7).contains(&nr) {
                continue;
            }
            let sq = (nr * 8 + nf) as usize;
            let piece = board.piece_at(sq);
            // Square is available if empty or occupied by enemy (capturable)
            if piece == 0 || (piece * enemy_sign > 0) {
                king_moves += 1;
            }
        }
    }
    danger += (8 - king_moves) as f32 * 0.5;

    // 2. Open files toward king — check king's file and adjacent files for missing friendly pawns
    for df in -1..=1i32 {
        let file = kf + df;
        if !(0..=7).contains(&file) {
            continue;
        }
        let mut has_friendly_pawn = false;
        for rank in 0..8i32 {
            let sq = (rank * 8 + file) as usize;
            if board.piece_at(sq) == friendly_pawn {
                has_friendly_pawn = true;
                break;
            }
        }
        if !has_friendly_pawn {
            danger += 0.5;
        }
    }

    // 3. Open diagonals toward king — trace 4 diagonal rays from king,
    //    count unblocked ones (lanes for bishop/queen attacks)
    let diag_dirs: [(i32, i32); 4] = [(-1, -1), (-1, 1), (1, -1), (1, 1)];
    for &(df, dr) in &diag_dirs {
        let mut open_len = 0;
        let mut f = kf + df;
        let mut r = kr + dr;
        while (0..=7).contains(&f) && (0..=7).contains(&r) {
            let sq = (r * 8 + f) as usize;
            let piece = board.piece_at(sq);
            if piece != 0 {
                // Ray is blocked — check if blocker is an enemy diagonal attacker
                let enemy_piece = piece * enemy_sign;
                if enemy_piece == 3 || enemy_piece == 5 {
                    // Enemy bishop or queen on this diagonal = very dangerous
                    danger += 1.5;
                }
                break;
            }
            open_len += 1;
            f += df;
            r += dr;
        }
        if open_len >= 2 {
            danger += 0.5; // long open diagonal toward king
        }
    }

    // 4. Enemy attacker pressure in 5x5 radius (same as king_safety but additive)
    for df in -2..=2i32 {
        for dr in -2..=2i32 {
            if df == 0 && dr == 0 {
                continue;
            }
            let nf = kf + df;
            let nr = kr + dr;
            if !(0..=7).contains(&nf) || !(0..=7).contains(&nr) {
                continue;
            }
            let sq = (nr * 8 + nf) as usize;
            let piece = board.piece_at(sq);
            let enemy_piece = piece * enemy_sign;
            match enemy_piece {
                2 | 3 => danger += 0.5, // enemy knight or bishop nearby
                4 | 5 => danger += 1.0, // enemy rook or queen nearby
                _ => {}
            }
        }
    }

    danger
}
