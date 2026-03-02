use std::sync::LazyLock;

const MOVE_FLAG_PROMOTE: u32 = 1 << 12;
const MOVE_FLAG_EN_PASSANT: u32 = 1 << 13;

const PIECE_NONE: i8 = 0;
const PIECE_PAWN: i8 = 1;
const PIECE_KNIGHT: i8 = 2;
const PIECE_BISHOP: i8 = 3;
const PIECE_ROOK: i8 = 4;
const PIECE_QUEEN: i8 = 5;
const PIECE_KING: i8 = 6;

static KNIGHT_ATTACKS: LazyLock<[u64; 64]> = LazyLock::new(|| {
    let mut table = [0u64; 64];
    for (sq, val) in table.iter_mut().enumerate() {
        let bb = 1u64 << sq;
        let mut attacks = 0u64;
        if sq % 8 > 1 {
            attacks |= bb >> 10;
            attacks |= bb << 6;
        }
        if sq % 8 > 0 {
            attacks |= bb >> 17;
            attacks |= bb << 15;
        }
        if sq % 8 < 7 {
            attacks |= bb >> 15;
            attacks |= bb << 17;
        }
        if sq % 8 < 6 {
            attacks |= bb >> 6;
            attacks |= bb << 10;
        }
        *val = attacks;
    }
    table
});

static KING_ATTACKS: LazyLock<[u64; 64]> = LazyLock::new(|| {
    let mut table = [0u64; 64];
    for (sq, val) in table.iter_mut().enumerate() {
        let f = file_of(sq) as i32;
        let r = rank_of(sq) as i32;
        let mut attacks = 0u64;
        for df in -1..=1 {
            for dr in -1..=1 {
                if df == 0 && dr == 0 {
                    continue;
                }
                let nf = f + df;
                let nr = r + dr;
                if !(0..=7).contains(&nf) || !(0..=7).contains(&nr) {
                    continue;
                }
                attacks |= 1u64 << square(nf as usize, nr as usize);
            }
        }
        *val = attacks;
    }
    table
});

static PAWN_ATTACKS_WHITE: LazyLock<[u64; 64]> = LazyLock::new(|| {
    let mut table = [0u64; 64];
    for (sq, val) in table.iter_mut().enumerate() {
        let f = file_of(sq) as i32;
        let r = rank_of(sq) as i32;
        let mut attacks = 0u64;
        let nr = r + 1;
        if (0..=7).contains(&nr) {
            for df in [-1, 1] {
                let nf = f + df;
                if !(0..=7).contains(&nf) {
                    continue;
                }
                attacks |= 1u64 << square(nf as usize, nr as usize);
            }
        }
        *val = attacks;
    }
    table
});

static PAWN_ATTACKS_BLACK: LazyLock<[u64; 64]> = LazyLock::new(|| {
    let mut table = [0u64; 64];
    for (sq, val) in table.iter_mut().enumerate() {
        let f = file_of(sq) as i32;
        let r = rank_of(sq) as i32;
        let mut attacks = 0u64;
        let nr = r - 1;
        if (0..=7).contains(&nr) {
            for df in [-1, 1] {
                let nf = f + df;
                if !(0..=7).contains(&nf) {
                    continue;
                }
                attacks |= 1u64 << square(nf as usize, nr as usize);
            }
        }
        *val = attacks;
    }
    table
});

static RAY_N: LazyLock<[u64; 64]> = LazyLock::new(|| compute_ray_table(0, 1));
static RAY_S: LazyLock<[u64; 64]> = LazyLock::new(|| compute_ray_table(0, -1));
static RAY_E: LazyLock<[u64; 64]> = LazyLock::new(|| compute_ray_table(1, 0));
static RAY_W: LazyLock<[u64; 64]> = LazyLock::new(|| compute_ray_table(-1, 0));
static RAY_NE: LazyLock<[u64; 64]> = LazyLock::new(|| compute_ray_table(1, 1));
static RAY_NW: LazyLock<[u64; 64]> = LazyLock::new(|| compute_ray_table(-1, 1));
static RAY_SE: LazyLock<[u64; 64]> = LazyLock::new(|| compute_ray_table(1, -1));
static RAY_SW: LazyLock<[u64; 64]> = LazyLock::new(|| compute_ray_table(-1, -1));

#[derive(Clone, Copy, Default)]
pub struct BitBoard(pub u64);

#[derive(Clone, Copy)]
pub struct ChessBoard {
    pub bb: [BitBoard; 12],
    pub pieces: [i8; 64],
    pub side_to_move: u8,
    pub castling_rights: u8,
    pub en_passant_sq: i8,
    pub halfmove_clock: u32,
}

impl ChessBoard {
    pub fn from_fen(fen: &str) -> Option<Self> {
        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.len() < 4 {
            return None;
        }

        let mut board = ChessBoard {
            bb: [BitBoard(0); 12],
            pieces: [PIECE_NONE; 64],
            side_to_move: 0,
            castling_rights: 0,
            en_passant_sq: -1,
            halfmove_clock: 0,
        };

        // Piece placement (rank 8 down to rank 1)
        let ranks: Vec<&str> = parts[0].split('/').collect();
        if ranks.len() != 8 {
            return None;
        }
        for (rank_idx, rank_str) in ranks.iter().enumerate() {
            let rank = 7 - rank_idx; // FEN rank 8 = internal rank 7
            let mut file = 0usize;
            for ch in rank_str.chars() {
                if let Some(skip) = ch.to_digit(10) {
                    file += skip as usize;
                } else {
                    let piece = match ch {
                        'P' => PIECE_PAWN,
                        'N' => PIECE_KNIGHT,
                        'B' => PIECE_BISHOP,
                        'R' => PIECE_ROOK,
                        'Q' => PIECE_QUEEN,
                        'K' => PIECE_KING,
                        'p' => -PIECE_PAWN,
                        'n' => -PIECE_KNIGHT,
                        'b' => -PIECE_BISHOP,
                        'r' => -PIECE_ROOK,
                        'q' => -PIECE_QUEEN,
                        'k' => -PIECE_KING,
                        _ => return None,
                    };
                    let sq = rank * 8 + file;
                    board.set_piece_at(sq, piece);
                    file += 1;
                }
            }
        }

        // Side to move
        board.side_to_move = match parts[1] {
            "w" => 0,
            "b" => 1,
            _ => return None,
        };

        // Castling rights
        if parts[2] != "-" {
            for ch in parts[2].chars() {
                match ch {
                    'K' => board.castling_rights |= 0b0001,
                    'Q' => board.castling_rights |= 0b0010,
                    'k' => board.castling_rights |= 0b0100,
                    'q' => board.castling_rights |= 0b1000,
                    _ => return None,
                }
            }
        }

        // En passant
        if parts[3] != "-" {
            let ep_bytes = parts[3].as_bytes();
            if ep_bytes.len() == 2 {
                let ep_file = (ep_bytes[0] - b'a') as usize;
                let ep_rank = (ep_bytes[1] - b'1') as usize;
                if ep_file < 8 && ep_rank < 8 {
                    board.en_passant_sq = (ep_rank * 8 + ep_file) as i8;
                }
            }
        }

        // Halfmove clock
        if parts.len() > 4 {
            board.halfmove_clock = parts[4].parse().unwrap_or(0);
        }

        Some(board)
    }

    pub fn startpos() -> Self {
        let w_pawns = 0x0000_0000_0000_ff00u64;
        let w_rooks = 0x0000_0000_0000_0081u64;
        let w_knights = 0x0000_0000_0000_0042u64;
        let w_bishops = 0x0000_0000_0000_0024u64;
        let w_queens = 0x0000_0000_0000_0008u64;
        let w_king = 0x0000_0000_0000_0010u64;

        let b_pawns = 0x00ff_0000_0000_0000u64;
        let b_rooks = 0x8100_0000_0000_0000u64;
        let b_knights = 0x4200_0000_0000_0000u64;
        let b_bishops = 0x2400_0000_0000_0000u64;
        let b_queens = 0x0800_0000_0000_0000u64;
        let b_king = 0x1000_0000_0000_0000u64;

        let bb = [
            BitBoard(w_pawns),
            BitBoard(w_knights),
            BitBoard(w_bishops),
            BitBoard(w_rooks),
            BitBoard(w_queens),
            BitBoard(w_king),
            BitBoard(b_pawns),
            BitBoard(b_knights),
            BitBoard(b_bishops),
            BitBoard(b_rooks),
            BitBoard(b_queens),
            BitBoard(b_king),
        ];

        let mut pieces = [PIECE_NONE; 64];
        for (sq, val) in pieces.iter_mut().enumerate() {
            let mask = 1u64 << sq;
            if bb[0].0 & mask != 0 {
                *val = PIECE_PAWN;
            } else if bb[1].0 & mask != 0 {
                *val = PIECE_KNIGHT;
            } else if bb[2].0 & mask != 0 {
                *val = PIECE_BISHOP;
            } else if bb[3].0 & mask != 0 {
                *val = PIECE_ROOK;
            } else if bb[4].0 & mask != 0 {
                *val = PIECE_QUEEN;
            } else if bb[5].0 & mask != 0 {
                *val = PIECE_KING;
            } else if bb[6].0 & mask != 0 {
                *val = -PIECE_PAWN;
            } else if bb[7].0 & mask != 0 {
                *val = -PIECE_KNIGHT;
            } else if bb[8].0 & mask != 0 {
                *val = -PIECE_BISHOP;
            } else if bb[9].0 & mask != 0 {
                *val = -PIECE_ROOK;
            } else if bb[10].0 & mask != 0 {
                *val = -PIECE_QUEEN;
            } else if bb[11].0 & mask != 0 {
                *val = -PIECE_KING;
            }
        }

        ChessBoard {
            bb,
            pieces,
            side_to_move: 0,
            castling_rights: 0b1111,
            en_passant_sq: -1,
            halfmove_clock: 0,
        }
    }

    pub fn get_legal_moves(&self) -> Vec<u32> {
        let mut pseudo = Vec::with_capacity(256);
        self.get_legal_moves_with_buf(&mut pseudo)
    }

    /// Like get_legal_moves but reuses the provided buffer for pseudo-legal moves,
    /// avoiding repeated heap allocation in hot loops.
    pub fn get_legal_moves_with_buf(&self, pseudo: &mut Vec<u32>) -> Vec<u32> {
        self.generate_pseudo_legal_moves_into(pseudo);
        let mut legal = Vec::with_capacity(pseudo.len());
        for &mv in pseudo.iter() {
            let mut copy = *self;
            copy.make_move_unchecked(mv);
            if !copy.is_in_check(1 - copy.side_to_move) {
                legal.push(mv);
            }
        }
        legal
    }

    pub fn make_move(&self, mv: u32) -> Self {
        let mut next = *self;
        next.make_move_unchecked(mv);
        next
    }

    pub fn make_move_unchecked(&mut self, mv: u32) {
        let from = ((mv >> 6) & 0x3f) as usize;
        let to = (mv & 0x3f) as usize;
        let flags = mv & !0xfff;
        let piece = self.piece_at(from);
        let abs_p = piece.abs();
        let mut captured = self.piece_at(to);

        if abs_p == PIECE_PAWN && (flags & MOVE_FLAG_EN_PASSANT) != 0 {
            let ep_sq = if self.side_to_move == 0 {
                to as i32 - 8
            } else {
                to as i32 + 8
            } as usize;
            captured = self.piece_at(ep_sq);
            self.remove_piece_at(ep_sq, captured);
        }

        if abs_p == PIECE_PAWN || captured != PIECE_NONE {
            self.halfmove_clock = 0;
        } else {
            self.halfmove_clock += 1;
        }

        self.en_passant_sq = -1;
        if abs_p == PIECE_PAWN {
            let from_rank = rank_of(from) as i32;
            let to_rank = rank_of(to) as i32;
            if (to_rank - from_rank).abs() == 2 {
                let file = file_of(from) as i32;
                let mid_rank = (from_rank + to_rank) / 2;
                self.en_passant_sq = square(file as usize, mid_rank as usize) as i8;
            }
        }

        if abs_p == PIECE_KING {
            if to as i32 - from as i32 == 2 {
                let rook_from = to + 1;
                let rook_to = to - 1;
                let rook_piece = self.piece_at(rook_from);
                self.remove_piece_at(rook_from, rook_piece);
                self.set_piece_at(rook_to, rook_piece);
            } else if from as i32 - to as i32 == 2 {
                let rook_from = to - 2;
                let rook_to = to + 1;
                let rook_piece = self.piece_at(rook_from);
                self.remove_piece_at(rook_from, rook_piece);
                self.set_piece_at(rook_to, rook_piece);
            }
        }

        if abs_p == PIECE_KING {
            if self.side_to_move == 0 {
                self.castling_rights &= 0b1100;
            } else {
                self.castling_rights &= 0b0011;
            }
        }
        if from == 0 || to == 0 {
            self.castling_rights &= !0b0010;
        }
        if from == 7 || to == 7 {
            self.castling_rights &= !0b0001;
        }
        if from == 56 || to == 56 {
            self.castling_rights &= !0b1000;
        }
        if from == 63 || to == 63 {
            self.castling_rights &= !0b0100;
        }

        self.remove_piece_at(from, piece);
        if captured != PIECE_NONE {
            self.remove_piece_at(to, captured);
        }
        self.set_piece_at(to, piece);

        if abs_p == PIECE_PAWN
            && ((self.side_to_move == 0 && rank_of(to) == 7)
                || (self.side_to_move == 1 && rank_of(to) == 0))
        {
            self.remove_piece_at(to, piece);
            let promo_piece = if self.side_to_move == 0 {
                PIECE_QUEEN
            } else {
                -PIECE_QUEEN
            };
            self.set_piece_at(to, promo_piece);
        }

        self.side_to_move ^= 1;
    }

    pub fn is_in_check(&self, color: u8) -> bool {
        let king_bb = if color == 0 {
            self.bb[5].0
        } else {
            self.bb[11].0
        };
        if king_bb == 0 {
            return false;
        }
        let king_sq = king_bb.trailing_zeros() as usize;
        self.is_square_attacked(king_sq, 1 - color)
    }

    fn generate_pseudo_legal_moves_into(&self, moves: &mut Vec<u32>) {
        moves.clear();
        let white_occ = self.white_occ();
        let black_occ = self.black_occ();
        let all_occ = white_occ | black_occ;
        let friendly_occ = if self.side_to_move == 0 {
            white_occ
        } else {
            black_occ
        };
        let enemy_occ = if self.side_to_move == 0 {
            black_occ
        } else {
            white_occ
        };

        if self.side_to_move == 0 {
            self.add_pawn_moves(self.bb[0].0, 0, friendly_occ, enemy_occ, moves);
            self.add_knight_moves(self.bb[1].0, friendly_occ, moves);
            self.add_slider_moves(self.bb[2].0, all_occ, friendly_occ, moves, true);
            self.add_slider_moves(self.bb[3].0, all_occ, friendly_occ, moves, false);
            self.add_slider_moves(self.bb[4].0, all_occ, friendly_occ, moves, true);
            self.add_slider_moves(self.bb[4].0, all_occ, friendly_occ, moves, false);
            self.add_king_moves(self.bb[5].0, friendly_occ, enemy_occ, all_occ, moves);
        } else {
            self.add_pawn_moves(self.bb[6].0, 1, friendly_occ, enemy_occ, moves);
            self.add_knight_moves(self.bb[7].0, friendly_occ, moves);
            self.add_slider_moves(self.bb[8].0, all_occ, friendly_occ, moves, true);
            self.add_slider_moves(self.bb[9].0, all_occ, friendly_occ, moves, false);
            self.add_slider_moves(self.bb[10].0, all_occ, friendly_occ, moves, true);
            self.add_slider_moves(self.bb[10].0, all_occ, friendly_occ, moves, false);
            self.add_king_moves(self.bb[11].0, friendly_occ, enemy_occ, all_occ, moves);
        }
    }

    fn add_pawn_moves(
        &self,
        pawns: u64,
        color: u8,
        friendly_occ: u64,
        enemy_occ: u64,
        moves: &mut Vec<u32>,
    ) {
        let dir: i32 = if color == 0 { 8 } else { -8 };
        let start_rank: i32 = if color == 0 { 1 } else { 6 };
        let promotion_rank: i32 = if color == 0 { 7 } else { 0 };
        let pawn_attacks = if color == 0 {
            &*PAWN_ATTACKS_WHITE
        } else {
            &*PAWN_ATTACKS_BLACK
        };

        let mut bb = pawns;
        while bb != 0 {
            let sq = bb.trailing_zeros() as usize;
            bb &= bb - 1;
            let r = rank_of(sq) as i32;

            let one_sq = sq as i32 + dir;
            if (0..64).contains(&one_sq) && ((friendly_occ | enemy_occ) & (1u64 << one_sq)) == 0 {
                let flags = if rank_of(one_sq as usize) as i32 == promotion_rank {
                    MOVE_FLAG_PROMOTE
                } else {
                    0
                };
                moves.push(encode_move(sq as u32, one_sq as u32, flags));
                if r == start_rank {
                    let two_sq = sq as i32 + dir * 2;
                    if (0..64).contains(&two_sq)
                        && ((friendly_occ | enemy_occ) & (1u64 << two_sq)) == 0
                    {
                        moves.push(encode_move(sq as u32, two_sq as u32, 0));
                    }
                }
            }

            let mut attacks = pawn_attacks[sq] & enemy_occ;
            while attacks != 0 {
                let to_sq = attacks.trailing_zeros() as usize;
                attacks &= attacks - 1;
                let cap_flags = if rank_of(to_sq) as i32 == promotion_rank {
                    MOVE_FLAG_PROMOTE
                } else {
                    0
                };
                moves.push(encode_move(sq as u32, to_sq as u32, cap_flags));
            }

            if self.en_passant_sq != -1 {
                let ep_sq = self.en_passant_sq as usize;
                let ep_mask = 1u64 << ep_sq;
                if pawn_attacks[sq] & ep_mask != 0 {
                    moves.push(encode_move(sq as u32, ep_sq as u32, MOVE_FLAG_EN_PASSANT));
                }
            }
        }
    }

    fn add_knight_moves(&self, knights: u64, friendly_occ: u64, moves: &mut Vec<u32>) {
        let mut bb = knights;
        while bb != 0 {
            let sq = bb.trailing_zeros() as usize;
            bb &= bb - 1;
            let mut attacks = KNIGHT_ATTACKS[sq] & !friendly_occ;
            while attacks != 0 {
                let to_sq = attacks.trailing_zeros() as usize;
                attacks &= attacks - 1;
                moves.push(encode_move(sq as u32, to_sq as u32, 0));
            }
        }
    }

    fn add_slider_moves(
        &self,
        sliders: u64,
        occupancy: u64,
        friendly_occ: u64,
        moves: &mut Vec<u32>,
        is_diagonal: bool,
    ) {
        let mut bb = sliders;
        while bb != 0 {
            let sq = bb.trailing_zeros() as usize;
            bb &= bb - 1;
            let mut attacks = self.slider_attacks(sq, occupancy, is_diagonal);
            attacks &= !friendly_occ;
            while attacks != 0 {
                let to_sq = attacks.trailing_zeros() as usize;
                attacks &= attacks - 1;
                moves.push(encode_move(sq as u32, to_sq as u32, 0));
            }
        }
    }

    fn add_king_moves(
        &self,
        king: u64,
        friendly_occ: u64,
        _enemy_occ: u64,
        occupancy: u64,
        moves: &mut Vec<u32>,
    ) {
        if king == 0 {
            return;
        }
        let sq = king.trailing_zeros() as usize;
        let mut attacks = KING_ATTACKS[sq] & !friendly_occ;
        while attacks != 0 {
            let to_sq = attacks.trailing_zeros() as usize;
            attacks &= attacks - 1;
            moves.push(encode_move(sq as u32, to_sq as u32, 0));
        }

        if self.side_to_move == 0 && sq == 4 {
            if (self.castling_rights & 0b0001) != 0
                && bit_at(occupancy, 5) == 0
                && bit_at(occupancy, 6) == 0
                && bit_at(self.bb[3].0, 7) == 1
                && !self.is_square_attacked(4, 1)
                && !self.is_square_attacked(5, 1)
                && !self.is_square_attacked(6, 1)
            {
                moves.push(encode_move(4, 6, 0));
            }
            if (self.castling_rights & 0b0010) != 0
                && bit_at(occupancy, 3) == 0
                && bit_at(occupancy, 2) == 0
                && bit_at(occupancy, 1) == 0
                && bit_at(self.bb[3].0, 0) == 1
                && !self.is_square_attacked(4, 1)
                && !self.is_square_attacked(3, 1)
                && !self.is_square_attacked(2, 1)
            {
                moves.push(encode_move(4, 2, 0));
            }
        } else if self.side_to_move == 1 && sq == 60 {
            if (self.castling_rights & 0b0100) != 0
                && bit_at(occupancy, 61) == 0
                && bit_at(occupancy, 62) == 0
                && bit_at(self.bb[9].0, 63) == 1
                && !self.is_square_attacked(60, 0)
                && !self.is_square_attacked(61, 0)
                && !self.is_square_attacked(62, 0)
            {
                moves.push(encode_move(60, 62, 0));
            }
            if (self.castling_rights & 0b1000) != 0
                && bit_at(occupancy, 59) == 0
                && bit_at(occupancy, 58) == 0
                && bit_at(occupancy, 57) == 0
                && bit_at(self.bb[9].0, 56) == 1
                && !self.is_square_attacked(60, 0)
                && !self.is_square_attacked(59, 0)
                && !self.is_square_attacked(58, 0)
            {
                moves.push(encode_move(60, 58, 0));
            }
        }
    }

    fn is_square_attacked(&self, sq: usize, by_color: u8) -> bool {
        let enemy_occ = if by_color == 1 {
            self.black_occ()
        } else {
            self.white_occ()
        };
        let friendly_occ = if by_color == 1 {
            self.white_occ()
        } else {
            self.black_occ()
        };
        let occupancy = enemy_occ | friendly_occ;

        if by_color == 0 {
            if PAWN_ATTACKS_BLACK[sq] & self.bb[0].0 != 0 {
                return true;
            }
            if KNIGHT_ATTACKS[sq] & self.bb[1].0 != 0 {
                return true;
            }
            if KING_ATTACKS[sq] & self.bb[5].0 != 0 {
                return true;
            }
            if self.slider_attacks(sq, occupancy, true) & (self.bb[2].0 | self.bb[4].0) != 0 {
                return true;
            }
            if self.slider_attacks(sq, occupancy, false) & (self.bb[3].0 | self.bb[4].0) != 0 {
                return true;
            }
        } else {
            if PAWN_ATTACKS_WHITE[sq] & self.bb[6].0 != 0 {
                return true;
            }
            if KNIGHT_ATTACKS[sq] & self.bb[7].0 != 0 {
                return true;
            }
            if KING_ATTACKS[sq] & self.bb[11].0 != 0 {
                return true;
            }
            if self.slider_attacks(sq, occupancy, true) & (self.bb[8].0 | self.bb[10].0) != 0 {
                return true;
            }
            if self.slider_attacks(sq, occupancy, false) & (self.bb[9].0 | self.bb[10].0) != 0 {
                return true;
            }
        }

        false
    }

    fn slider_attacks(&self, sq: usize, occupancy: u64, diagonal: bool) -> u64 {
        let mut attacks = 0u64;
        if diagonal {
            attacks |= ray_attacks(sq, occupancy, &RAY_NE, true);
            attacks |= ray_attacks(sq, occupancy, &RAY_NW, true);
            attacks |= ray_attacks(sq, occupancy, &RAY_SE, false);
            attacks |= ray_attacks(sq, occupancy, &RAY_SW, false);
        } else {
            attacks |= ray_attacks(sq, occupancy, &RAY_N, true);
            attacks |= ray_attacks(sq, occupancy, &RAY_E, true);
            attacks |= ray_attacks(sq, occupancy, &RAY_S, false);
            attacks |= ray_attacks(sq, occupancy, &RAY_W, false);
        }
        attacks
    }

    fn white_occ(&self) -> u64 {
        self.bb[0].0 | self.bb[1].0 | self.bb[2].0 | self.bb[3].0 | self.bb[4].0 | self.bb[5].0
    }

    fn black_occ(&self) -> u64 {
        self.bb[6].0 | self.bb[7].0 | self.bb[8].0 | self.bb[9].0 | self.bb[10].0 | self.bb[11].0
    }

    pub fn piece_at(&self, sq: usize) -> i8 {
        self.pieces[sq]
    }

    fn set_piece_at(&mut self, sq: usize, piece: i8) {
        let mask = 1u64 << sq;
        self.pieces[sq] = piece;
        match piece {
            PIECE_PAWN => self.bb[0].0 |= mask,
            PIECE_KNIGHT => self.bb[1].0 |= mask,
            PIECE_BISHOP => self.bb[2].0 |= mask,
            PIECE_ROOK => self.bb[3].0 |= mask,
            PIECE_QUEEN => self.bb[4].0 |= mask,
            PIECE_KING => self.bb[5].0 |= mask,
            p if p == -PIECE_PAWN => self.bb[6].0 |= mask,
            p if p == -PIECE_KNIGHT => self.bb[7].0 |= mask,
            p if p == -PIECE_BISHOP => self.bb[8].0 |= mask,
            p if p == -PIECE_ROOK => self.bb[9].0 |= mask,
            p if p == -PIECE_QUEEN => self.bb[10].0 |= mask,
            p if p == -PIECE_KING => self.bb[11].0 |= mask,
            _ => {}
        }
    }

    fn remove_piece_at(&mut self, sq: usize, piece: i8) {
        if piece == PIECE_NONE {
            return;
        }
        let mask = !(1u64 << sq);
        self.pieces[sq] = PIECE_NONE;
        match piece {
            PIECE_PAWN => self.bb[0].0 &= mask,
            PIECE_KNIGHT => self.bb[1].0 &= mask,
            PIECE_BISHOP => self.bb[2].0 &= mask,
            PIECE_ROOK => self.bb[3].0 &= mask,
            PIECE_QUEEN => self.bb[4].0 &= mask,
            PIECE_KING => self.bb[5].0 &= mask,
            p if p == -PIECE_PAWN => self.bb[6].0 &= mask,
            p if p == -PIECE_KNIGHT => self.bb[7].0 &= mask,
            p if p == -PIECE_BISHOP => self.bb[8].0 &= mask,
            p if p == -PIECE_ROOK => self.bb[9].0 &= mask,
            p if p == -PIECE_QUEEN => self.bb[10].0 &= mask,
            p if p == -PIECE_KING => self.bb[11].0 &= mask,
            _ => {}
        }
    }
}

fn encode_move(from: u32, to: u32, flags: u32) -> u32 {
    (from << 6) | to | flags
}

fn file_of(sq: usize) -> usize {
    sq % 8
}

fn rank_of(sq: usize) -> usize {
    sq / 8
}

fn square(file: usize, rank: usize) -> usize {
    rank * 8 + file
}

fn compute_ray_table(df: i32, dr: i32) -> [u64; 64] {
    let mut table = [0u64; 64];
    for (sq, val) in table.iter_mut().enumerate() {
        let f = file_of(sq) as i32;
        let r = rank_of(sq) as i32;
        let mut attacks = 0u64;
        let mut nf = f + df;
        let mut nr = r + dr;
        while (0..=7).contains(&nf) && (0..=7).contains(&nr) {
            attacks |= 1u64 << square(nf as usize, nr as usize);
            nf += df;
            nr += dr;
        }
        *val = attacks;
    }
    table
}

fn ray_attacks(sq: usize, occupancy: u64, ray_table: &[u64; 64], positive: bool) -> u64 {
    let mut attacks = ray_table[sq];
    let blockers = attacks & occupancy;
    if blockers != 0 {
        let blocker_sq = if positive {
            blockers.trailing_zeros() as usize
        } else {
            63 - blockers.leading_zeros() as usize
        };
        attacks &= !ray_table[blocker_sq];
    }
    attacks
}

fn bit_at(bb: u64, sq: usize) -> u8 {
    if (bb & (1u64 << sq)) != 0 {
        1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sq(file: char, rank: u8) -> u32 {
        let file_idx = (file as u8 - b'a') as u32;
        let rank_idx = (rank - 1) as u32;
        rank_idx * 8 + file_idx
    }

    fn perft(board: &ChessBoard, depth: u32) -> u64 {
        if depth == 0 {
            return 1;
        }
        let moves = board.get_legal_moves();
        let mut nodes = 0u64;
        for mv in moves {
            let next = board.make_move(mv);
            nodes += perft(&next, depth - 1);
        }
        nodes
    }

    #[test]
    fn startpos_has_20_moves_and_perft() {
        let board = ChessBoard::startpos();
        let moves = board.get_legal_moves();
        assert_eq!(moves.len(), 20);
        assert!(moves.contains(&encode_move(sq('e', 2), sq('e', 4), 0)));
        assert_eq!(perft(&board, 1), 20);
        assert_eq!(perft(&board, 2), 400);
    }

    #[test]
    fn en_passant_is_generated() {
        let mut board = ChessBoard::startpos();
        board = board.make_move(encode_move(sq('e', 2), sq('e', 4), 0));
        board = board.make_move(encode_move(sq('a', 7), sq('a', 6), 0));
        board = board.make_move(encode_move(sq('e', 4), sq('e', 5), 0));
        board = board.make_move(encode_move(sq('d', 7), sq('d', 5), 0));
        let moves = board.get_legal_moves();
        let ep_move = encode_move(sq('e', 5), sq('d', 6), MOVE_FLAG_EN_PASSANT);
        assert!(moves.contains(&ep_move));
    }

    #[test]
    fn castling_available_when_clear() {
        let mut board = ChessBoard::startpos();
        board = board.make_move(encode_move(sq('e', 2), sq('e', 4), 0));
        board = board.make_move(encode_move(sq('a', 7), sq('a', 6), 0));
        board = board.make_move(encode_move(sq('g', 1), sq('f', 3), 0));
        board = board.make_move(encode_move(sq('a', 6), sq('a', 5), 0));
        board = board.make_move(encode_move(sq('f', 1), sq('e', 2), 0));
        board = board.make_move(encode_move(sq('a', 5), sq('a', 4), 0));
        let moves = board.get_legal_moves();
        assert!(moves.contains(&encode_move(4, 6, 0)));
    }

    #[test]
    fn fen_startpos_matches_startpos() {
        let from_fen =
            ChessBoard::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
                .unwrap();
        let startpos = ChessBoard::startpos();
        assert_eq!(from_fen.side_to_move, startpos.side_to_move);
        assert_eq!(from_fen.castling_rights, startpos.castling_rights);
        assert_eq!(from_fen.en_passant_sq, startpos.en_passant_sq);
        assert_eq!(from_fen.halfmove_clock, startpos.halfmove_clock);
        for i in 0..12 {
            assert_eq!(
                from_fen.bb[i].0, startpos.bb[i].0,
                "bitboard {} mismatch",
                i
            );
        }
        for i in 0..64 {
            assert_eq!(
                from_fen.pieces[i], startpos.pieces[i],
                "piece at sq {} mismatch",
                i
            );
        }
        assert_eq!(from_fen.get_legal_moves().len(), 20);
    }

    #[test]
    fn fen_midgame_position() {
        // After 1. e4 e5 2. Nf3
        let board =
            ChessBoard::from_fen("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2")
                .unwrap();
        assert_eq!(board.side_to_move, 1);
        assert_eq!(board.castling_rights, 0b1111);
        assert_eq!(board.halfmove_clock, 1);
        // Knight on f3 = square(5, 2) = 21
        assert_eq!(board.piece_at(21), PIECE_KNIGHT);
        // Black pawn on e5 = square(4, 4) = 36
        assert_eq!(board.piece_at(36), -PIECE_PAWN);
        // White pawn on e4 = square(4, 3) = 28
        assert_eq!(board.piece_at(28), PIECE_PAWN);
    }

    #[test]
    fn fen_en_passant() {
        // Position with en passant possible on d6
        let board =
            ChessBoard::from_fen("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3")
                .unwrap();
        // d6 = square(3, 5) = 43
        assert_eq!(board.en_passant_sq, 43);
        let moves = board.get_legal_moves();
        let ep_move = encode_move(sq('e', 5), sq('d', 6), MOVE_FLAG_EN_PASSANT);
        assert!(moves.contains(&ep_move));
    }

    #[test]
    fn fen_no_castling() {
        let board =
            ChessBoard::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1").unwrap();
        assert_eq!(board.castling_rights, 0);
    }

    #[test]
    fn fen_invalid_returns_none() {
        assert!(ChessBoard::from_fen("invalid").is_none());
        assert!(ChessBoard::from_fen("").is_none());
        assert!(ChessBoard::from_fen("8/8/8 w").is_none()); // not enough ranks
    }

    #[test]
    fn fen_perft_kiwipete() {
        // Kiwipete position - well-known perft test
        let board = ChessBoard::from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        )
        .unwrap();
        assert_eq!(perft(&board, 1), 48);
    }

    #[test]
    fn in_check_filters_illegal_moves() {
        let mut board = ChessBoard {
            bb: [BitBoard(0); 12],
            pieces: [PIECE_NONE; 64],
            side_to_move: 0,
            castling_rights: 0,
            en_passant_sq: -1,
            halfmove_clock: 0,
        };
        board.set_piece_at(4, PIECE_KING);
        board.set_piece_at(0, PIECE_ROOK);
        board.set_piece_at(60, -PIECE_ROOK);
        board.set_piece_at(56, -PIECE_KING);
        assert!(board.is_in_check(0));
        let moves = board.get_legal_moves();
        for mv in moves {
            let from = ((mv >> 6) & 0x3f) as usize;
            assert_ne!(from, 0);
        }
    }
}
