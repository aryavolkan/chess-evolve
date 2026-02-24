#![allow(clippy::needless_range_loop, clippy::wrong_self_convention)]

use godot::prelude::*;

use crate::board::ChessBoard;
use crate::nn::DenseNetwork;

#[derive(GodotClass)]
#[class(base=RefCounted)]
pub struct RustChessBoard {
    board: ChessBoard,
    base: Base<RefCounted>,
}

#[godot_api]
impl IRefCounted for RustChessBoard {
    fn init(base: Base<RefCounted>) -> Self {
        Self {
            board: ChessBoard::startpos(),
            base,
        }
    }
}

#[godot_api]
impl RustChessBoard {
    #[func]
    pub fn from_fen(&mut self, _fen: GString) {
        self.board = ChessBoard::startpos();
    }

    #[func]
    pub fn get_legal_moves(&self) -> PackedInt32Array {
        let moves = self.board.get_legal_moves();
        let mut arr = PackedInt32Array::new();
        for mv in moves {
            arr.push(mv as i32);
        }
        arr
    }

    #[func]
    pub fn make_move(&mut self, mv: i32) {
        self.board = self.board.make_move(mv as u32);
    }

    #[func]
    pub fn is_in_check(&self) -> bool {
        self.board.is_in_check(self.board.side_to_move)
    }

    #[func]
    pub fn side_to_move(&self) -> i32 {
        self.board.side_to_move as i32
    }
}

#[derive(GodotClass)]
#[class(base=RefCounted)]
pub struct RustDenseNetwork {
    net: DenseNetwork,
    scratch_hidden: Vec<f32>,
    scratch_output: Vec<f32>,
    base: Base<RefCounted>,
}

#[derive(GodotClass)]
#[class(base=RefCounted)]
pub struct RustBatchSimulator {
    base: Base<RefCounted>,
}

#[godot_api]
impl IRefCounted for RustDenseNetwork {
    fn init(base: Base<RefCounted>) -> Self {
        Self {
            net: DenseNetwork::from_weights(
                0,
                0,
                0,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            ),
            scratch_hidden: Vec::new(),
            scratch_output: Vec::new(),
            base,
        }
    }
}

#[godot_api]
impl RustDenseNetwork {
    #[func]
    pub fn setup(&mut self, input_size: i32, hidden_size: i32, output_size: i32) {
        let input_size = input_size as usize;
        let hidden_size = hidden_size as usize;
        let output_size = output_size as usize;
        self.net = DenseNetwork::from_weights(
            input_size,
            hidden_size,
            output_size,
            vec![0.0; input_size * hidden_size],
            vec![0.0; hidden_size],
            vec![0.0; hidden_size * output_size],
            vec![0.0; output_size],
        );
        self.scratch_hidden.resize(hidden_size, 0.0);
        self.scratch_output.resize(output_size, 0.0);
    }

    #[func]
    pub fn set_weights(
        &mut self,
        weights_ih: PackedFloat32Array,
        biases_h: PackedFloat32Array,
        weights_ho: PackedFloat32Array,
        biases_o: PackedFloat32Array,
    ) {
        self.net.weights_ih = weights_ih.to_vec();
        self.net.biases_h = biases_h.to_vec();
        self.net.weights_ho = weights_ho.to_vec();
        self.net.biases_o = biases_o.to_vec();
    }

    #[func]
    pub fn forward(&mut self, inputs: PackedFloat32Array) -> PackedFloat32Array {
        if self.scratch_hidden.len() != self.net.hidden_size {
            self.scratch_hidden.resize(self.net.hidden_size, 0.0);
        }
        if self.scratch_output.len() != self.net.output_size {
            self.scratch_output.resize(self.net.output_size, 0.0);
        }

        self.net.forward_into(
            inputs.as_slice(),
            &mut self.scratch_hidden,
            &mut self.scratch_output,
        );

        // Build PackedFloat32Array from slice — single allocation
        PackedFloat32Array::from(self.scratch_output.as_slice())
    }
}

#[godot_api]
impl IRefCounted for RustBatchSimulator {
    fn init(base: Base<RefCounted>) -> Self {
        Self { base }
    }
}

#[godot_api]
impl RustBatchSimulator {
    #[func]
    pub fn simulate_game(
        &self,
        white_weights: PackedFloat32Array,
        black_weights: PackedFloat32Array,
        input_size: i32,
        hidden_size: i32,
        output_size: i32,
        max_moves: i32,
    ) -> Dictionary<Variant, Variant> {
        let input_size = input_size as usize;
        let hidden_size = hidden_size as usize;
        let output_size = output_size as usize;
        let max_moves = max_moves as usize;

        let white = dense_from_flat_weights(
            input_size,
            hidden_size,
            output_size,
            white_weights.as_slice(),
        );
        let black = dense_from_flat_weights(
            input_size,
            hidden_size,
            output_size,
            black_weights.as_slice(),
        );

        let mut board = ChessBoard::startpos();
        let mut move_count = 0usize;
        let mut result: i32 = 2; // default draw

        let mut hidden = vec![0.0f32; hidden_size];
        let mut output = vec![0.0f32; output_size];
        let mut inputs = vec![0.0f32; 6 * 64 + 1 + 4];
        // Reuse pseudo-move buffer across iterations to avoid repeated allocation
        let mut pseudo_buf = Vec::with_capacity(256);

        while move_count < max_moves {
            let legal_moves = board.get_legal_moves_with_buf(&mut pseudo_buf);
            if legal_moves.is_empty() {
                if board.is_in_check(board.side_to_move) {
                    result = if board.side_to_move == 0 { -1 } else { 1 };
                } else {
                    result = 2;
                }
                break;
            }

            encode_board(&board, &mut inputs);
            let net = if board.side_to_move == 0 {
                &white
            } else {
                &black
            };
            net.forward_into(&inputs, &mut hidden, &mut output);
            let chosen = decode_move(&output, &legal_moves);
            board = board.make_move(chosen);
            move_count += 1;

            if board.halfmove_clock >= 100 {
                result = 2;
                break;
            }
        }

        let white_material = material_score(&board, 0);
        let black_material = material_score(&board, 1);
        let white_mobility = mobility_score(&board, 0);
        let black_mobility = mobility_score(&board, 1);
        let white_king_safety = king_safety_score(&board, 0);
        let black_king_safety = king_safety_score(&board, 1);

        let mut dict = Dictionary::new();
        let _ = dict.insert("result", result);
        let _ = dict.insert("move_count", move_count as i32);
        let _ = dict.insert("white_material", white_material);
        let _ = dict.insert("black_material", black_material);
        let _ = dict.insert("white_mobility", white_mobility);
        let _ = dict.insert("black_mobility", black_mobility);
        let _ = dict.insert("white_king_safety", white_king_safety);
        let _ = dict.insert("black_king_safety", black_king_safety);
        dict
    }
}

fn dense_from_flat_weights(
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    weights: &[f32],
) -> DenseNetwork {
    let ih = input_size * hidden_size;
    let bh = hidden_size;
    let ho = hidden_size * output_size;
    let bo = output_size;
    let total = ih + bh + ho + bo;
    let buf = vec![0.0f32; total];
    let src = if weights.len() >= total {
        weights
    } else {
        &buf
    };
    let mut cursor = 0usize;
    let weights_ih = take_from(src, &mut cursor, ih);
    let biases_h = take_from(src, &mut cursor, bh);
    let weights_ho = take_from(src, &mut cursor, ho);
    let biases_o = take_from(src, &mut cursor, bo);
    DenseNetwork::from_weights(
        input_size,
        hidden_size,
        output_size,
        weights_ih,
        biases_h,
        weights_ho,
        biases_o,
    )
}

fn take_from(src: &[f32], cursor: &mut usize, n: usize) -> Vec<f32> {
    let slice = &src[*cursor..*cursor + n];
    *cursor += n;
    slice.to_vec()
}

fn encode_board(board: &ChessBoard, out: &mut [f32]) {
    out.fill(0.0);
    let mut idx = 0usize;
    for piece_type in 1..=6 {
        for sq in 0..64 {
            let p = board.piece_at(sq);
            if p.abs() == piece_type as i8 {
                out[idx] = if p > 0 { 1.0 } else { -1.0 };
            }
            idx += 1;
        }
    }
    out[idx] = if board.side_to_move == 0 { 0.0 } else { 1.0 };
    idx += 1;
    out[idx] = if board.castling_rights & 0b0001 != 0 {
        1.0
    } else {
        0.0
    };
    idx += 1;
    out[idx] = if board.castling_rights & 0b0010 != 0 {
        1.0
    } else {
        0.0
    };
    idx += 1;
    out[idx] = if board.castling_rights & 0b0100 != 0 {
        1.0
    } else {
        0.0
    };
    idx += 1;
    out[idx] = if board.castling_rights & 0b1000 != 0 {
        1.0
    } else {
        0.0
    };
}

fn decode_move(outputs: &[f32], legal_moves: &[u32]) -> u32 {
    let mut best = legal_moves[0];
    let mut best_score = f32::NEG_INFINITY;
    for &mv in legal_moves {
        let from = ((mv >> 6) & 0x3f) as usize;
        let to = (mv & 0x3f) as usize;
        let from_score = if from < outputs.len() {
            outputs[from]
        } else {
            0.0
        };
        let to_score = if 64 + to < outputs.len() {
            outputs[64 + to]
        } else {
            0.0
        };
        let score = from_score + to_score;
        if score > best_score {
            best_score = score;
            best = mv;
        }
    }
    best
}

fn material_score(board: &ChessBoard, color: u8) -> f32 {
    // Use bitboard popcount instead of scanning all 64 squares — O(6) vs O(64).
    let values: [f32; 6] = [1.0, 3.0, 3.25, 5.0, 9.0, 0.0];
    let offset = if color == 0 { 0 } else { 6 };
    let mut total = 0.0f32;
    for i in 0..6 {
        total += board.bb[offset + i].0.count_ones() as f32 * values[i];
    }
    total
}

fn mobility_score(board: &ChessBoard, color: u8) -> i32 {
    let mut copy = *board;
    copy.side_to_move = color;
    let mut buf = Vec::with_capacity(256);
    copy.get_legal_moves_with_buf(&mut buf).len() as i32
}

fn king_safety_score(board: &ChessBoard, color: u8) -> f32 {
    // Use bitboard to find king in O(1) instead of scanning all 64 squares.
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
    let pawn_piece = if color == 0 { 1 } else { -1 };
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
