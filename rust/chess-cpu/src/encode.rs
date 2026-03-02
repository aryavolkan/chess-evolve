use crate::board::ChessBoard;
use crate::nn::DenseNetwork;

/// Encode board state into neural network input vector (389 floats).
///
/// Layout: 6 piece-type planes × 64 squares (signed: +1 white, -1 black),
/// then side_to_move, then 4 castling rights.
pub fn encode_board(board: &ChessBoard, out: &mut [f32]) {
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

/// Decode network output into a legal move.
///
/// When temperature <= 0.0, uses argmax (deterministic).
/// When > 0.0, uses softmax sampling with temperature for stochastic exploration.
pub fn decode_move(
    outputs: &[f32],
    legal_moves: &[u32],
    temperature: f32,
    rng: &mut impl rand::Rng,
) -> u32 {
    debug_assert!(!legal_moves.is_empty());

    // Collect scores for legal moves
    let scores: Vec<f32> = legal_moves
        .iter()
        .map(|&mv| {
            let from = ((mv >> 6) & 0x3f) as usize;
            let to = (mv & 0x3f) as usize;
            let idx = from * 64 + to;
            if idx < outputs.len() {
                outputs[idx]
            } else {
                0.0
            }
        })
        .collect();

    // Argmax (temperature <= 0 or single move)
    if temperature <= 0.0 || legal_moves.len() == 1 {
        let mut best_i = 0;
        let mut best_score = scores[0];
        for (i, &s) in scores.iter().enumerate().skip(1) {
            if s > best_score {
                best_score = s;
                best_i = i;
            }
        }
        return legal_moves[best_i];
    }

    // Softmax sampling with temperature
    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let inv_temp = 1.0 / temperature;

    let exp_vals: Vec<f32> = scores
        .iter()
        .map(|&s| ((s - max_score) * inv_temp).exp())
        .collect();
    let exp_sum: f32 = exp_vals.iter().sum();

    // Weighted random sample
    let r: f32 = rng.gen::<f32>() * exp_sum;
    let mut acc = 0.0;
    for (i, &e) in exp_vals.iter().enumerate() {
        acc += e;
        if r <= acc {
            return legal_moves[i];
        }
    }
    *legal_moves.last().unwrap()
}

/// Build a DenseNetwork from a flat weight vector.
///
/// Layout: [weights_ih | biases_h | weights_ho | biases_o]
pub fn dense_from_flat_weights(
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
    let weights_ih = src[cursor..cursor + ih].to_vec();
    cursor += ih;
    let biases_h = src[cursor..cursor + bh].to_vec();
    cursor += bh;
    let weights_ho = src[cursor..cursor + ho].to_vec();
    cursor += ho;
    let biases_o = src[cursor..cursor + bo].to_vec();
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
