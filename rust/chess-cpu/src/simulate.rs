use std::collections::HashSet;

use rand::SeedableRng;
use rayon::prelude::*;

use crate::board::ChessBoard;
use crate::encode::{decode_move, decode_move_factored, dense_from_flat_weights, encode_board};
use crate::evaluate::{king_safety_score, material_score, mobility_score};
use crate::sparse_nn::SparseNetwork;

/// Result of a single simulated game.
pub struct GameResult {
    pub white_idx: usize,
    pub black_idx: usize,
    /// 1 = white wins, -1 = black wins, 2 = draw
    pub result: i32,
    pub move_count: usize,
    pub white_material: f32,
    pub black_material: f32,
    pub white_mobility: i32,
    pub black_mobility: i32,
    pub white_king_safety: f32,
    pub black_king_safety: f32,
}

/// Compute a simple position hash from bitboards for threefold repetition detection.
///
/// Uses XOR + rotate to combine all 12 bitboards, side_to_move, castling, and en_passant
/// into a single u64.
fn position_hash(board: &ChessBoard) -> u64 {
    let mut h: u64 = 0;
    for i in 0..12 {
        h ^= board.bb[i].0;
        h = h.rotate_left(5);
    }
    h ^= board.side_to_move as u64;
    h = h.rotate_left(3);
    h ^= board.castling_rights as u64;
    h = h.rotate_left(3);
    h ^= (board.en_passant_sq as u64) & 0xff;
    h
}

/// Simulate a single chess game between two neural networks.
pub fn simulate_game(
    white_weights: &[f32],
    black_weights: &[f32],
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    max_moves: usize,
    temperature: f32,
    mercy_min_moves: usize,
    mercy_material_threshold: f32,
    rng: &mut impl rand::Rng,
) -> GameResult {
    let white_net = dense_from_flat_weights(input_size, hidden_size, output_size, white_weights);
    let black_net = dense_from_flat_weights(input_size, hidden_size, output_size, black_weights);

    let mut board = ChessBoard::startpos();
    let mut move_count = 0usize;
    let mut result: i32 = 2; // default draw

    let mut hidden = vec![0.0f32; hidden_size];
    let mut output = vec![0.0f32; output_size];
    let mut inputs = vec![0.0f32; input_size];
    let mut pseudo_buf = Vec::with_capacity(256);

    // Threefold repetition detection
    let mut position_hashes = HashSet::with_capacity(max_moves);

    while move_count < max_moves {
        let legal_moves = board.get_legal_moves_with_buf(&mut pseudo_buf);
        if legal_moves.is_empty() {
            if board.is_in_check(board.side_to_move) {
                result = if board.side_to_move == 0 { -1 } else { 1 };
            } else {
                result = 2; // stalemate
            }
            break;
        }

        // Threefold repetition check
        let hash = position_hash(&board);
        if !position_hashes.insert(hash) {
            // Hash already seen — count occurrences
            // For simplicity, treat any repeated position as a draw
            // (a true threefold needs 3 occurrences, but this is a reasonable
            // approximation that also catches infinite loops faster)
            result = 2;
            break;
        }

        encode_board(&board, &mut inputs);
        let net = if board.side_to_move == 0 {
            &white_net
        } else {
            &black_net
        };
        net.forward_into(&inputs, &mut hidden, &mut output);
        let chosen = decode_move(&output, &legal_moves, temperature, rng);
        board = board.make_move(chosen);
        move_count += 1;

        // 50-move rule
        if board.halfmove_clock >= 100 {
            result = 2;
            break;
        }

        // Mercy rule: if material difference is too large after minimum moves
        if mercy_material_threshold > 0.0 && move_count >= mercy_min_moves {
            let w_mat = material_score(&board, 0);
            let b_mat = material_score(&board, 1);
            let diff = (w_mat - b_mat).abs();
            if diff >= mercy_material_threshold {
                // The side with more material wins
                result = if w_mat > b_mat { 1 } else { -1 };
                break;
            }
        }
    }

    let white_material = material_score(&board, 0);
    let black_material = material_score(&board, 1);
    let white_mobility = mobility_score(&board, 0);
    let black_mobility = mobility_score(&board, 1);
    let white_king_safety = king_safety_score(&board, 0);
    let black_king_safety = king_safety_score(&board, 1);

    GameResult {
        white_idx: 0,
        black_idx: 0,
        result,
        move_count,
        white_material,
        black_material,
        white_mobility,
        black_mobility,
        white_king_safety,
        black_king_safety,
    }
}

/// Simulate a batch of games in parallel using rayon.
///
/// Populations are flat contiguous f32 slices: individual i's weights are at
/// `[i * genome_size .. (i+1) * genome_size]`.
pub fn simulate_games_batch_flat(
    white_flat: &[f32],
    black_flat: &[f32],
    genome_size: usize,
    pairings: &[(usize, usize)],
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    max_moves: usize,
    temperature: f32,
    mercy_min_moves: usize,
    mercy_material_threshold: f32,
) -> Vec<GameResult> {
    pairings
        .par_iter()
        .enumerate()
        .map(|(game_idx, &(w_idx, b_idx))| {
            let w_start = w_idx * genome_size;
            let b_start = b_idx * genome_size;
            let w_weights = &white_flat[w_start..w_start + genome_size];
            let b_weights = &black_flat[b_start..b_start + genome_size];

            // Each thread gets its own RNG seeded from game index for reproducibility
            let mut rng = rand::rngs::SmallRng::seed_from_u64(
                (game_idx as u64)
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407),
            );
            let mut gr = simulate_game(
                w_weights,
                b_weights,
                input_size,
                hidden_size,
                output_size,
                max_moves,
                temperature,
                mercy_min_moves,
                mercy_material_threshold,
                &mut rng,
            );
            gr.white_idx = w_idx;
            gr.black_idx = b_idx;
            gr
        })
        .collect()
}

/// Simulate a single chess game between two NEAT sparse networks.
pub fn simulate_neat_game(
    white_net: &SparseNetwork,
    black_net: &SparseNetwork,
    output_size: usize,
    max_moves: usize,
    temperature: f32,
    mercy_min_moves: usize,
    mercy_material_threshold: f32,
    rng: &mut impl rand::Rng,
) -> GameResult {
    let input_size = 389;
    let mut board = ChessBoard::startpos();
    let mut move_count = 0usize;
    let mut result: i32 = 2;

    let w_node_count = white_net.node_count;
    let b_node_count = black_net.node_count;
    let mut w_activations = vec![0.0f32; w_node_count];
    let mut b_activations = vec![0.0f32; b_node_count];
    let mut output = vec![0.0f32; output_size];
    let mut inputs = vec![0.0f32; input_size];
    let mut pseudo_buf = Vec::with_capacity(256);

    let mut position_hashes = HashSet::with_capacity(max_moves);

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

        let hash = position_hash(&board);
        if !position_hashes.insert(hash) {
            result = 2;
            break;
        }

        encode_board(&board, &mut inputs);
        if board.side_to_move == 0 {
            white_net.forward_into(&inputs, &mut w_activations, &mut output);
        } else {
            black_net.forward_into(&inputs, &mut b_activations, &mut output);
        }
        let chosen = if output_size <= 128 {
            decode_move_factored(&output, &legal_moves, temperature, rng)
        } else {
            decode_move(&output, &legal_moves, temperature, rng)
        };
        board = board.make_move(chosen);
        move_count += 1;

        if board.halfmove_clock >= 100 {
            result = 2;
            break;
        }

        if mercy_material_threshold > 0.0 && move_count >= mercy_min_moves {
            let w_mat = material_score(&board, 0);
            let b_mat = material_score(&board, 1);
            let diff = (w_mat - b_mat).abs();
            if diff >= mercy_material_threshold {
                result = if w_mat > b_mat { 1 } else { -1 };
                break;
            }
        }
    }

    GameResult {
        white_idx: 0,
        black_idx: 0,
        result,
        move_count,
        white_material: material_score(&board, 0),
        black_material: material_score(&board, 1),
        white_mobility: mobility_score(&board, 0),
        black_mobility: mobility_score(&board, 1),
        white_king_safety: king_safety_score(&board, 0),
        black_king_safety: king_safety_score(&board, 1),
    }
}

/// Simulate a batch of NEAT games in parallel using rayon.
///
/// Each rayon thread: parse genome JSON -> build SparseNetwork -> simulate game.
pub fn simulate_neat_games_batch(
    white_genomes_json: &[String],
    black_genomes_json: &[String],
    pairings: &[(usize, usize)],
    output_size: usize,
    max_moves: usize,
    temperature: f32,
    mercy_min_moves: usize,
    mercy_material_threshold: f32,
) -> Vec<GameResult> {
    pairings
        .par_iter()
        .enumerate()
        .map(|(game_idx, &(w_idx, b_idx))| {
            let w_json = &white_genomes_json[w_idx];
            let b_json = &black_genomes_json[b_idx];

            let w_net = SparseNetwork::from_genome_json(w_json)
                .expect("Failed to parse white genome JSON");
            let b_net = SparseNetwork::from_genome_json(b_json)
                .expect("Failed to parse black genome JSON");

            let mut rng = rand::rngs::SmallRng::seed_from_u64(
                (game_idx as u64)
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407),
            );

            let mut gr = simulate_neat_game(
                &w_net,
                &b_net,
                output_size,
                max_moves,
                temperature,
                mercy_min_moves,
                mercy_material_threshold,
                &mut rng,
            );
            gr.white_idx = w_idx;
            gr.black_idx = b_idx;
            gr
        })
        .collect()
}
