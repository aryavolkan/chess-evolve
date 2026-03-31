use std::panic;

use rand::SeedableRng;
use rayon::prelude::*;

use crate::board::ChessBoard;
use crate::encode::{
    decode_move, decode_move_factored, decode_move_piece_dest, dense_from_flat_weights,
    encode_board,
};
use crate::evaluate::{king_danger_score, king_safety_score, material_score, mobility_score};
use crate::nn::DenseNetwork;
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
    pub white_king_danger: f32,
    pub black_king_danger: f32,
    pub white_captures_value: f32,
    pub black_captures_value: f32,
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

/// Material value of a captured piece (by absolute piece type).
fn capture_value(piece_abs: i8) -> f32 {
    match piece_abs {
        1 => 1.0,  // pawn
        2 => 3.0,  // knight
        3 => 3.25, // bishop
        4 => 5.0,  // rook
        5 => 9.0,  // queen
        _ => 0.0,
    }
}

/// Simulate a single chess game between two neural networks (from flat weight slices).
///
/// This is a convenience wrapper that builds `DenseNetwork` objects from flat weights
/// and delegates to `simulate_game_with_nets`. For batch simulation, prefer
/// `simulate_games_batch_flat` which pre-builds networks to avoid redundant copies.
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
    simulate_game_with_nets(
        &white_net,
        &black_net,
        input_size,
        hidden_size,
        output_size,
        max_moves,
        temperature,
        mercy_min_moves,
        mercy_material_threshold,
        rng,
    )
}

/// Simulate a single chess game between two pre-built dense neural networks.
pub fn simulate_game_with_nets(
    white_net: &DenseNetwork,
    black_net: &DenseNetwork,
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    max_moves: usize,
    temperature: f32,
    mercy_min_moves: usize,
    mercy_material_threshold: f32,
    rng: &mut impl rand::Rng,
) -> GameResult {
    let mut board = ChessBoard::startpos();
    let mut move_count = 0usize;
    let mut result: i32 = 2; // default draw

    let mut hidden = vec![0.0f32; hidden_size];
    let mut output = vec![0.0f32; output_size];
    let mut inputs = vec![0.0f32; input_size];
    let mut pseudo_buf = Vec::with_capacity(256);

    // Threefold repetition detection — Vec-based linear scan.
    // Positions rarely repeat so linear search over ~0-3 entries is faster than HashMap.
    let mut position_hashes: Vec<(u64, u8)> = Vec::with_capacity(16);

    // Mid-game king danger tracking (sampled every 10 moves)
    let mut w_danger_acc = 0.0f32;
    let mut b_danger_acc = 0.0f32;
    let mut danger_samples = 0u32;

    // Capture tracking
    let mut w_captures_value = 0.0f32;
    let mut b_captures_value = 0.0f32;

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
        if let Some(entry) = position_hashes.iter_mut().find(|(h, _)| *h == hash) {
            entry.1 += 1;
            if entry.1 >= 3 {
                result = 2;
                break;
            }
        } else {
            position_hashes.push((hash, 1));
        }

        encode_board(&board, &mut inputs);
        let net = if board.side_to_move == 0 {
            white_net
        } else {
            black_net
        };
        net.forward_into(&inputs, &mut hidden, &mut output);
        let chosen = decode_move(&output, &legal_moves, temperature, rng);

        // Track captures
        let to_sq = (chosen & 0x3f) as usize;
        let victim = board.piece_at(to_sq);
        if victim != 0 {
            let val = capture_value(victim.abs());
            if board.side_to_move == 0 {
                w_captures_value += val;
            } else {
                b_captures_value += val;
            }
        }
        // En passant capture (pawn not on destination square)
        if (chosen & 0x2000) != 0 {
            if board.side_to_move == 0 {
                w_captures_value += 1.0;
            } else {
                b_captures_value += 1.0;
            }
        }

        board = board.make_move(chosen);
        move_count += 1;

        // Sample king danger every 10 moves
        if move_count % 10 == 0 {
            w_danger_acc += king_danger_score(&board, 1); // danger white poses to black's king
            b_danger_acc += king_danger_score(&board, 0); // danger black poses to white's king
            danger_samples += 1;
        }

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

    // Also sample final position
    w_danger_acc += king_danger_score(&board, 1);
    b_danger_acc += king_danger_score(&board, 0);
    danger_samples += 1;

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
        white_king_danger: w_danger_acc / danger_samples.max(1) as f32,
        black_king_danger: b_danger_acc / danger_samples.max(1) as f32,
        white_captures_value: w_captures_value,
        black_captures_value: b_captures_value,
    }
}

/// Simulate a batch of games in parallel using rayon.
///
/// Populations are flat contiguous f32 slices: individual i's weights are at
/// `[i * genome_size .. (i+1) * genome_size]`.
///
/// Pre-builds `DenseNetwork` objects for all genomes before the parallel loop
/// to avoid redundant ~1.1 MB Vec copies when the same genome appears in
/// multiple pairings.
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
    // Pre-build all DenseNetworks (avoids redundant Vec copy per game)
    let n_white = white_flat.len() / genome_size;
    let n_black = black_flat.len() / genome_size;
    let white_nets: Vec<DenseNetwork> = (0..n_white)
        .map(|i| {
            let start = i * genome_size;
            dense_from_flat_weights(
                input_size,
                hidden_size,
                output_size,
                &white_flat[start..start + genome_size],
            )
        })
        .collect();
    let black_nets: Vec<DenseNetwork> = (0..n_black)
        .map(|i| {
            let start = i * genome_size;
            dense_from_flat_weights(
                input_size,
                hidden_size,
                output_size,
                &black_flat[start..start + genome_size],
            )
        })
        .collect();

    pairings
        .par_iter()
        .enumerate()
        .map(|(game_idx, &(w_idx, b_idx))| {
            let w_net = &white_nets[w_idx];
            let b_net = &black_nets[b_idx];

            // Each thread gets its own RNG seeded from game index for reproducibility
            let mut rng = rand::rngs::SmallRng::seed_from_u64(
                (game_idx as u64)
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407),
            );
            let mut gr = simulate_game_with_nets(
                w_net,
                b_net,
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

    let mut position_hashes: Vec<(u64, u8)> = Vec::with_capacity(16);

    let mut w_danger_acc = 0.0f32;
    let mut b_danger_acc = 0.0f32;
    let mut danger_samples = 0u32;

    // Capture tracking
    let mut w_captures_value = 0.0f32;
    let mut b_captures_value = 0.0f32;

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
        if let Some(entry) = position_hashes.iter_mut().find(|(h, _)| *h == hash) {
            entry.1 += 1;
            if entry.1 >= 3 {
                result = 2;
                break;
            }
        } else {
            position_hashes.push((hash, 1));
        }

        encode_board(&board, &mut inputs);
        if board.side_to_move == 0 {
            white_net.forward_into(&inputs, &mut w_activations, &mut output);
        } else {
            black_net.forward_into(&inputs, &mut b_activations, &mut output);
        }
        let chosen = if output_size <= 128 {
            decode_move_factored(&output, &legal_moves, temperature, rng)
        } else if output_size <= 384 {
            decode_move_piece_dest(&output, &legal_moves, &board, temperature, rng)
        } else {
            decode_move(&output, &legal_moves, temperature, rng)
        };

        // Track captures
        let to_sq = (chosen & 0x3f) as usize;
        let victim = board.piece_at(to_sq);
        if victim != 0 {
            let val = capture_value(victim.abs());
            if board.side_to_move == 0 {
                w_captures_value += val;
            } else {
                b_captures_value += val;
            }
        }
        if (chosen & 0x2000) != 0 {
            if board.side_to_move == 0 {
                w_captures_value += 1.0;
            } else {
                b_captures_value += 1.0;
            }
        }

        board = board.make_move(chosen);
        move_count += 1;

        // Sample king danger every 10 moves
        if move_count % 10 == 0 {
            w_danger_acc += king_danger_score(&board, 1);
            b_danger_acc += king_danger_score(&board, 0);
            danger_samples += 1;
        }

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

    // Also sample final position
    w_danger_acc += king_danger_score(&board, 1);
    b_danger_acc += king_danger_score(&board, 0);
    danger_samples += 1;

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
        white_king_danger: w_danger_acc / danger_samples.max(1) as f32,
        black_king_danger: b_danger_acc / danger_samples.max(1) as f32,
        white_captures_value: w_captures_value,
        black_captures_value: b_captures_value,
    }
}

/// Simulate a batch of NEAT games in parallel using rayon.
///
/// Pre-builds SparseNetwork objects for all unique genomes to avoid redundant
/// JSON parsing when the same genome appears in multiple pairings.
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
    // Pre-build all unique SparseNetworks (sequential JSON parse, avoids
    // redundant parsing when same genome appears in multiple pairings)
    let white_nets: Vec<Option<SparseNetwork>> = white_genomes_json
        .iter()
        .map(|json| SparseNetwork::from_genome_json(json))
        .collect();
    let black_nets: Vec<Option<SparseNetwork>> = black_genomes_json
        .iter()
        .map(|json| SparseNetwork::from_genome_json(json))
        .collect();

    pairings
        .par_iter()
        .enumerate()
        .map(|(game_idx, &(w_idx, b_idx))| {
            let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                let w_net = match &white_nets[w_idx] {
                    Some(net) => net,
                    None => {
                        return GameResult {
                            white_idx: w_idx,
                            black_idx: b_idx,
                            result: -1,
                            move_count: 0,
                            white_material: 0.0,
                            black_material: 39.0,
                            white_mobility: 0,
                            black_mobility: 20,
                            white_king_safety: 0.0,
                            black_king_safety: 0.0,
                            white_king_danger: 0.0,
                            black_king_danger: 0.0,
                            white_captures_value: 0.0,
                            black_captures_value: 0.0,
                        };
                    }
                };
                let b_net = match &black_nets[b_idx] {
                    Some(net) => net,
                    None => {
                        return GameResult {
                            white_idx: w_idx,
                            black_idx: b_idx,
                            result: 1,
                            move_count: 0,
                            white_material: 39.0,
                            black_material: 0.0,
                            white_mobility: 20,
                            black_mobility: 0,
                            white_king_safety: 0.0,
                            black_king_safety: 0.0,
                            white_king_danger: 0.0,
                            black_king_danger: 0.0,
                            white_captures_value: 0.0,
                            black_captures_value: 0.0,
                        };
                    }
                };

                let mut rng = rand::rngs::SmallRng::seed_from_u64(
                    (game_idx as u64)
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407),
                );

                let mut gr = simulate_neat_game(
                    w_net,
                    b_net,
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
            }));

            result.unwrap_or(GameResult {
                white_idx: w_idx,
                black_idx: b_idx,
                result: 2,
                move_count: 0,
                white_material: 0.0,
                black_material: 0.0,
                white_mobility: 0,
                black_mobility: 0,
                white_king_safety: 0.0,
                black_king_safety: 0.0,
                white_king_danger: 0.0,
                black_king_danger: 0.0,
                white_captures_value: 0.0,
                black_captures_value: 0.0,
            })
        })
        .collect()
}

/// Result of a single puzzle evaluation.
pub struct PuzzleResult {
    pub genome_idx: usize,
    pub correct: u32,
    pub total: u32,
    /// Soft score: sum of per-puzzle rank-based scores (0.0 to total).
    /// Each puzzle contributes 1.0 if correct, else (n_legal - rank) / n_legal
    /// where rank is the 0-indexed position of the correct move when sorted by
    /// network output (higher output = lower rank). Gives smooth evolutionary
    /// signal instead of binary correct/wrong.
    pub soft_score: f32,
}

/// Parse a UCI move string (e.g. "e2e4") into the internal move encoding.
/// Searches legal moves for a matching from-to pair.
fn uci_to_legal_move(uci: &str, legal_moves: &[u32]) -> Option<u32> {
    if uci.len() < 4 {
        return None;
    }
    let bytes = uci.as_bytes();
    let from_file = (bytes[0] - b'a') as u32;
    let from_rank = (bytes[1] - b'1') as u32;
    let to_file = (bytes[2] - b'a') as u32;
    let to_rank = (bytes[3] - b'1') as u32;
    let from_sq = from_rank * 8 + from_file;
    let to_sq = to_rank * 8 + to_file;

    legal_moves
        .iter()
        .find(|&&m| (m >> 6) & 0x3f == from_sq && m & 0x3f == to_sq)
        .copied()
}

/// Score a single legal move using the appropriate output encoding.
fn score_move(mv: u32, output: &[f32], output_size: usize, board: &ChessBoard) -> f32 {
    let from = ((mv >> 6) & 0x3f) as usize;
    let to = (mv & 0x3f) as usize;
    if output_size <= 128 {
        // Factored: from-score + to-score
        let fs = if from < output.len() {
            output[from]
        } else {
            0.0
        };
        let ts = if 64 + to < output.len() {
            output[64 + to]
        } else {
            0.0
        };
        fs + ts
    } else if output_size <= 384 {
        // Piece × destination
        let piece_abs = board.piece_at(from).unsigned_abs() as usize;
        let piece_idx = if (1..=6).contains(&piece_abs) {
            piece_abs - 1
        } else {
            0
        };
        let idx = piece_idx * 64 + to;
        if idx < output.len() {
            output[idx]
        } else {
            0.0
        }
    } else {
        // Full 4096 from×to
        let idx = from * 64 + to;
        if idx < output.len() {
            output[idx]
        } else {
            0.0
        }
    }
}

/// Evaluate a batch of NEAT genomes on chess puzzles in parallel.
///
/// Each puzzle is a (FEN, best_move_uci) pair. Each genome is scored on how
/// many puzzles it solves (picks the correct move). Returns per-genome scores
/// with both binary `correct` count and continuous `soft_score`.
pub fn evaluate_puzzles_batch(
    genomes_json: &[String],
    fens: &[String],
    best_moves: &[String],
    output_size: usize,
) -> Vec<PuzzleResult> {
    let input_size = 389;
    let total = fens.len() as u32;

    // Pre-parse all puzzle positions
    let puzzles: Vec<_> = fens
        .iter()
        .zip(best_moves.iter())
        .filter_map(|(fen, uci)| {
            let board = ChessBoard::from_fen(fen)?;
            Some((board, uci.clone()))
        })
        .collect();
    let actual_total = puzzles.len() as u32;

    genomes_json
        .par_iter()
        .enumerate()
        .map(|(idx, json)| {
            let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                let net = match SparseNetwork::from_genome_json(json) {
                    Some(n) => n,
                    None => {
                        return PuzzleResult {
                            genome_idx: idx,
                            correct: 0,
                            total: actual_total,
                            soft_score: 0.0,
                        };
                    }
                };

                let mut inputs = vec![0.0f32; input_size];
                let mut output = vec![0.0f32; output_size];
                let mut activations = vec![0.0f32; net.node_count];
                let mut pseudo_buf = Vec::with_capacity(256);
                let mut correct = 0u32;
                let mut soft_score = 0.0f32;

                for (board, expected_uci) in &puzzles {
                    // Encode position
                    encode_board(board, &mut inputs);

                    // Forward pass
                    net.forward_into(&inputs, &mut activations, &mut output);

                    // Get legal moves
                    let legal_moves = board.get_legal_moves_with_buf(&mut pseudo_buf);
                    if legal_moves.is_empty() {
                        continue;
                    }

                    let expected = match uci_to_legal_move(expected_uci, &legal_moves) {
                        Some(m) => m,
                        None => continue,
                    };

                    // Score all legal moves
                    let expected_score = score_move(expected, &output, output_size, board);

                    // Count how many legal moves score strictly higher than the correct one
                    let n_legal = legal_moves.len() as f32;
                    let mut rank = 0u32; // 0 = best
                    for &mv in &legal_moves {
                        let s = score_move(mv, &output, output_size, board);
                        if s > expected_score {
                            rank += 1;
                        }
                    }

                    if rank == 0 {
                        // Correct move has highest (or tied-highest) score
                        correct += 1;
                        soft_score += 1.0;
                    } else {
                        // Partial credit: higher rank = more credit
                        // rank=1 (2nd best) → (n-1)/n ≈ 0.97 for 30 moves
                        // rank=n-1 (worst) → 1/n ≈ 0.03
                        soft_score += (n_legal - rank as f32) / n_legal;
                    }
                }

                PuzzleResult {
                    genome_idx: idx,
                    correct,
                    total: actual_total,
                    soft_score,
                }
            }));

            result.unwrap_or(PuzzleResult {
                genome_idx: idx,
                correct: 0,
                total,
                soft_score: 0.0,
            })
        })
        .collect()
}
