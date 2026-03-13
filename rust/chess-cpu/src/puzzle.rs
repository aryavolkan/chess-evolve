use rayon::prelude::*;
use serde::Deserialize;

use crate::board::{uci_to_move, ChessBoard};
use crate::encode::{decode_move, decode_move_factored, decode_move_piece_dest, encode_board};
use crate::sparse_nn::SparseNetwork;

#[derive(Deserialize)]
struct Puzzle {
    fen: String,
    solution: String,
}

/// Evaluate NEAT genomes on chess puzzles.
///
/// Returns a `Vec<f32>` of average puzzle scores per genome.
///
/// Scoring per puzzle:
///   - exact move match (same from + to): 1.0
///   - same source square: 0.3
///   - network's move captures a piece (but wrong move): 0.2
///   - otherwise: 0.0
pub fn evaluate_puzzles_batch(
    genomes_json: &[String],
    puzzles_json: &str,
    output_size: usize,
    temperature: f32,
) -> Vec<f32> {
    let puzzles: Vec<Puzzle> = match serde_json::from_str(puzzles_json) {
        Ok(p) => p,
        Err(_) => return vec![0.0; genomes_json.len()],
    };

    if puzzles.is_empty() {
        return vec![0.0; genomes_json.len()];
    }

    // Pre-parse all puzzle positions and solutions
    let parsed: Vec<Option<(ChessBoard, u32)>> = puzzles
        .iter()
        .map(|p| {
            let board = ChessBoard::from_fen(&p.fen)?;
            let solution_mv = uci_to_move(&p.solution, &board)?;
            Some((board, solution_mv))
        })
        .collect();

    let valid_count = parsed.iter().filter(|p| p.is_some()).count();
    if valid_count == 0 {
        return vec![0.0; genomes_json.len()];
    }

    // Parallel over genomes
    genomes_json
        .par_iter()
        .map(|genome_json| {
            let net = match SparseNetwork::from_genome_json(genome_json) {
                Some(n) => n,
                None => return 0.0,
            };

            let mut activations = vec![0.0f32; net.node_count];
            let mut output = vec![0.0f32; output_size];
            let mut inputs = vec![0.0f32; 389];
            let mut pseudo_buf = Vec::with_capacity(256);
            let mut rng = rand::rngs::SmallRng::from_entropy();
            let mut total_score = 0.0f32;
            let mut puzzles_evaluated = 0u32;

            for parsed_puzzle in &parsed {
                let (board, solution_mv) = match parsed_puzzle {
                    Some((b, s)) => (b, *s),
                    None => continue,
                };

                let legal_moves = board.get_legal_moves_with_buf(&mut pseudo_buf);
                if legal_moves.is_empty() {
                    continue;
                }

                // Check that the solution is actually legal
                if !legal_moves.contains(&solution_mv) {
                    continue;
                }

                encode_board(board, &mut inputs);
                net.forward_into(&inputs, &mut activations, &mut output);

                let chosen = if output_size <= 128 {
                    decode_move_factored(&output, &legal_moves, temperature, &mut rng)
                } else if output_size <= 384 {
                    decode_move_piece_dest(&output, &legal_moves, board, temperature, &mut rng)
                } else {
                    decode_move(&output, &legal_moves, temperature, &mut rng)
                };

                let solution_from = (solution_mv >> 6) & 0x3f;
                let solution_to = solution_mv & 0x3f;
                let chosen_from = (chosen >> 6) & 0x3f;
                let chosen_to = chosen & 0x3f;

                let score = if chosen_from == solution_from && chosen_to == solution_to {
                    1.0 // Exact match
                } else if chosen_from == solution_from {
                    0.3 // Right piece, wrong destination
                } else {
                    // Check if network's move at least captures something
                    let to_sq = chosen_to as usize;
                    let victim = board.piece_at(to_sq);
                    if victim != 0 {
                        0.2 // Captured something (but wrong move)
                    } else {
                        0.0
                    }
                };

                total_score += score;
                puzzles_evaluated += 1;
            }

            if puzzles_evaluated > 0 {
                total_score / puzzles_evaluated as f32
            } else {
                0.0
            }
        })
        .collect()
}

use rand::SeedableRng;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_puzzles() {
        let genomes = vec!["{}".to_string()];
        let result = evaluate_puzzles_batch(&genomes, "[]", 384, 0.1);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 0.0);
    }

    #[test]
    fn test_invalid_puzzle_json() {
        let genomes = vec!["{}".to_string()];
        let result = evaluate_puzzles_batch(&genomes, "not json", 384, 0.1);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 0.0);
    }

    #[test]
    fn test_uci_to_move_basic() {
        let board = ChessBoard::startpos();
        let mv = uci_to_move("e2e4", &board).unwrap();
        let from = ((mv >> 6) & 0x3f) as usize;
        let to = (mv & 0x3f) as usize;
        assert_eq!(from, 12); // e2 = file 4, rank 1 = 1*8+4 = 12
        assert_eq!(to, 28); // e4 = file 4, rank 3 = 3*8+4 = 28
    }

    #[test]
    fn test_uci_to_move_promotion() {
        // White pawn on e7, black king on a8
        let board = ChessBoard::from_fen("k7/4P3/8/8/8/8/8/K7 w - - 0 1").unwrap();
        let mv = uci_to_move("e7e8q", &board).unwrap();
        // Should have promotion flag
        assert_ne!(mv & (1 << 12), 0);
    }

    #[test]
    fn test_uci_to_move_invalid() {
        let board = ChessBoard::startpos();
        assert!(uci_to_move("zz", &board).is_none());
        assert!(uci_to_move("", &board).is_none());
    }
}
