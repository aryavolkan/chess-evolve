#![allow(clippy::needless_range_loop)]

use std::panic;

use pyo3::prelude::*;
use pyo3::types::PyDict;

mod board;
mod encode;
mod evaluate;
mod nn;
mod simulate;
mod sparse_nn;

/// Reinterpret a byte slice as f32 slice (native endian).
fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    assert!(
        bytes.len() % 4 == 0,
        "byte buffer length must be a multiple of 4"
    );
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Simulate a single chess game between two neural networks.
///
/// Returns a dict with: result, move_count, white_material, black_material,
/// white_mobility, black_mobility, white_king_safety, black_king_safety.
#[pyfunction]
#[pyo3(signature = (
    white_weights,
    black_weights,
    input_size = 389,
    hidden_size = 64,
    output_size = 4096,
    max_moves = 100,
    temperature = 0.5,
    mercy_min_moves = 30,
    mercy_material_threshold = 20.0,
))]
fn simulate_game(
    py: Python<'_>,
    white_weights: Vec<f32>,
    black_weights: Vec<f32>,
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    max_moves: usize,
    temperature: f32,
    mercy_min_moves: usize,
    mercy_material_threshold: f32,
) -> PyResult<Py<PyDict>> {
    let mut rng = rand::thread_rng();
    let gr = simulate::simulate_game(
        &white_weights,
        &black_weights,
        input_size,
        hidden_size,
        output_size,
        max_moves,
        temperature,
        mercy_min_moves,
        mercy_material_threshold,
        &mut rng,
    );
    let dict = PyDict::new(py);
    dict.set_item("result", gr.result)?;
    dict.set_item("move_count", gr.move_count)?;
    dict.set_item("white_material", gr.white_material)?;
    dict.set_item("black_material", gr.black_material)?;
    dict.set_item("white_mobility", gr.white_mobility)?;
    dict.set_item("black_mobility", gr.black_mobility)?;
    dict.set_item("white_king_safety", gr.white_king_safety)?;
    dict.set_item("black_king_safety", gr.black_king_safety)?;
    Ok(dict.into())
}

/// Simulate a batch of chess games in parallel using rayon.
///
/// Populations are passed as bytes (raw f32 little-endian) for memory efficiency.
/// Each population is a flat contiguous buffer: individual i's weights are at
/// byte offset `i * genome_size * 4`.
///
/// Arguments:
///     white_pop_bytes: bytes buffer of white population weights (numpy .tobytes())
///     black_pop_bytes: bytes buffer of black population weights (numpy .tobytes())
///     genome_size: number of f32 weights per individual
///     pairings: list of (white_idx, black_idx) tuples
///     input_size, hidden_size, output_size: network dimensions
///     max_moves: maximum moves per game
///     temperature: softmax temperature for move selection
///     mercy_min_moves: minimum moves before mercy rule applies
///     mercy_material_threshold: material difference threshold for mercy rule
///
/// Returns: list of dicts, each with white_idx, black_idx, result, move_count,
///          white_material, black_material, white_mobility, black_mobility,
///          white_king_safety, black_king_safety.
#[pyfunction]
#[pyo3(signature = (
    white_pop_bytes,
    black_pop_bytes,
    genome_size,
    pairings,
    input_size = 389,
    hidden_size = 64,
    output_size = 4096,
    max_moves = 100,
    temperature = 0.5,
    mercy_min_moves = 30,
    mercy_material_threshold = 20.0,
))]
fn simulate_games_batch(
    py: Python<'_>,
    white_pop_bytes: Vec<u8>,
    black_pop_bytes: Vec<u8>,
    genome_size: usize,
    pairings: Vec<(usize, usize)>,
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    max_moves: usize,
    temperature: f32,
    mercy_min_moves: usize,
    mercy_material_threshold: f32,
) -> PyResult<Vec<Py<PyDict>>> {
    // Convert bytes to f32 slices (compact: 4 bytes/float vs 28 bytes/Python float)
    let white_flat = bytes_to_f32_vec(&white_pop_bytes);
    drop(white_pop_bytes); // free the byte buffer immediately
    let black_flat = bytes_to_f32_vec(&black_pop_bytes);
    drop(black_pop_bytes);

    // Release the GIL during the heavy computation
    let results = py.detach(move || {
        simulate::simulate_games_batch_flat(
            &white_flat,
            &black_flat,
            genome_size,
            &pairings,
            input_size,
            hidden_size,
            output_size,
            max_moves,
            temperature,
            mercy_min_moves,
            mercy_material_threshold,
        )
    });

    // Convert results to Python dicts (must hold GIL)
    results
        .into_iter()
        .map(|gr| {
            let dict = PyDict::new(py);
            dict.set_item("white_idx", gr.white_idx)?;
            dict.set_item("black_idx", gr.black_idx)?;
            dict.set_item("result", gr.result)?;
            dict.set_item("move_count", gr.move_count)?;
            dict.set_item("white_material", gr.white_material)?;
            dict.set_item("black_material", gr.black_material)?;
            dict.set_item("white_mobility", gr.white_mobility)?;
            dict.set_item("black_mobility", gr.black_mobility)?;
            dict.set_item("white_king_safety", gr.white_king_safety)?;
            dict.set_item("black_king_safety", gr.black_king_safety)?;
            Ok(dict.into())
        })
        .collect()
}

/// Simulate a batch of NEAT chess games in parallel using rayon.
///
/// NEAT genomes are variable-topology, passed as JSON strings.
/// Each rayon thread: parse JSON -> build SparseNetwork -> simulate game.
#[pyfunction]
#[pyo3(signature = (
    white_genomes,
    black_genomes,
    pairings,
    output_size = 4096,
    max_moves = 100,
    temperature = 0.5,
    mercy_min_moves = 30,
    mercy_material_threshold = 20.0,
))]
fn simulate_neat_games_batch(
    py: Python<'_>,
    white_genomes: Vec<String>,
    black_genomes: Vec<String>,
    pairings: Vec<(usize, usize)>,
    output_size: usize,
    max_moves: usize,
    temperature: f32,
    mercy_min_moves: usize,
    mercy_material_threshold: f32,
) -> PyResult<Vec<Py<PyDict>>> {
    // Release the GIL during heavy computation.
    // Wrap in catch_unwind so ANY panic returns empty results instead of crashing.
    let pairings_clone = pairings.clone();
    let results = py.detach(move || {
        let batch_result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            simulate::simulate_neat_games_batch(
                &white_genomes,
                &black_genomes,
                &pairings,
                output_size,
                max_moves,
                temperature,
                mercy_min_moves,
                mercy_material_threshold,
            )
        }));
        match batch_result {
            Ok(results) => results,
            Err(_) => {
                // Entire batch panicked — return all draws
                pairings_clone
                    .iter()
                    .map(|&(w_idx, b_idx)| simulate::GameResult {
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
                    })
                    .collect()
            }
        }
    });

    results
        .into_iter()
        .map(|gr| {
            let dict = PyDict::new(py);
            dict.set_item("white_idx", gr.white_idx)?;
            dict.set_item("black_idx", gr.black_idx)?;
            dict.set_item("result", gr.result)?;
            dict.set_item("move_count", gr.move_count)?;
            dict.set_item("white_material", gr.white_material)?;
            dict.set_item("black_material", gr.black_material)?;
            dict.set_item("white_mobility", gr.white_mobility)?;
            dict.set_item("black_mobility", gr.black_mobility)?;
            dict.set_item("white_king_safety", gr.white_king_safety)?;
            dict.set_item("black_king_safety", gr.black_king_safety)?;
            Ok(dict.into())
        })
        .collect()
}

/// Encode a FEN string into the neural network input vector (389 floats).
/// Useful for debugging and testing.
#[pyfunction]
fn encode_board_fen(fen: &str) -> PyResult<Vec<f32>> {
    let board = board::ChessBoard::from_fen(fen).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("Invalid FEN: {}", fen))
    })?;
    let mut out = vec![0.0f32; 389];
    encode::encode_board(&board, &mut out);
    Ok(out)
}

/// The Python module exposed by this crate.
#[pymodule]
fn chess_cpu(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate_game, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_games_batch, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_neat_games_batch, m)?)?;
    m.add_function(wrap_pyfunction!(encode_board_fen, m)?)?;
    Ok(())
}
