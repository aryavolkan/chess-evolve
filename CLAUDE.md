# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chess-Evolve uses **coevolutionary neuroevolution** to train chess-playing neural networks. Two populations (white and black) evolve against each other, creating an arms race. The system has three backends: Rust CPU (primary), PyTorch GPU/CPU, and Godot (original).

## Build & Test Commands

### Rust crates (PyO3 extensions for Python)
```bash
# Build all Rust crates (chess-cpu, evolve-ga, neat-ga)
cd rust/chess-cpu && maturin develop --release && cd ../..
cd rust/evolve-ga && maturin develop --release && cd ../..
cd rust/neat-ga && maturin develop --release && cd ../..

# Build Godot GDExtension (separate from PyO3 crates)
cargo build --release --manifest-path rust/chess-native/Cargo.toml

# Rust lint
cargo fmt --check --manifest-path rust/chess-native/Cargo.toml
cargo clippy --manifest-path rust/chess-native/Cargo.toml -- -D warnings
```

### Python
```bash
# Run all Python tests
python -m pytest tests/python -q

# Run a single test file
python -m pytest tests/python/test_cpu_trainer.py -v

# Lint
ruff check python/ scripts/ configs/ train_wandb.py
```

### GDScript
```bash
# Run GDScript tests (headless)
godot --headless --path . -s test/test_runner.gd

# Lint
./scripts/lint_gdscript.sh
```

### All lints at once
```bash
./scripts/lint_and_test.sh
```

### Training
```bash
# Single run (auto-detects backend: Rust > PyTorch GPU > PyTorch CPU > Godot)
python train_wandb.py

# With custom config
python train_wandb.py --config my_config.json

# Join a W&B sweep
python train_wandb.py --sweep <sweep-id>

# Chained runs (each seeds from previous best)
python train_wandb.py --chain 10
```

## Architecture

### Three-layer system

1. **Rust layer** (`rust/`): Four crates, all `cdylib`:
   - `chess-cpu` (PyO3): Parallel game simulation with bitboard move generation, NN forward pass, fitness metrics. Called from Python via `chess_cpu.simulate_games_batch()` / `chess_cpu.simulate_neat_games_batch()`.
   - `evolve-ga` (PyO3): GA operators — tournament selection, crossover, mutation, speciation, fitness sharing.
   - `neat-ga` (PyO3): NEAT evolution — variable-topology genomes, speciation, crossover, topology mutation.
   - `chess-native` (gdext): Godot GDExtension for accelerating the Godot-based training path.

2. **Python layer** (`python/`, `train_wandb.py`, `overnight-agent/`):
   - `python/cpu_trainer.py` / `python/neat_cpu_trainer.py`: Training loops using Rust PyO3 crates. Fixed-topology (CPUTrainer) vs variable-topology NEAT (NeatCPUTrainer).
   - `python/lichess_bot.py`: Lichess bot with ensemble voting from Hall of Fame genomes.
   - `train_wandb.py`: Entry point for all training (stays at root). Auto-detects backend, handles W&B logging, sweep integration, chained training.
   - `configs/`: Sweep configs and JSON training configs.
   - `overnight-agent/`: W&B sweep workers, monitoring, global elite pool for cross-run genome sharing.

3. **Godot layer** (`ai/`, `chess/`, `ui/`, `scenes/`):
   - `chess/`: Self-contained chess logic (board_state, encoder, constants). **No AI dependencies.**
   - `ai/`: Neural networks, evolution, fitness, training manager. **Depends on chess/.**
   - `ui/`: Board renderer, dashboard, replay viewer. **Depends on both.**

### Key data flow

Populations are numpy float32 arrays (`pop_size x genome_size`). Each generation: generate pairings -> `chess_cpu.simulate_games_batch()` (parallel Rust) -> compute fitness in Python -> evolve via `evolve_ga` or `neat_ga` -> log metrics to W&B.

### Neural network

- Input: 389 floats (6 piece planes x 64 squares, signed +/-1, plus side-to-move and castling)
- Hidden: 64 neurons, tanh (fixed-topology) or variable (NEAT)
- Output: 4096 logits (64x64 from-to pairs), masked to legal moves
- ~33K trainable parameters (fixed-topology)

### Fitness function

Primary objective is winning (win_bonus=10 + checkmate_bonus=10 = 20 total). Secondary signals: material difference (1.0x), opponent king safety exposure (1.5x), own king safety (0.5x), mobility (0.3x), move count penalty (-0.002x). The key optimization metric is `combined_best = min(white_best, black_best)` for balanced improvement.

### Benchmark system

A fixed random population (20 genomes) measures absolute progress, since coevolutionary metrics hide improvement when both sides improve simultaneously.

## Dependency rules

- `chess/` has NO AI dependencies — board logic is self-contained
- `ai/` depends on `chess/`
- `ui/` depends on both
- Rust PyO3 crates (`chess-cpu`, `evolve-ga`, `neat-ga`) are independent of each other and of `chess-native`

## Conventions

- GDScript + optional Rust — no C#, no external GDScript dependencies
- RefCounted base for non-scene GDScript classes
- GDScript tests go in `test/test_*.gd` extending `test_base.gd`, methods prefixed `test_`
- Python tests go in `tests/python/test_*.py`
- Piece types, values, symbols all in `chess/constants.gd`
- Signed piece planes (+/-1) instead of 12 separate planes
- 4096 output space (64x64 from-to), masked to legal moves
- Independent white/black evolution allows asymmetric specialization

## CI

Two workflows (`.github/workflows/`):
- `tests.yml`: ruff, gdlint, cargo fmt, cargo clippy -D warnings, pytest, Godot headless tests
- `pr-quality.yml`: Python lint+tests, GDScript lint

Run locally before PR: `./scripts/lint_and_test.sh`
