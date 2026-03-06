# Chess-Evolve Architecture

## Overview

Chess-Evolve uses **coevolutionary neuroevolution** to train chess-playing neural networks — two populations (white and black) compete against each other each generation, creating an arms race that drives improvement. The system has three backends: Rust CPU (primary), PyTorch GPU/CPU, and Godot (original).

```
┌─────────────────────────────────────────────────────────────────┐
│                       Python Harness                            │
│  train_wandb.py  (auto-detects backend, W&B logging, sweeps)   │
│  cpu_trainer.py / neat_cpu_trainer.py  (training loops)         │
│  sweep_config.py  (hyperparameter sweep configs)                │
│  lichess_bot.py   (play evolved genomes on Lichess)             │
│  overnight-agent/ (sweep workers, monitor, global elite pool)   │
└────────────────────────┬────────────────────────────────────────┘
                         │ numpy arrays / PyO3 bindings
┌────────────────────────▼────────────────────────────────────────┐
│                    Rust PyO3 Crates                              │
│                                                                  │
│  chess-cpu:   Parallel game simulation, bitboard move gen,       │
│               NN forward pass, fitness metrics                   │
│  evolve-ga:   Tournament selection, crossover, mutation,         │
│               speciation, fitness sharing                        │
│  neat-ga:     NEAT evolution — variable-topology genomes,        │
│               speciation, crossover, topology mutation            │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                    Godot 4 Engine (optional UI)                   │
│                                                                  │
│  scenes/main.gd                                                  │
│    ├── TrainingManager                                           │
│    │     ├── ChessEvolution (or ChessNeatEvolution)              │
│    │     ├── ChessFitness (per-game evaluation)                  │
│    │     └── MetricsLogger → metrics.json                        │
│    ├── BoardState / BitboardState (game logic)                   │
│    ├── ChessEncoder (board → 389-float vector)                   │
│    └── UI: BoardRenderer, TrainingDashboard, ReplayViewer        │
│                                                                  │
│  Rust GDExtension (chess-native, ~3-6x faster than GDScript)    │
│    └── move generation, NN eval, fitness                         │
└──────────────────────────────────────────────────────────────────┘
```

## Component Breakdown

### Python Layer (primary training path)

| File | Role |
|------|------|
| `train_wandb.py` | Entry point: auto-detects backend, W&B logging, sweep integration, chained runs |
| `cpu_trainer.py` | Fixed-topology training loop using Rust PyO3 crates |
| `neat_cpu_trainer.py` | NEAT variable-topology training loop using Rust PyO3 crates |
| `sweep_config.py` | Bayesian, grid, and deep sweep configurations |
| `lichess_bot.py` | Lichess bot — play evolved NEAT genomes online |
| `godot_wandb.py` | Godot subprocess integration (fallback backend) |

### Rust PyO3 Crates

| Crate | Role |
|-------|------|
| `chess-cpu` | Parallel game simulation with bitboard move gen, NN forward pass, fitness metrics. Exposes `simulate_games_batch()` / `simulate_neat_games_batch()` |
| `evolve-ga` | GA operators: tournament selection, two-point crossover, Gaussian mutation, speciation, fitness sharing, island migration |
| `neat-ga` | NEAT evolution: variable-topology genomes, innovation tracking, speciation, topology mutation (add node/connection), crossover by historical alignment |

### GDScript Layer

| File | Role |
|------|------|
| `scenes/main.gd` | Entry point: UI mode, headless auto-train, replay viewer |
| `ai/training_manager.gd` | Game loop, coevolution orchestration, curriculum |
| `ai/evolution.gd` | Standard NE: two-population, Hall of Fame, adaptive mutation |
| `ai/neural_network.gd` | Feedforward net (input -> hidden -> output, tanh) |
| `ai/chess_neat_evolution.gd` | NEAT variant of the evolution engine |
| `ai/fitness.gd` | Per-game fitness: win/draw/loss + material + mobility + king safety |
| `chess/board_state.gd` | Full chess game logic (move gen, make/unmake, detection) |
| `chess/encoder.gd` | Board -> 389-float neural network input vector |
| `chess/constants.gd` | Piece enums, material values, piece symbols |

### Overnight Agent (sweep workers)

| File | Role |
|------|------|
| `overnight-agent/chess_sweep_worker.py` | W&B sweep worker: config -> Godot -> poll -> log |
| `overnight-agent/chess_monitor.py` | Monitor running workers, auto-spawn new ones |
| `overnight-agent/global_elite.py` | Cross-run genome pool (shared across parallel workers) |

---

## Data Flow: One Generation (Rust CPU Backend)

```
1. Python generates pairings (white_idx, black_idx)

2. chess_cpu.simulate_games_batch(white_pop, black_pop, pairings, max_moves):
   For each pairing (parallel across CPU cores):
     a. Initialize fresh bitboard position
     b. Loop (up to max_moves):
        - Encode board → 389-float vector
        - NN forward pass → 4096 logits
        - Mask to legal moves, pick highest → best legal move
        - Apply move
     c. Compute fitness components (win/loss, material, mobility, king safety)
     d. Return per-game results

3. Python computes final fitness from game results

4. evolve_ga / neat_ga operators:
   a. Sort by fitness, preserve elites
   b. Tournament selection + crossover + mutation
   c. Immigration (random individuals for diversity)
   d. Fitness sharing (reward genetic distance)

5. Log metrics to W&B, save best genomes
6. Repeat
```

---

## File Layout

```
chess-evolve/
├── train_wandb.py                 # Main entry point
├── cpu_trainer.py                 # Fixed-topology trainer (Rust backend)
├── neat_cpu_trainer.py            # NEAT trainer (Rust backend)
├── sweep_config.py                # W&B sweep configs
├── lichess_bot.py                 # Lichess bot integration
├── godot_wandb.py                 # Godot subprocess integration
├── rust/
│   ├── chess-cpu/                 # PyO3: game simulation
│   ├── evolve-ga/                 # PyO3: GA operators
│   ├── neat-ga/                   # PyO3: NEAT evolution
│   └── chess-native/              # Godot GDExtension
├── ai/
│   ├── neural_network.gd          # Standard feedforward net
│   ├── evolution.gd               # Standard coevolution engine
│   ├── chess_neat_evolution.gd    # NEAT variant
│   ├── training_manager.gd        # Training orchestration
│   ├── fitness.gd                 # Fitness evaluation
│   └── minimax_player.gd          # Minimax opponent
├── chess/
│   ├── board_state.gd             # Game logic
│   ├── bitboard_state.gd          # Bitboard-accelerated logic
│   ├── encoder.gd                 # Board encoder (389 inputs)
│   └── constants.gd               # Piece enums and values
├── overnight-agent/
│   ├── chess_sweep_worker.py      # W&B sweep worker
│   ├── chess_monitor.py           # Worker monitor
│   └── global_elite.py            # Cross-run elite pool
├── scripts/                       # Lint, test, utility scripts
├── test/                          # GDScript tests
├── tests/python/                  # Python pytest tests
├── scenes/                        # Godot scenes
├── ui/                            # Godot UI components
└── docs/                          # This directory
```

---

## Backend Selection

`train_wandb.py` auto-detects the best available backend:

| Priority | Backend | Requirements | Speed |
|----------|---------|-------------|-------|
| 1 | Rust CPU | `chess_cpu` + `evolve_ga` PyO3 crates | Fastest |
| 2 | PyTorch GPU | `torch` with CUDA, `python-chess` | Fast |
| 3 | PyTorch CPU | `torch` (CPU-only), `python-chess` | Medium |
| 4 | Godot | Just `godot` binary | Slowest |

### Rust GDExtension

The `chess-native` Rust extension (`gdext`) accelerates the Godot training path:

| Accelerates | Implementation |
|------------|----------------|
| Move generation | `board.rs` bitboard-based legal move enumeration |
| NN forward pass | Vectorised float arithmetic |
| Fitness evaluation | Material counting, mobility, king safety |

Build: `cargo build --release --manifest-path rust/chess-native/Cargo.toml`
