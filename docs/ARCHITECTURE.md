# Chess-Evolve Architecture

## Overview

Chess-Evolve uses **coevolutionary neuroevolution** to train chess-playing neural networks — two populations (white and black) compete against each other each generation, creating an arms race that drives improvement. The system is built on Godot 4 with an optional Rust GDExtension for performance, and a Python harness for overnight hyperparameter sweeps.

```
┌─────────────────────────────────────────────────────────────────┐
│                       Python Harness                            │
│  chess_sweep_worker.py  ←→  sweep_config.yaml  ←→  W&B        │
│  chess_monitor.py       ←→  shared-evolve-utils/               │
│  global_elite.py        (cross-run genome sharing)             │
└────────────────────────┬────────────────────────────────────────┘
                         │ JSON metrics / sweep_config.json
┌────────────────────────▼────────────────────────────────────────┐
│                      Godot 4 Engine                             │
│                                                                 │
│  scenes/main.gd                                                 │
│    ├── TrainingManager                                          │
│    │     ├── ChessEvolution (or ChessNeatEvolution)             │
│    │     │     ├── white_pop[pop_size]  NeuralNetwork           │
│    │     │     ├── black_pop[pop_size]  NeuralNetwork           │
│    │     │     ├── Hall of Fame (white + black, Elo-ranked)     │
│    │     │     └── Global Elite seed-in / contrib-out           │
│    │     ├── ChessFitness (per-game evaluation)                 │
│    │     ├── MetricsLogger → metrics.json + metrics.jsonl       │
│    │     └── GameRecorder (optional replay files)               │
│    ├── BoardState / BitboardState (game logic)                  │
│    ├── ChessEncoder (board → 389-float vector)                  │
│    └── UI: BoardRenderer, TrainingDashboard, ReplayViewer       │
│                                                                 │
│  Rust GDExtension (optional, ~3-6× faster)                     │
│    └── chess-native: move generation, NN eval, fitness          │
└─────────────────────────────────────────────────────────────────┘
```

## Component Breakdown

### GDScript Layer

| File | Role |
|------|------|
| `scenes/main.gd` | Entry point: UI mode, headless auto-train, replay viewer |
| `ai/training_manager.gd` | Game loop, coevolution orchestration, curriculum |
| `ai/evolution.gd` | Standard NE: two-population, Hall of Fame, adaptive mutation |
| `ai/neural_network.gd` | Feedforward net (input→hidden→output, tanh activations) |
| `ai/chess_neat_evolution.gd` | NEAT variant of the evolution engine |
| `ai/neat_network.gd` | Sparse network forward pass for NEAT genomes |
| `ai/fitness.gd` | Per-game fitness: win/draw/loss + material + mobility + king safety |
| `ai/metrics_logger.gd` | Writes `metrics.json` + `metrics.jsonl` per generation |
| `ai/game_recorder.gd` | Optional: records games to replay files |
| `ai/minimax_player.gd` | Minimax search opponent (for tournament mode) |
| `chess/board_state.gd` | Full chess game logic (move gen, make/unmake, detection) |
| `chess/bitboard_state.gd` | Bitboard-accelerated variant of board_state |
| `chess/encoder.gd` | Board → 389-float neural network input vector |
| `chess/constants.gd` | Piece enums, material values, piece symbols |

### Python Layer

| File | Role |
|------|------|
| `overnight-agent/chess_sweep_worker.py` | W&B sweep worker: config → Godot → poll → log |
| `overnight-agent/chess_monitor.py` | Monitor running workers, auto-spawn new ones |
| `overnight-agent/global_elite.py` | Cross-run genome pool (shared across parallel workers) |
| `overnight-agent/sweep_config.yaml` | Bayesian sweep configuration for W&B |
| `overnight-agent/start_workers.sh` | Shell helper: launch N workers |
| `shared-evolve-utils/godot_wandb.py` | Shared: Godot launch, metrics polling, W&B helpers |
| `shared-evolve-utils/worker_monitor.py` | Shared: generic worker monitoring and auto-spawn |

---

## Data Flow: One Generation

```
1. For each (white_idx, black_idx) pair in the tournament:
   a. Initialize fresh BoardState
   b. Loop (up to max_moves):
      - Encode board → 389-float vector
      - NeuralNetwork.forward(inputs) → 128 logits
      - ChessEncoder.decode_move(logits, legal_moves) → best legal move
      - board.make_move(move)
   c. ChessFitness.evaluate(result, material, mobility, king_safety) → score
   d. Accumulate score into white_fitness[w] and black_fitness[b]

2. ChessEvolution.evolve():
   a. Sort each population by fitness
   b. Elite individuals cloned directly
   c. Tournament selection + crossover/mutation for remainder
   d. Best individuals added to Hall of Fame (Elo-ranked)
   e. Adaptive mutation: tighten if improved, loosen if stagnant
   f. Optionally seed hall of fame from GlobalElitePool

3. MetricsLogger.write_metrics(stats)
   → overwrites metrics.json (latest snapshot)
   → appends to metrics.jsonl (full history)

4. After all generations: evolution.write_elite_contrib(worker_id)
   → contrib file harvested by Python, merged into GlobalElitePool
```

---

## File Layout

```
chess-evolve/
├── scenes/
│   ├── main.tscn / main.gd          # Entry point
│   └── ui/
│       ├── board_renderer.gd
│       ├── training_dashboard.gd
│       └── replay_viewer.gd
├── ai/
│   ├── neural_network.gd            # Standard feedforward net
│   ├── evolution.gd                 # Standard coevolution engine
│   ├── chess_neat_evolution.gd      # NEAT variant
│   ├── neat_network.gd / neat_*.gd  # NEAT infrastructure
│   ├── training_manager.gd          # Training orchestration
│   ├── fitness.gd                   # Fitness evaluation
│   ├── metrics_logger.gd            # Metrics I/O
│   ├── minimax_player.gd            # Minimax opponent
│   ├── game_recorder.gd             # Replay recording
│   └── chess_map_elites.gd          # MAP-Elites variant
├── chess/
│   ├── board_state.gd               # Game logic
│   ├── bitboard_state.gd            # Bitboard-accelerated logic
│   ├── encoder.gd                   # Board encoder (389 inputs)
│   └── constants.gd                 # Piece enums and values
├── rust/
│   └── chess-native/                # Rust GDExtension
│       └── src/
│           ├── board.rs             # Move generation
│           ├── godot_classes.rs     # GDScript-callable bindings
│           └── lib.rs
├── overnight-agent/
│   ├── chess_sweep_worker.py        # W&B sweep worker
│   ├── chess_monitor.py             # Worker monitor
│   ├── global_elite.py              # Cross-run elite pool
│   ├── sweep_config.yaml            # Sweep hyperparameters
│   └── start_workers.sh
├── tests/
│   ├── python/                      # pytest tests
│   └── test/                        # GDScript tests
└── docs/                            # This directory
```

---

## Rust GDExtension

The `chess-native` Rust extension (`gdext`) accelerates the hottest code paths. It exposes GDScript-callable classes:

| Class | Accelerates |
|-------|------------|
| Move generation | `board.rs` bitboard-based legal move enumeration |
| NN forward pass | Vectorised float arithmetic |
| Fitness evaluation | Material counting, mobility, king safety |

Build: `cargo build --release --manifest-path rust/chess-native/Cargo.toml`

The `BitboardState` GDScript class delegates move generation to the Rust extension when available, falling back to pure GDScript `BoardState`.
