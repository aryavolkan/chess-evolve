# Chess Evolve

Evolutionary neural networks learn to play chess through coevolution. Two populations of neural networks — one playing white, one playing black — evolve against each other. Networks that win more games survive and reproduce. Over generations, both populations develop increasingly sophisticated chess strategies.

## Architecture

The system has three backends: **Rust CPU** (primary), **PyTorch GPU/CPU**, and **Godot** (original).

```
┌─────────────────────────────────────────────────────────┐
│                    Python Harness                        │
│  train_wandb.py ──► auto-detect backend                 │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ cpu_trainer  │  │ neat_cpu_    │  │ lichess_bot   │  │
│  │   .py        │  │ trainer.py   │  │   .py         │  │
│  └──────┬───────┘  └──────┬───────┘  └───────────────┘  │
│         │                 │                              │
│  ┌──────▼─────────────────▼──────┐                      │
│  │     Rust PyO3 Crates          │                      │
│  │  chess-cpu  evolve-ga  neat-ga│                      │
│  └───────────────────────────────┘                      │
│                                                          │
│  Populations (numpy float32) ──► Rust simulation         │
│  ──► fitness ──► evolve ──► W&B logging                  │
└─────────────────────────────────────────────────────────┘
```

### Neural Network

- **Input (389):** 6 piece-type planes x 64 squares (signed +/-1), side to move, castling rights
- **Hidden:** 64 neurons, tanh activation
- **Output (4096):** one logit per (from_square x 64 + to_square) pair
- **Move selection:** score = output[from*64 + to], masked to legal moves, pick highest
- **~33K trainable parameters**

### Fitness Function

| Component | Weight | Description |
|-----------|--------|-------------|
| Win | 10.0 | Bonus for winning the game |
| Checkmate | 10.0 | Extra bonus for checkmate |
| Draw | 0.0 | No bonus (material advantage provides small signal) |
| Loss | -5.0 | Penalty for losing |
| Material | 1.0x | Net material advantage |
| Mobility | 0.3x | Legal move count |
| Own King Safety | 0.5x | Friendly pawns near king |
| Opp King Safety | 1.5x | Reward for attacking opponent's king |
| King Danger | 1.0x | King danger score (attack signals) |
| Move Count | -0.002x | Penalty per move (encourages decisive play) |

Primary optimization metric: `combined_best = min(white_best, black_best)` for balanced improvement.

### Evolution

- Tournament selection (k=2)
- Two-point crossover (70% rate)
- Gaussian mutation (rate=0.15, sigma=0.2)
- Elitism (top 2 preserved)
- Independent evolution for white and black populations
- Fitness sharing (sigma=0.08) for diversity
- Immigration (10% random replacement per generation)

### Benchmark System

A fixed random population (20 genomes) measures absolute progress, since coevolutionary metrics hide improvement when both sides improve simultaneously. Seeds are saved independently per color based on per-color benchmark win rate.

## Project Structure

```
chess-evolve/
├── train_wandb.py                 # Main entry point (auto-detects backend)
├── python/
│   ├── cpu_trainer.py             # Fixed-topology training loop (Rust backend)
│   ├── neat_cpu_trainer.py        # NEAT variable-topology training loop
│   ├── fitness.py                 # Shared fitness computation (used by both trainers)
│   ├── lichess_bot.py             # Lichess bot (play evolved genomes online)
│   └── godot_wandb.py             # Godot subprocess integration
├── rust/
│   ├── chess-cpu/                 # PyO3: game simulation, NN forward pass, fitness
│   ├── evolve-ga/                 # PyO3: GA operators (selection, crossover, mutation)
│   ├── neat-ga/                   # PyO3: NEAT evolution (speciation, topology mutation)
│   └── chess-native/              # gdext: Godot GDExtension acceleration
├── ai/
│   ├── neural_network.gd          # Feedforward network
│   ├── evolution.gd               # Coevolutionary population manager + Hall of Fame + Elo
│   ├── fitness.gd                 # Multi-factor fitness + endgame evaluation
│   ├── training_manager.gd        # Orchestrates games and evolution
│   ├── neat_evolution.gd          # NEAT topology evolution manager
│   ├── neat_genome.gd             # NEAT genome representation
│   └── neat_network.gd            # NEAT network forward pass
├── chess/
│   ├── constants.gd               # Piece types, values
│   ├── board_state.gd             # Full chess logic (moves, check, castling, en passant)
│   ├── encoder.gd                 # Board -> NN input encoding, output -> move decoding
│   └── pgn.gd                     # PGN export (Standard Algebraic Notation)
├── ui/
│   ├── board_renderer.gd          # Visual chess board with animation
│   ├── human_play.gd              # Human vs AI game mode
│   ├── replay_viewer.gd           # Game replay with PGN export
│   └── training_dashboard.gd      # Stats display and training controls
├── configs/                       # JSON training configs and sweep definitions
├── overnight-agent/
│   ├── chess_sweep_worker.py      # W&B sweep worker
│   ├── chess_monitor.py           # Worker monitor + auto-spawn
│   └── global_elite.py            # Cross-run genome sharing
├── scripts/                       # Lint, test, and utility scripts
├── test/                          # GDScript tests
├── tests/python/                  # Python pytest tests
└── docs/                          # Detailed documentation
```

## Getting Started

### Prerequisites

- **Python 3.10+** with numpy, wandb
- **Rust toolchain** (stable) with maturin (`pip install maturin`)
- **Godot 4.5+** (only needed for UI/Godot training path)

### Build Rust Crates

```bash
# Build all PyO3 crates (required for Rust CPU backend)
cd rust/chess-cpu && maturin develop --release && cd ../..
cd rust/evolve-ga && maturin develop --release && cd ../..
cd rust/neat-ga && maturin develop --release && cd ../..

# Build Godot GDExtension (optional, for Godot training path)
cargo build --release --manifest-path rust/chess-native/Cargo.toml
```

### Run Training

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

### Run Tests

```bash
# Python tests
python -m pytest tests/python -q

# GDScript tests (headless)
godot --headless --path . -s test/test_runner.gd

# All lints + tests
./scripts/lint_and_test.sh
```

### Lichess Bot

Play evolved genomes on Lichess:

```bash
# Test the bot (dry run)
python lichess_bot.py --test

# Accept challenges
python lichess_bot.py --games 5

# Challenge a specific bot
python lichess_bot.py --challenge <bot-username>
```

Requires a Lichess bot account and API token (set `LICHESS_TOKEN` env var). Uses the best NEAT genome from `neat_best_genomes.json`.

### Using the Training Dashboard (Godot UI)

1. Click **Start Training** to begin evolution
2. Monitor live counters: generation, per-color best + average fitness, games played
3. Use the speed selector (1x, 2x, 4x, 8x) for multiple generations per frame
4. Every 5 generations the board viewers show **showcase games** from the best networks

## Chess Logic

Full legal move generation including:
- All piece types with correct movement
- Castling (kingside and queenside, both colors)
- En passant
- Pawn promotion (auto-queen)
- Check, checkmate, and stalemate detection
- 50-move rule draw

## Documentation

Detailed documentation lives in `docs/`:
- [Architecture](docs/ARCHITECTURE.md) — system design and data flow
- [Training](docs/TRAINING.md) — running training, sweep config, metrics
- [Improving Training](docs/IMPROVING_TRAINING.md) — hyperparameter tuning, diagnosing issues
- [AI System](docs/AI_SYSTEM.md) — network architecture, evolution, fitness, Hall of Fame
- [Game System](docs/GAME_SYSTEM.md) — chess rules, board representation, encoder
