# Chess Evolve 🧬♟️

Evolutionary neural networks learn to play chess through coevolution. A sister project to [Evolve](../evolve), adapted for chess.

## Core Concept

Two populations of neural networks — one playing white, one playing black — evolve against each other. Networks that win more games survive and reproduce. Over generations, both populations develop increasingly sophisticated chess strategies.

## Architecture

```
┌─────────────────────────────────────────────┐
│                 Training Manager             │
│  ┌──────────┐  ┌──────────┐                 │
│  │ White Pop │  │ Black Pop │  (coevolution) │
│  │ (50 nets) │  │ (50 nets) │                │
│  └─────┬─────┘  └─────┬─────┘               │
│        │               │                     │
│        └───┬───────┬───┘                     │
│            ▼       ▼                         │
│         Game Engine (BoardState)             │
│            │       │                         │
│         Encoder ──► Neural Network           │
│         (389 in)   (64 hidden → 4096 out)    │
│            │                                 │
│         Fitness Evaluation                   │
│  (material, mobility, king safety, result)   │
└─────────────────────────────────────────────┘
```

### Neural Network

- **Input (389):** 6 piece-type planes × 64 squares (signed ±1), side to move, castling rights
- **Hidden:** 64 neurons, tanh activation
- **Output (4096):** one logit per (from_square × 64 + to_square) pair
- **Move selection:** score = output[from×64 + to], masked to legal moves, pick highest
- **~33K trainable parameters**

### Fitness Function

| Component | Weight | Description |
|-----------|--------|-------------|
| Win | 10.0 | Bonus for winning the game |
| Checkmate | 5.0 | Extra bonus for checkmate |
| Draw | 3.0 | Modest reward for draws |
| Material | 1.0× | Net material advantage |
| Mobility | 0.05× | Legal move count |
| King Safety | 0.5× | Friendly pawns near king |
| Game Length | 0.01× | Reward for surviving longer |

### Evolution

- Tournament selection (k=3)
- Two-point crossover (70% rate)
- Gaussian mutation (rate=0.15, σ=0.2)
- Elitism (top 5 preserved)
- Independent evolution for white and black populations

## Project Structure

```
chess-evolve/
├── ai/
│   ├── neural_network.gd    # Feedforward network
│   ├── evolution.gd          # Coevolutionary population manager + Hall of Fame + Elo
│   ├── fitness.gd            # Multi-factor fitness + endgame evaluation
│   ├── minimax_player.gd     # Minimax search with NN evaluation
│   ├── training_manager.gd   # Orchestrates games and evolution
│   ├── neat_evolution.gd     # NEAT topology evolution manager
│   ├── neat_genome.gd        # NEAT genome representation
│   └── neat_network.gd       # NEAT network forward pass
├── chess/
│   ├── constants.gd          # Piece types, values
│   ├── board_state.gd        # Full chess logic (moves, check, castling, en passant)
│   ├── encoder.gd            # Board → NN input encoding, output → move decoding
│   ├── endgame_hints.gd      # Endgame pattern recognition and evaluation
│   ├── opening_book.gd       # Embedded opening book (~30 openings)
│   └── pgn.gd                # PGN export (Standard Algebraic Notation)
├── ui/
│   ├── board_renderer.gd     # Visual chess board with animation and click handling
│   ├── human_play.gd         # Human vs AI game mode
│   ├── replay_viewer.gd      # Game replay with PGN export
│   └── training_dashboard.gd # Stats display and training controls
├── scenes/
│   ├── main.gd               # Main scene controller
│   └── main.tscn             # Entry scene
├── scripts/
│   └── wandb_bridge.py       # Stream metrics to Weights & Biases
├── test/
│   ├── test_base.gd          # Test framework
│   ├── test_runner.gd        # Headless test runner
│   ├── test_board_state.gd   # Chess logic tests
│   ├── test_encoder.gd       # Encoding tests
│   ├── test_neural_network.gd # NN tests
│   ├── test_evolution.gd     # Evolution tests
│   ├── test_fitness.gd       # Fitness tests
│   ├── test_training.gd      # Integration tests
│   ├── test_pgn.gd           # PGN export tests
│   ├── test_opening_book.gd  # Opening book tests
│   └── test_endgame.gd       # Endgame hints tests
└── README.md
```

## Getting Started

### Prerequisites
- Godot 4.5+ (same as Evolve)

### Run Tests
```bash
cd ~/Projects/chess-evolve
godot --headless --script test/test_runner.gd
```

### Run the Project
Open in Godot Editor or:
```bash
godot --path ~/Projects/chess-evolve
```

### Using the Training Dashboard

1. Click **Start Training** to begin evolution (button flips to **Pause/Resume**)
2. Monitor the live counters: generation, per-color best + average fitness, and cumulative games played
3. Use the speed selector (1×, 2×, 4×, 8×) to run multiple generations per frame — it also adjusts Godot's `time_scale`
4. Every 5 generations the 4 board viewers show **showcase games** from the best networks
5. Training runs continuously — press Pause to halt without resetting stats

### Metrics + W&B Logging

- Each generation writes `metrics.json` under `~/Library/Application Support/Godot/app_userdata/Chess Evolve/metrics.json`.
- Fields tracked: generation, white/black best + average fitness, combined aggregates, population size, total games played, games per generation, and `updated_at`.
- Use `scripts/wandb_bridge.py` to stream the JSON into Weights & Biases while training is running.

Example snippet:
```json
{
  "generation": 12,
  "white_best": 15.3,
  "black_best": 14.8,
  "white_avg": 3.2,
  "black_avg": 2.9,
  "best_fitness": 15.3,
  "avg_fitness": 3.05,
  "population_size": 30,
  "games_played": 720,
  "games_per_generation": 60,
  "updated_at": 1739498123
}
```

Run the bridge alongside Godot:
```bash
pip install --upgrade wandb
python scripts/wandb_bridge.py --project chess-evolve --run-name dev-test
```

Hit `Ctrl+C` to stop streaming or point `--metrics-path` to logs synced from another machine.

## Chess Logic

Full legal move generation including:
- All piece types with correct movement
- Castling (kingside and queenside, both colors)
- En passant
- Pawn promotion (auto-queen)
- Check, checkmate, and stalemate detection
- 50-move rule draw

## Phase 2 Features

All Phase 2 items are complete:

- [x] **NEAT topology evolution** — `neat_evolution.gd`, `neat_genome.gd`, `neat_network.gd` enable evolving network topology alongside weights
- [x] **Hall of Fame** — Top 20 networks per color archived with Elo ratings, used as training opponents with weighted selection
- [x] **Opening book** — ~30 embedded openings (Italian, Sicilian, Caro-Kann, Queen's Gambit, etc.) for training diversity; configurable depth
- [x] **Endgame hints** — Pattern recognition for K vs K, K+Q/R vs K, insufficient material; driving heuristic integrated into fitness
- [x] **W&B Elo tracking** — Elo distribution metrics (min/p25/median/p75/max) per color streamed to Weights & Biases
- [x] **Human play mode** — Play against evolved networks via `--human-play` CLI flag or dashboard button; supports minimax opponent
- [x] **Move animation** — Tween-based piece animation with capture fade-out on board viewer and replay viewer
- [x] **PGN export** — Full Standard Algebraic Notation with disambiguation, castling, check/checkmate; export button in replay viewer

### Human Play Mode

```bash
godot --path . -- --human-play
```

Click a piece to see legal moves highlighted, then click a destination to move. The AI responds automatically.

## Relationship to Evolve

Borrows patterns from Evolve:
- `neural_network.gd` — same feedforward architecture, adapted sizes
- `evolution.gd` — tournament selection, crossover, mutation (extended for coevolution + Hall of Fame)
- `training_manager.gd` — orchestration pattern (extended with curriculum, tournament mode)
- `test/` — same test framework and conventions

## Documentation

Detailed documentation lives in `docs/`:
- [Architecture](docs/ARCHITECTURE.md) — system design and data flow
- [Training](docs/TRAINING.md) — running training, sweep config, metrics
- [Improving Training](docs/IMPROVING_TRAINING.md) — hyperparameter tuning, diagnosing issues, performance
- [AI System](docs/AI_SYSTEM.md) — network architecture, evolution, fitness, Hall of Fame
- [Game System](docs/GAME_SYSTEM.md) — chess rules, board representation, encoder
