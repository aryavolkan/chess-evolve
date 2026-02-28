# Training Guide

> For hyperparameter tuning advice, diagnosing training issues, and performance tips, see [Improving Training](IMPROVING_TRAINING.md).

## Prerequisites

1. **Godot 4.2+** with `--headless` support
2. **Rust toolchain** (stable) — to build the native GDExtension
3. **Python 3.10+** — for the overnight harness
4. **Weights & Biases account** — for sweep tracking (optional)

### Build the Rust extension
```bash
cargo build --release --manifest-path rust/chess-native/Cargo.toml
```

---

## Running a Training Session

### Headless (auto-train mode)
```bash
godot --headless --path . -- --auto-train
```

### With UI
```bash
godot --path .
```
Shows a 2×2 grid of live board viewers and the training dashboard.

### Replay viewer
```bash
godot --path . -- --replay my_game.replay
# Also searches user://replays/my_game.replay and user://my_game.replay
```

---

## sweep_config.json Reference

Write to the Godot user data directory before launching:
- **Linux**: `~/.local/share/godot/app_userdata/Chess Evolve/sweep_config.json`
- **macOS**: `~/Library/Application Support/Godot/app_userdata/Chess Evolve/sweep_config.json`

All keys are optional; defaults shown.

```json
{
  "population_size":        30,     // Genomes per side per generation
  "hidden_size":            64,     // Hidden layer neurons (standard NE)
  "input_size":             389,    // Input vector size (don't change)
  "output_size":            4096,   // 64×64 from-to output space (don't change)
  "elite_count":            2,      // Elites preserved unchanged
  "mutation_rate":          0.25,   // Per-weight mutation probability
  "mutation_strength":      0.12,   // Gaussian std dev for mutations
  "crossover_rate":         0.70,   // Fraction using crossover (vs. clone+mutate)
  "games_per_individual":   2,      // Games each individual plays per gen
  "max_moves_per_game":     100,    // Hard cap on game length
  "max_generations":        50,     // Stop after this many generations
  "use_neat":               false,  // Use NEAT instead of standard NE
  "use_minimax":            false,  // Use minimax search for move selection
  "minimax_depth":          2,      // Minimax search depth
  "use_tournament":         true,   // Tournament evaluation mode
  "tournament_mode":        "round_robin",
  "tournament_opponents":   2,      // Opponents per individual per gen
  "immigration_rate":       0.1,    // Fraction of population replaced with random individuals
  "fitness_sharing_sigma":  0.08,   // RMS genetic distance threshold for fitness sharing
  "tournament_k":           2       // Tournament selection size
}
```

---

## Training Loop

### Generation lifecycle
```
1. Reset fitness arrays
2. For each individual i in white_pop:
   For game j in 1..games_per_individual:
     - Select black opponent (from black_pop or Hall of Fame)
     - Play game (max max_moves_per_game moves):
         encode_board → network.forward → decode_move → make_move
     - ChessFitness.evaluate(result, material, mobility, king_safety)
     - white_fitness[i] += score
     - black_fitness[opponent] += opponent_score

3. ChessEvolution.evolve():
   - Sort by fitness
   - Elite cloning
   - Tournament selection + crossover + mutation
   - Update Hall of Fame
   - Adaptive mutation schedule

4. MetricsLogger.write_metrics(stats)
5. Repeat
```

### Curriculum stages (auto-applied by training manager)

| Stage | Gen range | Max moves | Notes |
|-------|-----------|-----------|-------|
| 0 | 0–2 | 60 | Short games, high win bonus |
| 1 | 3–7 | 100 | Medium games |
| 2 | 8–14 | 130 | Longer games, checkmate bonus enabled |
| 3 | 15+ | 150 | Full evaluation weights |

Curriculum is controlled via `training_manager.use_curriculum = true`.

---

## Metrics Output

Written to `user://metrics.json` (snapshot) and `user://metrics.jsonl` (appended history) after each generation.

### Per-Generation Fields

| Key | Description |
|-----|-------------|
| `generation` | Current generation (0-based) |
| `white_best` | Best white fitness this generation |
| `white_avg` | Mean white fitness |
| `black_best` | Best black fitness |
| `black_avg` | Mean black fitness |
| `best_fitness` | `max(white_best, black_best)` |
| `avg_fitness` | `(white_avg + black_avg) / 2` |
| `games_played` | Total games played so far |
| `total_games_this_gen` | Games played this generation |
| `avg_game_length` | Mean moves per game |
| `games_per_sec` | Training throughput |
| `moves_per_sec` | Move throughput |
| `white_win_rate` | White win fraction this gen |
| `white_draw_rate` | Draw fraction |
| `white_loss_rate` | Loss fraction |
| `black_win_rate/draw_rate/loss_rate` | Same for black |
| `white_hof_size` / `black_hof_size` | Hall of Fame sizes |
| `white_tournament_score_best/avg` | Best/mean tournament scores |
| `black_tournament_score_best/avg` | Same for black |
| `white_material_avg` / `black_material_avg` | Mean material at game end |
| `generation_time_sec` | Wall time for this generation |
| `combined_best` | `min(white_best, black_best)` — balanced optimization metric |
| `white_hof_avg_elo` / `black_hof_avg_elo` | Mean Elo in Hall of Fame |
| `white_hof_top_elo` / `black_hof_top_elo` | Top Elo in Hall of Fame |
| `white_elo_min/p25/median/p75/max` | White Elo distribution |
| `black_elo_min/p25/median/p75/max` | Black Elo distribution |

---

## Overnight / Sweep Training (Python)

### Setup
```bash
pip install wandb
wandb login
```

### Create a sweep
```bash
wandb sweep overnight-agent/sweep_config.yaml --project chess-evolve
```

### Launch workers
```bash
python overnight-agent/chess_sweep_worker.py \
  --sweep-id <sweep-id> \
  --project chess-evolve \
  --count 5 \
  --timeout-minutes 45
```

### Monitor workers
```bash
# Status table
python overnight-agent/chess_monitor.py

# Auto-spawn if idle
python overnight-agent/chess_monitor.py --auto-spawn

# Fill 4 workers and send WhatsApp notification
python overnight-agent/chess_monitor.py --fill --max-workers 4 --notify
```

### Parallel workers (bash)
```bash
SWEEP_ID=<id>
for i in 1 2 3 4; do
  python overnight-agent/chess_sweep_worker.py \
    --sweep-id $SWEEP_ID --count 1 &
done
wait
```

### Key CLI flags for chess_sweep_worker.py

| Flag | Default | Description |
|------|---------|-------------|
| `--sweep-id` | required | W&B sweep ID |
| `--project` | `chess-evolve` | W&B project name |
| `--entity` | None | W&B entity |
| `--count` | 1 | Runs this worker executes |
| `--visible` | false | Show Godot window |
| `--poll-interval` | 2.0 | Seconds between metrics checks |
| `--max-stale` | 300 | Stale polls before abort |
| `--timeout-minutes` | 45.0 | Hard timeout per run |
| `--worker-id` | auto | Fixed worker ID |

---

## Sweep Hyperparameter Ranges

The Bayesian sweep (`sweep_config.py`) maximises `combined_best = min(white_best, black_best)` for balanced improvement.

| Parameter | Type | Range / Values |
|-----------|------|----------------|
| `population_size` | values | 30, 50 |
| `hidden_size` | values | 32, 64 |
| `elite_count` | fixed | 2 |
| `mutation_rate` | uniform | [0.15, 0.40] |
| `mutation_strength` | uniform | [0.05, 0.20] |
| `crossover_rate` | uniform | [0.60, 0.85] |
| `games_per_individual` | values | 2, 3 |
| `max_moves_per_game` | fixed | 100 |
| `max_generations` | fixed | 10 |
| `tournament_opponents` | fixed | 5 |
| `immigration_rate` | uniform | [0.05, 0.20] |
| `fitness_sharing_sigma` | uniform | [0.03, 0.15] |
| `tournament_k` | fixed | 2 |

See `sweep_config.py` for alternative profiles (`sweep_config_quick`, `sweep_config_deep`).
For guidance on tuning these parameters, see [Improving Training](IMPROVING_TRAINING.md).

---

## CI Checks

Every PR runs:

| Job | What it checks |
|-----|----------------|
| `lint` | `ruff` (Python), `gdlint` (GDScript), `cargo fmt`, `cargo clippy -D warnings` |
| `python-tests` | `pytest tests/python/` |
| `godot-tests` | Headless GDUnit4 runner |

Run locally:
```bash
ruff check scripts/ train_wandb.py sweep_config.py
gdlint ai/ chess/
cargo fmt --check --manifest-path rust/chess-native/Cargo.toml
cargo clippy --manifest-path rust/chess-native/Cargo.toml -- -D warnings
```
