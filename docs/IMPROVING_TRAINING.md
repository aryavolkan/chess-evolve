# Improving Training

A practical guide for getting better chess play out of chess-evolve's neuroevolution pipeline. Covers hyperparameter tuning, diagnosing common failure modes, leveraging performance backends, and architectural directions for stronger play.

---

## Quick Wins

Before diving into sweep optimization, these changes have the highest impact-to-effort ratio:

1. **Use the Rust backend** — 3-6x faster than Godot, enabling larger populations and more games per generation. Build with `cargo build --release --manifest-path rust/chess-native/Cargo.toml`.
2. **Enable tournament mode** — `use_tournament: true` with `tournament_opponents: 5` produces far more stable fitness rankings than random pairings.
3. **Keep elite_count at 2** — sweep data consistently shows `elite_count=2` outperforms 3 or 5. Higher elitism stifles exploration.
4. **Use `tournament_k: 2`** — lower selection pressure lets more diverse strategies survive, preventing premature convergence.
5. **Enable fitness sharing** — `fitness_sharing_sigma: 0.08` rewards genetic diversity and prevents the population from collapsing to one strategy.

---

## Hyperparameter Guide

### Parameters That Matter Most

Based on Bayesian sweep analysis (sweep `uq8yh9ck` and subsequent runs), these parameters have the strongest effect on `combined_best` (= `min(white_best, black_best)`):

| Parameter | Recommended | Range to Sweep | Why |
|-----------|------------|----------------|-----|
| `population_size` | 30-50 | {30, 50} | Larger pops find better solutions but cost linearly more. 30 is the sweet spot for <10 gen runs. |
| `elite_count` | 2 | {2, 3} | 2 strongly favors breakout runs. Higher values reduce exploration. |
| `mutation_rate` | 0.20-0.30 | [0.15, 0.40] | Per-weight probability. Too low → stagnation; too high → destructive. |
| `mutation_strength` | 0.10-0.15 | [0.05, 0.20] | Gaussian σ. Pairs with rate: high rate + low strength is often best. |
| `crossover_rate` | 0.70 | [0.60, 0.85] | Fraction that use two-point crossover vs clone+mutate. |
| `immigration_rate` | 0.10-0.15 | [0.05, 0.20] | Random individuals injected per generation. Maintains diversity. |

### Parameters with Moderate Impact

| Parameter | Recommended | Notes |
|-----------|------------|-------|
| `hidden_size` | 32 or 64 | 128 consistently underperforms (too many parameters for the signal). 32 trains faster; 64 has slightly higher ceiling. |
| `games_per_individual` | 2-3 | More games = less fitness variance, but costs linearly. 2 is usually sufficient with tournament mode. |
| `tournament_opponents` | 5 | Number of opponents per individual per generation. 4-6 range is fine. |
| `max_moves_per_game` | 100 | Higher values let endgames play out but slow training. Curriculum auto-adjusts this. |
| `fitness_sharing_sigma` | 0.05-0.10 | RMS genetic distance threshold. Lower = more pressure for diversity. |

### Parameters to Leave Alone

| Parameter | Default | Why |
|-----------|---------|-----|
| `input_size` | 389 | Determined by the encoder; changing requires rewriting `ChessEncoder`. |
| `output_size` | 4096 | Full 64×64 from-to output space. Changing requires rewriting the decoder. |
| `tournament_k` | 2 | Selection pressure. k=3 converges faster but to worse optima. |
| `tournament_mode` | `"round_robin"` | Swiss is slightly faster for large pops but round-robin is more reliable. |

---

## Diagnosing Training Problems

### Fitness Stalls (No Improvement for 5+ Generations)

**Symptoms:** `white_best` and `black_best` plateau; `combined_best` flatlines.

**Causes and fixes:**
- **Population collapsed to one strategy** → increase `immigration_rate` to 0.15-0.20, enable `fitness_sharing_sigma`
- **Mutation too weak** → the adaptive mutation system should handle this automatically, but check that `adaptive_mutation` is enabled in `evolution.gd`
- **Not enough games** → increase `games_per_individual` to 3 or `tournament_opponents` to 6 for better signal
- **Too few generations** — neuroevolution is slow. 10 generations is a quick check; 25-50 for meaningful progress

### Arms Race Cycling

**Symptoms:** `white_best` and `black_best` oscillate — white improves, then black catches up and white regresses.

**Causes and fixes:**
- **No Hall of Fame** → ensure `hall_of_fame_ratio: 0.5` so individuals play against historical opponents
- **Hall of Fame too small** → increase beyond the default 20 if populations are large
- **Symmetric weakness** → monitor `white_win_rate` vs `black_win_rate`; persistent asymmetry suggests one side found a gimmick the other exploits

### One Side Dominates

**Symptoms:** `white_best` >> `black_best` (or vice versa) persistently.

**Causes and fixes:**
- This is normal in early training — white has first-move advantage
- If severe and persistent, check that both populations evolve independently (`ChessEvolution` maintains separate `white_pop` and `black_pop`)
- The `combined_best = min(white_best, black_best)` metric is specifically designed to penalize lopsided progress

### Slow Training Throughput

**Symptoms:** `games_per_sec` < 50 (Godot) or `generation_time_sec` > 60s.

**Causes and fixes:**
- **Use the Rust backend** — biggest single speedup
- **Reduce `max_moves_per_game`** — 40-60 for early exploration, 100+ only when networks are strong
- **Disable minimax** — `use_minimax: false` (minimax is 20-50x slower per move)
- **Reduce population** — halving population halves generation time
- **Disable opening book** — `use_opening_book: false` removes the lookup overhead (minor)

---

## Sweep Best Practices

### Running a Sweep

```bash
# Quick validation sweep (grid, ~16 runs)
python -c "
from sweep_config import sweep_config_quick
import wandb, json
sweep_id = wandb.sweep(sweep_config_quick, project='chess-evolve')
print(f'Sweep ID: {sweep_id}')
"

# Launch 4 parallel workers
for i in 1 2 3 4; do
  python train_wandb.py --sweep <sweep-id> &
done
wait
```

### Three Sweep Profiles

| Profile | Method | Generations | Use Case |
|---------|--------|------------|----------|
| `sweep_config_quick` | Grid | 10 | Validate changes quickly. ~16 runs. |
| `sweep_config` | Bayes | 10 | Main optimization. Bayesian picks best params. |
| `sweep_config_deep` | Bayes | 25 | Find optimal architecture. Expensive but thorough. |

### Interpreting Sweep Results

The primary optimization metric is **`combined_best`** = `min(white_best, black_best)`. This ensures balanced improvement — a run where one side gets very strong but the other is weak scores poorly.

**Key W&B charts to watch:**
- `combined_best` vs generation — should trend up
- `white_best` and `black_best` — should track roughly together
- `white_win_rate` — should hover near 0.45-0.55 (balanced)
- `avg_game_length` — should increase over generations (longer games = better defense)
- `games_per_sec` — throughput; higher is better for iteration speed

### Cross-Run Knowledge Transfer

The **Global Elite Pool** (`global_elite.py`) shares the best genomes across parallel sweep workers. After each run, a worker contributes its best networks; before the next run, it seeds its Hall of Fame from the shared pool.

This means later sweep runs benefit from earlier discoveries. Watch `global_elite/pool_size` in W&B to track pool growth.

---

## Curriculum Design

The training manager applies different fitness weights at different stages. The default curriculum (enabled via `use_curriculum: true`):

| Stage | Generations | Max Moves | Strategy |
|-------|------------|-----------|----------|
| Early | 0-2 | 60 | Short games, high win bonus (6.0), strong material weight (1.2). Teaches piece value. |
| Mid | 3-7 | 100 | Medium games, higher draw bonus (2.0). Teaches defense. |
| Late | 8-14 | 130 | Full-length games, checkmate bonus enabled (5.0). Teaches endgame. |
| Full | 15+ | 150 | Final weights with full king safety (0.5). Tests complete play. |

### Modifying the Curriculum

Edit `ai/training_manager.gd` in the `_apply_curriculum()` method. Each stage sets:
- `max_moves_per_game` — game length cap
- Fitness weights via `ChessFitness.set_weights({...})`

**Tips for custom curricula:**
- Start with very short games (30-40 moves) if populations are completely random — it reduces noise
- Introduce checkmate bonus only after networks can reliably capture material
- King safety weight should increase gradually — early networks don't understand king exposure
- Consider adding a stage between "Late" and "Full" if your run length is 50+ generations

---

## Performance Backends

Training can run on four backends, auto-detected in this priority order:

| Backend | Speed | Requirements |
|---------|-------|-------------|
| **Rust CPU** | Fastest | `chess_cpu` + `evolve_ga` PyO3 crates built |
| **PyTorch GPU** | Fast | `torch` with CUDA, `python-chess`, `gpu_trainer.py` |
| **PyTorch CPU** | Medium | `torch` (CPU-only), `python-chess`, `gpu_trainer.py` |
| **Godot** | Slowest | Just `godot` binary (always available) |

### Rust Backend Setup

```bash
# Build the GDExtension (for Godot integration)
cargo build --release --manifest-path rust/chess-native/Cargo.toml

# Build the PyO3 bindings (for Python-driven Rust training)
cd rust/evolve-ga && maturin develop --release
```

The Rust path accelerates:
- Move generation (magic bitboards, 100-1000x over GDScript)
- Neural network forward pass (SIMD vectorization)
- Genetic algorithm operators (selection, mutation, crossover)

### Backend-Specific Tips

- **Godot backend**: Keep `max_moves_per_game` ≤ 60 and `population_size` ≤ 30. GDScript move generation is the bottleneck.
- **Rust backend**: Can comfortably run `population_size: 100`, `max_moves_per_game: 150`, `tournament_opponents: 6`.
- **PyTorch GPU**: Best for experiments with larger hidden sizes (64-96) where matrix multiplication dominates.

---

## Fitness Function Tuning

The fitness function (`ai/fitness.gd`) combines multiple signals:

```
fitness = win_bonus × win + checkmate_bonus × checkmate + draw_bonus × draw
        + material_weight × (my_material - opp_material)
        + mobility_weight × mobility_score
        + king_safety_weight × king_safety_score
        + move_weight × moves_played
```

All components are clamped so `fitness ≥ 0`.

### Tuning Tips

- **Increase `win_bonus`** (default 10.0) if networks aren't learning to win games
- **Increase `material_weight`** (default 1.0) if networks sacrifice pieces recklessly
- **Decrease `mobility_weight`** (default 0.05) — it's a weak signal that can mislead early networks
- **Increase `king_safety_weight`** (default 0.5) only after networks understand basic tactics
- **Set `draw_bonus` to 0** temporarily if you want more aggressive play (networks will stop playing for draws)
- **Endgame bonus** (from `endgame_hints.gd`) rewards K+Q/R vs K positions — leave this enabled to help networks learn basic checkmates

---

## Architecture Decisions and Future Directions

### Current Architecture: 389 → 64 → 4096

The current standard NE network uses a single hidden layer (389 inputs, 64 hidden neurons with tanh, 4096 outputs with tanh). This is ~33K parameters.

**Why 4096 outputs?** Each output corresponds to one (from_square, to_square) pair. The previous 128-output factored scheme (64 from-logits + 64 to-logits, score = sum) was simpler but couldn't express move-specific preferences — e.g., it couldn't prefer Nf3 over Ng1 if both move from g1 and f3 respectively.

**Why single hidden layer?** Neuroevolution struggles with deep networks because gradient-free optimization has difficulty coordinating weights across many layers. A single hidden layer is the practical ceiling for populations of 30-50 with mutation-based learning.

### NEAT Topology Evolution

When `use_neat: true`, the system uses `ChessNeatEvolution` which starts with minimal networks (no hidden nodes) and grows complexity through mutation:
- **Node mutation**: splits an existing connection to insert a hidden node
- **Connection mutation**: adds a random new connection
- **Speciation**: protects novel topologies from premature competition

NEAT is theoretically superior for discovering the right architecture, but in practice requires larger populations (100+) and more generations (50+) to outperform a well-tuned fixed-topology GA. Use NEAT for long overnight runs, standard NE for quick experiments.

### Planned Improvements

These are partially implemented or planned upgrades that would improve training quality:

1. **Full bitboard migration** — `BitboardState` exists but the training pipeline still defaults to `BoardState`. Switching would give 5-15x speedup in GDScript.
2. **773-input encoder** — use 12 bitboard planes (768 bits) + 5 metadata instead of the current 389-element signed-plane encoding. More information but requires network retraining from scratch.
3. **NEAT output size reduction** — use 218 outputs (max legal moves in any position) with legal-move-index masking instead of 4096. Reduces NEAT search space dramatically.
4. **GDExtension for NEAT forward pass** — the sparse matrix-vector multiply in NEAT is the bottleneck once move generation is fast. Porting to Rust would unlock population sizes of 500+.
5. **Multi-hidden-layer support** — for Rust backend where forward pass cost is negligible, 2-3 hidden layers could capture more complex patterns.

### What Won't Help

- **Larger hidden sizes (128+)**: Sweep data shows 128 hidden neurons consistently underperform 32 or 64. The network has more capacity than the fitness signal can train.
- **Higher crossover rates (>0.85)**: Excessive crossover disrupts learned weight patterns. 0.70 is near-optimal.
- **Minimax search during training**: Makes moves much better but is 20-50x slower per move. The generation-level throughput loss outweighs the per-game quality gain. Reserve minimax for evaluation/human play only.
- **Very large populations (200+) in Godot**: GDScript can't evaluate fast enough. Only viable with Rust backend.

---

## Monitoring and Debugging

### Key Metrics to Watch

| Metric | Healthy Range | Warning Sign |
|--------|--------------|-------------|
| `combined_best` | Trending up | Flat for 5+ gens |
| `white_win_rate` | 0.40-0.60 | Persistently >0.70 or <0.30 |
| `avg_game_length` | Increasing over time | Stuck at `max_moves_per_game` (all draws) |
| `games_per_sec` | >50 (Godot), >500 (Rust) | <20 suggests a bottleneck |
| `white_hof_size` / `black_hof_size` | Growing to 15-20 | Stuck at 0-2 |
| `generation_time_sec` | <30s (Godot), <5s (Rust) | >60s per gen |

### Debugging with Replays

```bash
# Watch the best network play
godot --path . -- --replay user://replays/best_game.replay

# Play against the best network yourself
godot --path . -- --human-play
```

Replay files are saved per generation to `user://replays/`. Watching evolved networks play reveals:
- Whether they understand piece value (do they capture free pieces?)
- Whether they understand king safety (do they castle? do they attack the king?)
- Whether games end in real checkmates or just move-limit draws

### W&B Dashboard Setup

Create a W&B dashboard with these panels for effective monitoring:
1. **Line chart**: `combined_best`, `white_best`, `black_best` vs `generation`
2. **Line chart**: `white_win_rate`, `white_draw_rate` vs `generation`
3. **Line chart**: `avg_game_length` vs `generation`
4. **Scatter**: `games_per_sec` vs `population_size` (to find throughput sweet spot)
5. **Parallel coordinates**: all sweep parameters colored by `combined_best`
