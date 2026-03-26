# Steady Progress Pipeline — Design Spec

**Goal**: Build a reliable NEAT training pipeline where every overnight run produces measurable Elo improvement, targeting 1200+ Elo from a current baseline of 400-600.

**Scope**: NEAT path only. Fixed-topology GA is out of scope.

**Success criteria**: Three consecutive overnight runs (8-12 hours each, 2-4 workers) each show ≥30 Elo improvement on the Stockfish benchmark, with no regressions.

---

## 1. Five-Stage Curriculum

Replace the current 3-stage system (puzzles → random → coevolution) with 5 stages that bridge the puzzle-to-play transfer gap.

### Stage definitions

| Stage | Name | Description | Entry criterion | Exit criterion |
|-------|------|-------------|-----------------|----------------|
| 0 | Puzzles | Solve tactical puzzles, adaptive difficulty 600→2000 rating | Start of run (or seeded stage - 1) | Benchmark puzzle accuracy ≥ 0.60 AND puzzle rating ≥ 1400 |
| 1 | Guided Play | Full games vs weak opponents: random legal moves with 30% chance of piece-value heuristic move | Exit stage 0 | bench_win_rate ≥ 0.70 for 3 consecutive generations |
| 2 | Opponent Ladder | Full games vs saved HoF genomes from previous runs. Start with weakest, advance to strongest as win rate improves | Exit stage 1 | bench_win_rate ≥ 0.85 AND sf_avg_cpl ≤ 800 |
| 3 | Stockfish Shaping | Coevolution with CPL fitness blending. sf_fitness_weight ramps linearly from 0.2 to 0.5 over 20 generations | Exit stage 2 | sf_avg_cpl ≤ 400 for 5 consecutive generations |
| 4 | Coevolution Refinement | Full coevolution with HoF anti-cycling. CPL signal reduced to 0.1 (validation only). Near-deterministic play | Exit stage 3 | No exit — runs until compute budget exhausted |

### Transition blending

When advancing from stage N to N+1, spend 3 generations blending fitness:

```
blended_fitness = 0.7 * stage_N+1_fitness + 0.3 * stage_N_fitness
```

This prevents cliff-edge difficulty spikes at stage transitions.

### Stage 1 — Guided Play opponent detail

The "weak heuristic" opponent selects moves as follows:
- 70% of moves: uniform random from legal moves (current random behavior)
- 30% of moves: pick the legal move that maximizes material value of captured piece (if any capture available), otherwise random

This gives networks something slightly better than pure random to learn against, without being so strong that early networks can't score at all.

### Stage 2 — Opponent Ladder detail

- Load all genomes from `neat_best_genomes.json` (cross-run archive)
- Sort by their recorded benchmark win rate (weakest first)
- Divide into 3 tiers: bottom third, middle third, top third
- Start playing against bottom tier. When win rate vs current tier ≥ 0.60 for 2 gens, advance to next tier
- When all 3 tiers cleared, check exit criteria

---

## 2. Temperature Annealing

### Training temperature schedule

| Stage | Temperature | Notes |
|-------|-------------|-------|
| 0 (Puzzles) | 0.30 | Fixed — some exploration for discovering tactics |
| 1 (Guided Play) | 0.30 → 0.15 | Linear anneal over stage duration |
| 2 (Opponent Ladder) | 0.15 → 0.05 | Linear anneal over stage duration |
| 3 (SF Shaping) | 0.05 | Fixed — CPL signal requires clean moves |
| 4 (Coevo Refinement) | 0.05 | Fixed — exploit learned play |

Linear anneal formula within a stage:
```
T = T_start - (T_start - T_end) * (gen_in_stage / expected_stage_length)
```

Where `expected_stage_length` is estimated from previous runs (default: 15 gens per stage).

### Evaluation temperature

**All benchmark and Stockfish evaluations use temperature 0.0 (argmax).** This is a new config key `eval_temperature` (default 0.0), separate from `move_temperature`.

Current benchmarks use the training temperature, injecting noise into progress measurements. Fixing this alone will give cleaner W&B curves.

---

## 3. Stage-Specific Fitness Weights

Each stage emphasizes different aspects of play:

| Component | Stage 0 | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|-----------|---------|---------|---------|---------|---------|
| win_bonus | 10.0 | 10.0 | 10.0 | 10.0 | 10.0 |
| checkmate_bonus | 15.0 | 10.0 | 10.0 | 10.0 | 10.0 |
| draw_bonus | 0.0 | 3.0 | 3.0 | 3.0 | 3.0 |
| loss_penalty | -5.0 | -5.0 | -5.0 | -5.0 | -5.0 |
| material_weight | 0.5 | 1.5 | 1.0 | 0.5 | 1.0 |
| mobility_weight | 0.0 | 0.5 | 0.3 | 0.1 | 0.3 |
| king_danger_weight | 0.0 | 0.5 | 1.0 | 0.5 | 1.0 |
| capture_weight | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 |
| move_count_penalty | -0.002 | -0.002 | -0.002 | -0.002 | -0.002 |
| sf_fitness_weight | 0.0 | 0.0 | 0.0 | 0.2→0.5 | 0.1 |

### Rationale

- **Stage 0**: High checkmate bonus drives tactical puzzle solving. No mobility/king signals — puzzles don't reward positional play.
- **Stage 1**: High material weight teaches "don't blunder pieces" during first real games. Mobility introduced to encourage active play.
- **Stage 2**: Balanced weights. King danger rises — networks need attacking ideas against real opponents.
- **Stage 3**: Material weight drops as SF CPL provides a stronger per-move signal. Networks learn from Stockfish what "good moves" look like.
- **Stage 4**: Balanced defaults. SF signal drops to validation-only (0.1) so coevolution drives the arms race.

---

## 4. Benchmark & Progress Tracking

### Expanded benchmark population

- **Random benchmark**: Increase from 20 to 50 fixed random genomes. Seeded once and saved to `benchmark_random_50.json`. All runs use the same file for comparability.
- **Historical benchmark**: 10 genomes frozen from the best of previous runs. Saved to `benchmark_historical.json`. Updated at the end of each run by appending the run's best genome (FIFO, keep newest 10).

### Elo estimate metric

Add `elo_estimate` to W&B logging, computed from Stockfish CPL:

```python
elo_estimate = max(0, 2000 - sf_avg_cpl)
```

This is a rough linear mapping but provides a single trackable number across runs. Logged every generation where SF benchmark runs.

### Per-run progress report

At the end of each run, log a summary to W&B and print to stdout:

```
Run summary:
  Start stage: 1 (seeded from previous run)
  End stage: 3
  Start elo_estimate: 520
  End elo_estimate: 680
  Delta: +160
  Puzzle accuracy: 0.72
  SF avg CPL: 1320
  Benchmark win rate: 0.82
  Generations: 87
  Wall time: 9.2 hours
```

---

## 5. Cross-Run Seeding

### Genome seeding

At the start of each run:

1. Load `neat_best_genomes.json` from previous run
2. Take the single best genome (by benchmark win rate)
3. Create 5 mutated variants (standard NEAT mutation)
4. Inject all 6 into the starting population (replacing 6 random initializations)

### Stage seeding

Persist the highest stage reached to `run_state.json`:

```json
{
  "highest_stage": 2,
  "best_elo_estimate": 680,
  "best_genome_id": "w_gen87_0",
  "puzzle_max_rating": 1600,
  "timestamp": "2026-03-25T12:00:00Z"
}
```

Next run starts at `max(0, highest_stage - 1)` — one step back for robustness, since the seeded population may not immediately perform at its previous level in a new evolutionary context.

### HoF archive continuity

The Hall of Fame is already merged across runs. No change needed — just ensure `neat_best_genomes.json` is read at startup and merged into the initial HoF.

---

## 6. Implementation Scope

### Files to modify

| File | Changes |
|------|---------|
| `python/neat_cpu_trainer.py` | 5-stage curriculum logic, stage transitions, transition blending, temperature annealing, stage-specific fitness weights |
| `python/fitness.py` | Accept stage parameter for weight lookup, add `get_stage_weights(stage)` function |
| `train_wandb.py` | New config keys (`eval_temperature`, `stage_fitness_weights`, `benchmark_size`, `seed_from_previous`), cross-run seeding logic, run summary logging, `run_state.json` persistence |
| `rust/chess-cpu` | Support `eval_temperature` parameter in simulation functions (separate from training temperature). Add "guided play" opponent mode for Stage 1 (30% heuristic move selection based on capture value) |
| `configs/` | New `steady_progress_config.json` with tuned defaults for this pipeline |

### Files to create

| File | Purpose |
|------|---------|
| `python/curriculum.py` | Stage definitions, transition logic, temperature schedule, fitness weight tables — extracted from trainer to keep it clean |
| `benchmark_random_50.json` | Fixed 50-genome random benchmark (generated once) |
| `benchmark_historical.json` | Rolling 10-genome archive of previous bests |
| `run_state.json` | Cross-run state persistence |

### Not in scope

- Fixed-topology GA changes
- Multi-hidden-layer networks
- Legal-move-only output masking
- Minimax search integration
- Lichess bot improvements (will benefit automatically from better genomes)

---

## 7. Validation Plan

### Unit tests

- `test_curriculum.py`: Stage advancement logic, transition blending, temperature schedule, fitness weight lookup
- `test_cross_run_seeding.py`: Genome loading, mutation, stage persistence, HoF merge

### Integration test

- Run a 10-generation micro-training with `population_size=10` and verify:
  - Stage 0 → Stage 1 transition fires when puzzle accuracy threshold met
  - Temperature decreases within a stage
  - Fitness weights change at stage boundaries
  - Benchmark uses eval_temperature=0.0
  - `run_state.json` written at end

### Acceptance criteria

- Three consecutive overnight runs show monotonic `elo_estimate` improvement
- No run regresses more than 20 Elo from its starting point
- Stage transitions are logged clearly in W&B with markers
