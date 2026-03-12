# Curriculum Learning Plan for NEAT Chess Evolution

## Problem Statement

NEAT cannot discover useful chess heuristics through random weight/topology mutations against coevolutionary opponents. After 2000 generations with populations of 200-500, bench win rate plateaus at ~7% (barely above random). The fitness landscape is flat — most mutations don't affect game outcomes because:

1. Coevolution masks absolute progress (both sides improve → fitness stays flat)
2. Temperature-based move selection dilutes the effect of weight changes
3. 5 games per individual creates noisy fitness estimates
4. The task "win at chess" is too hard as a first objective

## Proposed Solution: Staged Curriculum

Train against progressively harder opponents, only advancing when the current stage is mastered. Each stage provides a clear, achievable gradient.

### Stage 1: Beat Random (target: >80% win rate)

**Opponent**: Fixed random policy (uniform random legal moves)
**Why first**: Random is the weakest possible opponent. If NEAT can't beat random, it can't do anything. This stage teaches basic material capture — taking hanging pieces is enough to win.

**Config**:
```python
{
    "curriculum_stage": 1,
    "opponent_type": "random",
    "promotion_threshold": 0.80,  # advance when bench_wr > 80%
    "temperature": 0.1,           # near-deterministic to amplify weight signal
    "games_per_individual": 20,   # reduce fitness noise
    "population_size": 200,
    "output_size": 384,
}
```

**What NEAT should learn**: Capture undefended pieces. Move toward opponent pieces. Basic material awareness.

**Fitness function**: Pure benchmark fitness (no coevolution). Win=+20, draw=+3, loss=-5, material difference.

### Stage 2: Beat Weak Self (target: >70% win rate vs stage 1 graduates)

**Opponent**: Hall of Fame from Stage 1 (frozen, not coevolving)
**Why**: Stage 1 graduates capture hanging pieces but have no defense. Stage 2 teaches "don't leave pieces hanging" against opponents that will take them.

**Config**:
```python
{
    "curriculum_stage": 2,
    "opponent_type": "frozen_hof",
    "opponent_hof_path": "stage1_hof.json",
    "promotion_threshold": 0.70,
    "temperature": 0.15,
    "games_per_individual": 15,
}
```

**What NEAT should learn**: Defend pieces. Avoid trading down. Basic positional awareness.

### Stage 3: Coevolution with CPL Pressure (target: CPL < 400)

**Opponent**: Coevolutionary (white vs black populations)
**Why**: Now that both sides can capture and defend, coevolution becomes meaningful. Add Stockfish CPL fitness to push toward good moves.

**Config**:
```python
{
    "curriculum_stage": 3,
    "opponent_type": "coevolution",
    "sf_fitness_weight": 0.5,
    "sf_fitness_interval": 1,
    "benchmark_fitness_weight": 0.3,  # still blend absolute fitness
    "temperature": 0.2,
    "promotion_threshold_cpl": 400,   # advance when avg CPL < 400
}
```

**What NEAT should learn**: Positional play. Avoid blunders. Make moves that Stockfish considers reasonable.

### Stage 4: Beat Stockfish Skill 0 (target: >30% win rate)

**Opponent**: Mix of coevolution + Stockfish skill 0 games
**Why**: Stockfish skill 0 plays real chess with intentional mistakes. Beating it means the network plays actual chess, not just "better random."

**Config**:
```python
{
    "curriculum_stage": 4,
    "opponent_type": "coevolution+stockfish",
    "sf_opponent_fraction": 0.3,  # 30% of games vs Stockfish
    "sf_skill_level": 0,
    "promotion_threshold_sf_wr": 0.30,
}
```

## Implementation Plan

### Phase 1: Stage 1 trainer (beat random)

**Files to modify**:
- `python/neat_cpu_trainer.py`: Add `_train_vs_opponent()` method that plays against a fixed policy instead of coevolution
- `train_wandb.py`: Add `--curriculum` flag to run staged training

**Key change**: Replace coevolutionary pairings with "evolving population vs fixed random" pairings. White population plays as white against random black; black population plays as black against random white.

```python
def _generate_curriculum_pairings(self, stage):
    """Generate pairings against curriculum opponents."""
    if stage == 1:
        # Each individual plays N games vs random benchmark
        return [(i, j) for i in range(self.pop_size)
                for j in range(self.benchmark_size)]
```

**Fitness**: Since opponent is fixed, raw fitness = absolute skill. No coevolutionary noise. This gives NEAT a clean gradient to follow.

### Phase 2: Stage transitions

**Auto-promotion logic**:
```python
def _check_promotion(self, metrics, stage):
    if stage == 1:
        return metrics["bench_avg_win_rate"] > 0.80
    elif stage == 2:
        return metrics["vs_hof_win_rate"] > 0.70
    elif stage == 3:
        return metrics["sf_avg_cpl"] < 400
    elif stage == 4:
        return metrics["sf_win_rate"] > 0.30
```

When promoted:
1. Save current HoF as `stage{N}_hof.json`
2. Load as frozen opponents for next stage
3. Log stage transition to W&B

### Phase 3: Sweep integration

Curriculum stages can be swept independently:
```yaml
# Stage 1 sweep
parameters:
  curriculum_stage: {value: 1}
  temperature: {values: [0.05, 0.1, 0.2]}
  games_per_individual: {values: [10, 20, 30]}
  population_size: {values: [100, 200]}
```

### Phase 4: Full pipeline

Chain all stages via `--curriculum`:
```bash
python train_wandb.py --curriculum --chain 20
```

Each chain run:
1. Load current stage from `curriculum_state.json`
2. Train until promotion or max_generations
3. If promoted, save state and start next stage
4. If not promoted, save progress for next chain run

## Key Design Decisions

### Why not just use Stockfish from the start?
Stockfish skill 0 still plays better than anything NEAT can produce initially. Against SF, every game is a loss, fitness is flat, no gradient. The curriculum gives NEAT opponents it can *actually beat*, creating a learnable gradient.

### Why frozen opponents in Stage 2?
Coevolution is a moving target — the opponent changes each generation, making the fitness signal noisy. Frozen opponents give a stable target to improve against. Once the population is strong enough, coevolution becomes productive.

### Why low temperature?
With temperature=0.5, move selection is ~50% random regardless of network output. The network's influence on the game is diluted. At temperature=0.1, the network's top-scored move is chosen ~90% of the time, so weight mutations directly affect play. Low temperature = stronger selection pressure on network quality.

### Why more games per individual?
With 5 games, a lucky/unlucky draw can dominate fitness. With 20 games, fitness reliably reflects skill. The cost is 4x more games per generation, but with 384 outputs and fast Rust simulation, this is ~0.5s per gen.

## Success Criteria

| Stage | Metric | Target | What it proves |
|-------|--------|--------|----------------|
| 1 | bench_win_rate | >80% | Can capture pieces |
| 2 | vs_hof_win_rate | >70% | Can defend pieces |
| 3 | sf_avg_cpl | <400 | Makes reasonable moves |
| 4 | sf_win_rate vs skill 0 | >30% | Plays actual chess |

## Risk: Stage 1 itself may be too hard

If NEAT can't beat random even with clean fitness signal + low temperature + 20 games, then the architecture (389→384 sparse network) may be fundamentally unable to learn even basic capture heuristics. In that case:
- Try even smaller output space (e.g. 64 destination-only)
- Add hand-crafted features to the input (e.g. attack maps, piece values at each square)
- Consider HyperNEAT or ES-HyperNEAT which exploit spatial regularity in the chessboard
