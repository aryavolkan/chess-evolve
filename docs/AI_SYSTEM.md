# AI System

## Overview

Chess-Evolve uses **coevolutionary neuroevolution**: two separate populations (white and black) compete against each other. Each game is a fitness evaluation — there is no fixed target to optimise against. Two variants are supported: **standard NE** (fixed-topology feedforward networks) and **NEAT** (evolving network topology).

---

## Neural Network (Standard NE)

### Architecture

```
Inputs (389)  →  Hidden (tanh, default 64)  →  Outputs (tanh, 4096)
```

| Layer | Default size | Notes |
|-------|-------------|-------|
| Input | 389 | Board encoding (see below) |
| Hidden | 64 (configurable) | Single hidden layer, tanh activation |
| Output | 4096 | One logit per (from_square × 64 + to_square) pair |

### Board Encoding (389 inputs)

| Range | Contents |
|-------|---------|
| 0–383 | 6 piece-type planes × 64 squares. Value: +1.0 = white piece, −1.0 = black piece, 0.0 = empty |
| 384 | Side to move (0.0 = white, 1.0 = black) |
| 385–388 | Castling rights: KQkq bits (0.0 or 1.0 each) |

The encoder also caches the result on the `BoardState` object and marks it dirty on `make_move()`, so repeated calls within one position are free (`_encoder_dirty` flag).

### Move Decoding

```
For each legal move (packed as from_sq * 64 + to_sq):
    score = outputs[from_sq * 64 + to_sq]

best_move = legal_moves.argmax(score)
```

Each of the 4096 outputs corresponds to one (from_square, to_square) pair. Only legal moves are considered — illegal move outputs are ignored. This replaces an earlier 128-output factored scheme and allows the network to express move-specific preferences.

### Weight Initialisation
- `weights_ih` / `weights_ho`: He initialisation, scale = `sqrt(2 / fan_in)`
- Biases: uniform `[-0.1, 0.1]`

### Mutation

Uses **geometric-skip mutation** — O(k) calls where k is the number of actual mutations, not O(n) per-weight:
```
log1mp = log(1 - mutation_rate)
i = floor(log(randf()) / log1mp)   # first mutation index
while i < array.size():
    array[i] += randfn(0, mutation_strength)
    i += 1 + floor(log(randf()) / log1mp)
```

### Crossover

Two-point crossover operating directly on weight sub-arrays. Two random crossover points `p1 < p2` are drawn over the full flattened weight vector; segments `[p1, p2)` come from parent B, rest from parent A.

---

## Coevolution Engine (`evolution.gd`)

### Two-Population Structure

```
white_pop[pop_size]   # Networks playing as white
black_pop[pop_size]   # Networks playing as black
white_fitness[pop_size]
black_fitness[pop_size]
```

### Selection

**Tournament selection** (k=2, configurable): randomly sample k individuals, return the best.

### Evolution Step (`evolve()`)
1. Sort each population by fitness (descending)
2. **Elitism**: top `elite_count` cloned directly
3. Fill remainder: tournament select parent A; if `randf() < crossover_rate` also select parent B and crossover; always mutate
4. Track all-time best white/black individuals
5. Update Hall of Fame (see below)
6. Cache average fitness before resetting arrays
7. Optionally run adaptive mutation schedule
8. Increment generation, reset fitness arrays, emit `generation_complete`

### Adaptive Mutation

When `adaptive_mutation = true` (default), mutation rate and strength auto-adjust based on a rolling `stagnation_window` (default 5 gens):
- If best fitness improved vs. `_best_history[0]`: tighten (× 0.9, floored at min)
- If stagnant: loosen (× 1.1, capped at max)

| Parameter | Default |
|-----------|---------|
| `mutation_rate` | 0.15 |
| `mutation_strength` | 0.2 |
| `crossover_rate` | 0.7 |
| `mutation_rate_min/max` | 0.05 / 0.35 |
| `mutation_strength_min/max` | 0.05 / 0.4 |

---

## Fitness Function

Fitness is computed per-game by `ChessFitness.evaluate(state, color, move_count)`.

### Components

| Component | Default Weight | Formula |
|-----------|---------------|---------|
| Win bonus | 10.0 | Applied if player won |
| Checkmate bonus | 10.0 | Extra if win was by checkmate |
| Draw bonus | 0.0 | No bonus (material advantage provides small signal) |
| Loss penalty | -5.0 | Applied on loss |
| Material advantage | 1.0 | `(my_material - opp_material) x weight` |
| Mobility | 0.3 | `mobility_score(color) x weight` |
| Own king safety | 0.5 | `king_safety_score(own_color) x weight` |
| Opp king safety | 1.5 | Reward for attacking opponent's king exposure |
| King danger | 1.0 | `king_danger_score x weight` (attack signals) |
| Move count | -0.002 | `moves_played x weight` (penalty encourages decisive play) |

Material values: Pawn=1, Knight=3, Bishop=3.25, Rook=5, Queen=9.

Fitness is clamped to >= 0.

### Curriculum Learning

Training manager applies different fitness weights at different generation ranges:

| Stage | Gens | Max Moves | Key changes |
|-------|------|-----------|------------|
| Early | 0–2 | 60 | win_bonus=6, material=1.2, mobility=0.1 |
| Mid | 3–7 | 100 | win_bonus=8, draw=2, material=1.0 |
| Late | 8–14 | 130 | win_bonus=10, checkmate=5 added |
| Full | 15+ | 150 | Final weights with full king safety |

---

## Hall of Fame

Each evolution maintains two Hall of Fame arrays (`hall_of_fame` for white, `black_hall_of_fame` for black), max size 20 each.

### Entry Structure

```gdscript
{
    "network": NeuralNetwork,    # Frozen clone
    "fitness": float,
    "elo": float,                # Starts at 1200
    "generation": int,
    "games_played": int,
    "games_won": int,
    "games_drawn": int,
    "games_lost": int,
}
```

### Elo-Weighted Selection

When choosing a HoF opponent, selection is weighted by Elo using a softmax with temperature 200:
```
weight[i] = exp((elo[i] - min_elo) / 200.0)
```
Higher-rated entries are more likely to be selected, encouraging agents to learn from strong historical opponents.

### Tournament Mode

`training_manager.use_tournament = true` enables each individual to play against multiple opponents per generation, producing more stable fitness estimates. The `tournament_opponents` parameter controls how many opponents each individual faces.

---

## NEAT Variant

When `use_neat = true`, `ChessNeatEvolution` replaces `ChessEvolution`. NEAT uses the same two-population coevolution structure but with sparse, variable-topology networks.

- Input/output sizes: same as standard NE (389 / 4096, though NEAT config currently defaults to 128)
- Speciation with compatibility threshold (default configurable)
- Node mutation: split existing connection → insert hidden node
- Connection mutation: add random new connection
- Crossover: align by innovation number, inherit matching genes randomly, excess/disjoint from fitter parent

---

## Global Elite Pool

Enables parallel sweep workers to share their best genomes across runs (see `global_elite.py`). After each training run, a worker writes a contribution file; before the next run, it reads a merged seed file and pre-populates its Hall of Fame.

```
Worker 1 → elite_contrib_w1.json ──┐
Worker 2 → elite_contrib_w2.json ──┼→ merge → global_elite_wN.json → seed HoF
Worker 3 → elite_contrib_w3.json ──┘
```

Seeding via `evolution.seed_from_global_elite(pool)` adds entries to each side's Hall of Fame via `_seed_hall_of_fame()`, which calls the standard HoF sort/trim logic.

Genomes are serialised via `NeuralNetwork.to_dict()` / `load_from_dict()` (JSON-safe float arrays per layer).
