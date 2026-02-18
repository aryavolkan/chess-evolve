# Bitboard Representation & NEAT Port — Planning Document

**Project:** Chess-Evolve  
**Date:** 2026-02-17  
**Status:** Planning  

---

## Overview

Two major architectural upgrades are planned for Chess-Evolve:

1. **Bitboard representation** — replace the current `Array[int]` board with 12 64-bit integers, enabling bulk bitwise move generation instead of per-square iteration.
2. **NEAT port** — replace the fixed-topology genetic algorithm with NEAT (NeuroEvolution of Augmenting Topologies), porting the working implementation from the sibling Evolve project.

These are independent changes that can (and should) be sequenced: bitboards first, then NEAT, since bitboards change the encoder's input format which affects NEAT's initial topology.

---

## Part 1: Bitboard Representation

### Current State

`chess/board_state.gd` (509 LOC) represents the board as:

```gdscript
var board: Array[int] = []  # 64 squares, signed int, row-major
```

Move generation iterates over all 64 squares repeatedly per piece type. A rook's attack generation, for example, walks ray directions square by square with bounds checks in a GDScript loop. The encoder (`chess/encoder.gd`) then iterates over all 64 squares × 6 piece types = 384 iterations per forward pass.

**Performance cost:** Every call to `get_legal_moves()` is O(n_pieces × board_size). For a typical midgame position with ~30 legal moves, this means thousands of GDScript operations per evaluation step.

### Bitboard Model

A **bitboard** is a 64-bit integer where bit `n` represents square `n`. Each bit is 1 if a piece of a specific type occupies that square, 0 otherwise.

Represent the full board as 12 integers:

```gdscript
# White pieces
var bb_w_pawns:   int  # bit n = white pawn on square n
var bb_w_knights: int
var bb_w_bishops: int
var bb_w_rooks:   int
var bb_w_queens:  int
var bb_w_king:    int

# Black pieces
var bb_b_pawns:   int
var bb_b_knights: int
var bb_b_bishops: int
var bb_b_rooks:   int
var bb_b_queens:  int
var bb_b_king:    int

# Derived (computed on update, not stored)
# var white_occ = bb_w_pawns | bb_w_knights | ... | bb_w_king
# var black_occ = bb_b_pawns | ...
# var all_occ   = white_occ | black_occ
```

**GDScript note:** Godot 4 `int` is 64-bit signed. Bitwise operations (`&`, `|`, `^`, `~`, `<<`, `>>`) all work correctly. Unsigned right shift requires masking: `(x >> n) & ((1 << (64 - n)) - 1)`. The `popcount` and `bit_scan_forward` operations are not natively available but can be approximated with loops or precomputed lookup tables.

### Precomputed Attack Tables

Replace per-call ray walking with **static lookup tables** (computed once at class load):

```gdscript
# Non-sliding pieces: one entry per square
static var KNIGHT_ATTACKS: Array[int]   # 64 entries, each a bitboard of attack squares
static var KING_ATTACKS:   Array[int]   # 64 entries
static var PAWN_ATTACKS_W: Array[int]   # 64 entries (white pawn captures)
static var PAWN_ATTACKS_B: Array[int]   # 64 entries (black pawn captures)

# Sliding pieces: ray tables per direction
static var RAY_N:  Array[int]  # 64 rays going north
static var RAY_S:  Array[int]
static var RAY_E:  Array[int]
static var RAY_W:  Array[int]
static var RAY_NE: Array[int]
static var RAY_NW: Array[int]
static var RAY_SE: Array[int]
static var RAY_SW: Array[int]
```

These are populated once via `_static_init()`. A slider's attack bitboard then becomes:

```gdscript
# Rook attacks in north direction from square sq
func _ray_attacks_north(sq: int, occupancy: int) -> int:
    var attacks = RAY_N[sq]
    var blockers = attacks & occupancy
    if blockers:
        var first_blocker = _bit_scan_forward(blockers)
        attacks &= ~RAY_N[first_blocker]  # cut off ray beyond blocker
    return attacks
```

This is ~3 operations vs the current loop of 8 iterations with bounds checks.

### Move Generation Rewrite

New `get_legal_moves()` flow:

1. Compute `white_occ`, `black_occ`, `all_occ` from bitboard OR.
2. Per piece type, get **pseudo-legal** moves (attacks ignoring check) via lookup tables.
3. Filter to **legal** moves by testing: apply move → is own king in check?
4. Check detection: `KING_ATTACKS[king_sq] & enemy_king | KNIGHT_ATTACKS[king_sq] & enemy_knights | ...`

Result: a `PackedInt32Array` of `(from_sq << 6 | to_sq)` move integers, with special encoding for promotions and en passant.

### Updated Encoder

Bitboards become the encoding directly:

```gdscript
# New encoder: 12 bitboards × 64 bits → unpack to 768 floats
# + 5 metadata = 773 inputs (vs current 389)
const INPUT_SIZE_BITBOARD := 773

static func encode_board_bitboard(state: BitboardState) -> PackedFloat32Array:
    var inputs := PackedFloat32Array()
    inputs.resize(INPUT_SIZE_BITBOARD)
    # Unpack each bitboard to 64 floats (0.0 or 1.0)
    for bb_idx in 12:
        var bb: int = state.get_bb(bb_idx)
        for sq in 64:
            inputs[bb_idx * 64 + sq] = 1.0 if (bb >> sq) & 1 else 0.0
    # Side to move, castling (4 bits), en passant file (1)
    inputs[768] = float(state.side_to_move)
    inputs[769] = float((state.castling_rights >> 3) & 1)
    inputs[770] = float((state.castling_rights >> 2) & 1)
    inputs[771] = float((state.castling_rights >> 1) & 1)
    inputs[772] = float(state.castling_rights & 1)
    return inputs
```

This **eliminates the encoding loop entirely** — the bitboard unpacking is the same cost as the current encoding, but move generation is now decoupled from the encoding step.

### Migration Plan

| Phase | Work | Tests |
|-------|------|-------|
| 1a | Add `BitboardState` class alongside `BoardState` (parallel, not replacing) | New unit tests for bb operations |
| 1b | Implement precomputed tables and verify against `BoardState.get_legal_moves()` | Fuzzing: compare outputs on 1000 random positions |
| 1c | Swap `training_manager.gd` to use `BitboardState`; keep `BoardState` as fallback | Full integration test suite must pass |
| 1d | Update encoder to `INPUT_SIZE_BITBOARD = 773`; retrain from scratch | Benchmark: moves/second before and after |
| 1e | Delete `BoardState` after 1-week burn-in | — |

**Expected speedup:** 5–15× for move generation in GDScript. In a future GDExtension/Rust port (see NEAT section), bitboards are the standard representation and the speedup becomes 100–1000×.

### Estimated Effort

- `BitboardState` implementation: ~300 LOC
- Precomputed table generator: ~150 LOC  
- Encoder update: ~50 LOC
- Test suite (comparison fuzzing): ~100 LOC
- **Total: ~3–4 days**

---

## Part 2: NEAT Port

### Current State

`ai/evolution.gd` (158 LOC) implements a **fixed-topology genetic algorithm**:
- All networks have the same architecture: `389 → 64 → 128` (or configurable hidden size)
- Evolution = weight mutation + crossover of weight arrays
- No speciation, no topology change

`ai/neural_network.gd` (162 LOC) implements a **2-layer feedforward net**:
- Dense: `weights_ih[input_size × hidden_size]` + `weights_ho[hidden_size × output_size]`
- Activation: `tanh` throughout

This is functionally equivalent to the "dense weight GA" in early Evolve. The main Evolve project already evolved past this to full NEAT with speciation.

### What NEAT Adds

| Feature | Current | NEAT |
|---------|---------|------|
| Topology | Fixed | Evolves (add nodes/edges) |
| Crossover | Weight array blend | Historical marking alignment |
| Speciation | None | Species protect innovation |
| Complexity | Constant | Starts minimal, grows |
| Parameters | hidden_size, mutation_rate | complexity_threshold, species_target, c1, c2, c3 |

For chess specifically, NEAT's **minimal network start** is valuable: initial networks are linear (no hidden nodes) and grow complexity only when fitness pressure demands it. This avoids wasting generations evaluating bloated random networks.

### Source Material: Evolve's NEAT

The main Evolve project has a complete, tested NEAT implementation:

```
~/Projects/evolve/
  ai/
    genome.gd           # Gene, ConnectionGene, NodeGene, Genome
    neat_population.gd  # Population, Species, speciation, reproduction
    neat_config.gd      # All NEAT hyperparameters
    neural_network.gd   # Sparse network built from Genome
```

Key design decisions already solved in Evolve:
- **Innovation numbers** tracked globally in a static dict (avoids recalculation)
- **Speciation** via compatibility distance: `δ = c1·E/N + c2·D/N + c3·W̄`
- **Interspecies crossover** at configurable rate
- **Node types**: INPUT, HIDDEN, OUTPUT (no recurrent in current impl)
- **Activation**: tanh for hidden, passthrough for input/output

### Chess-NEAT Architecture

#### Input

With bitboards (Phase 1): **773 inputs**  
Without bitboards (Phase 0, faster to ship): keep **389 inputs**

Start with 389 for the initial NEAT port (avoids two simultaneous rewrites), then switch to 773 after bitboards are done.

#### Output

Current: 128 outputs (from_sq 0-63 → one logit, to_sq 0-63 → one logit)  
Better: **4096 outputs** (one per from-to pair), masked to legal moves

The 128-output scheme requires a "decode from two 64-vectors" step that adds brittleness. With NEAT's variable topology, output size should be locked early. **Recommend: 4096 outputs, masked softmax.**

```gdscript
func select_move(network: NeatNetwork, state: BoardState) -> Move:
    var outputs = network.forward(encode_board(state))  # 4096 outputs
    var legal_moves = state.get_legal_moves()
    
    # Build legal move mask
    var best_score := -INF
    var best_move: Move
    for move in legal_moves:
        var idx = move.from_sq * 64 + move.to_sq
        if outputs[idx] > best_score:
            best_score = outputs[idx]
            best_move = move
    return best_move
```

#### Initial Genome

NEAT starts minimal: one input node per feature, one output node per move slot, **zero hidden nodes**. Connections are added by mutation:

```
Initial topology: 389 input nodes → 4096 output nodes, NO hidden, NO connections
After setup_minimal(): 389 × 4096 = 1,594,304 possible connections... too many.
```

**Problem:** 4096 outputs with 389 inputs means `setup_minimal()` (which typically connects all inputs to all outputs) creates ~1.6M connections. This is too large to be useful.

**Solution options:**

1. **Sparse initial connections** — connect each output to a random K inputs (e.g. K=10), not all inputs. NEAT adds more as needed.
2. **Reduced output space** — 64 outputs (from_square only), then a second NN selects to_square conditioned on from. Two-network coevolution.
3. **Action encoding** — encode moves as indices into a sorted legal move list (max ~218 legal moves per position). Output size = 218 (max), masked to actual legal count.

**Recommendation: Option 3** — 218-output network with legal-move masking:
- Keeps output size tractable for NEAT topology evolution
- Avoids the illegal-move problem entirely
- Consistent with policy gradient approaches in AlphaZero-style research

```gdscript
const MAX_LEGAL_MOVES := 218  # theoretical maximum in chess
# Outputs: logits over the legal move list, padded to MAX_LEGAL_MOVES
# At inference: softmax over [:num_legal_moves], argmax = selected move index
```

### Speciation for Coevolution

The main Evolve project uses standard NEAT speciation (one fitness dimension). Chess-Evolve's coevolution (white brain vs black brain) requires additional care:

**Hall of Fame (HOF):** Maintain a HOF of the best historical opponents. Each network is evaluated against:
- Current generation opponents (fresh fitness signal)
- HOF opponents (stability / avoids forgetting)
- Combined fitness = `0.7 × current_gen_score + 0.3 × hof_score`

**Separate populations for white and black:** Two NEAT populations evolving simultaneously, evaluated head-to-head. This is already the pattern in Evolve's coevolution module (`ai/coevolution.gd`).

**Species target:** Start with `species_target = 5` (small population, manageable). Adjust `compatibility_threshold` dynamically to maintain target.

### Port Plan

#### Phase 0: Genome + Sparse Network (no topology change yet)

Port `genome.gd` and build a `SparseNetwork` from a fixed-topology genome. This lets the existing evolution loop use NEAT-encoded weights without full speciation. Validates the genome → network pipeline before adding NEAT-specific evolution.

#### Phase 1: Full NEAT Population

Port `neat_population.gd` with speciation. Replace `ai/evolution.gd` with `ai/neat_evolution.gd`. Two instances: one for white, one for black.

Key config changes:
```gdscript
# neat_config.gd (chess-specific)
const POPULATION_SIZE    := 100
const SPECIES_TARGET     := 8
const COMPAT_THRESHOLD   := 3.0
const C1                 := 1.0   # excess gene coefficient
const C2                 := 1.0   # disjoint gene coefficient
const C3                 := 0.4   # weight difference coefficient
const WEIGHT_MUTATE_RATE := 0.8
const NODE_ADD_RATE      := 0.03  # lower than games (chess is harder to evaluate)
const CONN_ADD_RATE      := 0.05
const INTERSPECIES_RATE  := 0.001
const HOF_SIZE           := 20
const HOF_WEIGHT         := 0.3
```

#### Phase 2: Bitboard Integration

After Phase 1 is stable, switch encoder to bitboard-based 773-input encoding. Input node count changes; existing genomes are incompatible. Start a fresh training run with new encoder.

#### Phase 3: GDExtension (Stretch)

The bottleneck will shift from move generation to NEAT forward pass (4096 outputs, variable topology). Port the hot path to Rust via GDExtension:
- Sparse matrix-vector multiply in Rust
- Bitboard move generation in Rust
- Called from GDScript for orchestration

This unlocks ~100× speedup over pure GDScript, enabling population sizes of 500+ and deeper networks.

### Migration Plan

| Phase | Work | Depends on |
|-------|------|------------|
| 0 | Port `Genome` + `SparseNetwork`; add genome tests | Nothing |
| 1a | Port `NeatPopulation` with speciation | Phase 0 |
| 1b | Port `NeatConfig` (chess-tuned values) | Phase 0 |
| 1c | Replace `evolution.gd` with NEAT in `training_manager.gd` | Phase 1a/b |
| 1d | Add HOF for coevolution stability | Phase 1c |
| 2 | Bitboard encoder → 773 inputs; retrain from scratch | Bitboard Part 1 |
| 3 | GDExtension Rust port | Phase 2 stable |

### Estimated Effort

| Phase | Effort |
|-------|--------|
| Phase 0: Genome + SparseNetwork | 1–2 days |
| Phase 1: Full NEAT | 3–4 days |
| Phase 2: Bitboard integration | 1 day (after bitboards done) |
| Phase 3: GDExtension | 5–7 days |

---

## Sequencing Recommendation

```
Week 1: Bitboard Phase 1a–1c (parallel BitboardState, verified against BoardState)
Week 2: Bitboard Phase 1d–1e (encoder update, delete old BoardState)
         + NEAT Phase 0 (Genome + SparseNetwork port)
Week 3: NEAT Phase 1a–1c (full population + speciation)
Week 4: NEAT Phase 1d (HOF coevolution) + integration
Week 5+: GDExtension if needed
```

Both changes together will make Chess-Evolve a significantly more capable and faster system — bitboards enabling real-time move generation, NEAT enabling the architecture to grow in complexity as chess understanding deepens.

---

## Open Questions

1. **Output size final decision:** 218 (legal move index) vs 4096 (all from-to pairs)? The 218 scheme is simpler but requires sorting legal moves consistently between evaluation and policy decoding.
2. **Coevolution fitness assignment:** Symmetric (white_score + black_score averaged per individual)? Or fully separate populations with asymmetric fitness?
3. **GDExtension language:** Rust (fastest, more complex FFI) vs C++ (GDExtension has better C++ support, more examples)?
4. **Reuse Evolve's NEAT directly** via a shared GDExtension plugin vs porting to chess-evolve standalone?

---

*Next step: assign a coder agent to Phase 0 (Genome + SparseNetwork port) as the first concrete PR.*
