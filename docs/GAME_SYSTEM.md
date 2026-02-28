# Game System

## Chess Rules Implementation

Chess-Evolve implements standard chess rules in `chess/board_state.gd` (pure GDScript) and optionally `chess/bitboard_state.gd` (Rust-accelerated bitboard variant).

---

## Board Representation

### Square Indexing

```
Rank 8:  56 57 58 59 60 61 62 63
Rank 7:  48 49 50 51 52 53 54 55
...
Rank 1:   0  1  2  3  4  5  6  7
         a  b  c  d  e  f  g  h
```

Row-major, 0-indexed. Square 0 = a1, square 63 = h8.

### Piece Encoding

Stored in `board: Array[int]` (64 elements):
- Positive values = white pieces
- Negative values = black pieces
- Zero = empty square

| Value | Piece |
|-------|-------|
| ±1 | Pawn |
| ±2 | Knight |
| ±3 | Bishop |
| ±4 | Rook |
| ±5 | Queen |
| ±6 | King |

---

## Board State Fields

```gdscript
var board: Array[int]          # 64 squares
var side_to_move: int          # 0=white, 1=black
var castling_rights: int       # bitmask: 0b KQkq
var en_passant_square: int     # target square, or -1
var halfmove_clock: int        # 50-move rule counter
var fullmove_number: int       # increments after black's move
var is_game_over: bool
var result: int                # 0=ongoing, 1=white wins, -1=black wins, 2=draw
```

---

## Move Encoding

Moves are packed into a single `int`:
```
move = from_square * 64 + to_square
```

Special moves (castling, en passant, promotion) are detected by context during `make_move()` rather than requiring additional bits.

### Legal Move Generation

`board_state.generate_legal_moves() → PackedInt32Array`

1. Generate all pseudo-legal moves (piece movement rules)
2. For each pseudo-legal move:
   - `make_move()`, check if own king is in check, `unmake_move()`
   - If not in check → legal
3. Results are cached on the board state (`_legal_moves_cache`) and invalidated on `make_move()` via the `_legal_moves_dirty` flag

The `BitboardState` variant uses Rust bitboard attack tables for significantly faster generation (pre-computed magic bitboard attacks for sliding pieces).

---

## Piece Movement

| Piece | Movement |
|-------|---------|
| Pawn | One forward, two from start, diagonal captures, en passant |
| Knight | L-shapes: 8 offset vectors |
| Bishop | Diagonal rays until blocked |
| Rook | Orthogonal rays until blocked |
| Queen | All 8 rays until blocked |
| King | One step in any direction, castling |

### Castling Rights Bitmask
```
bit 0 (0x1): White kingside  (K)
bit 1 (0x2): White queenside (Q)
bit 2 (0x4): Black kingside  (k)
bit 3 (0x8): Black queenside (q)
```

---

## Game Termination

Games end when:

| Condition | Result |
|-----------|--------|
| Checkmate | Win for the mating side |
| Stalemate | Draw |
| 50-move rule (`halfmove_clock >= 100`) | Draw |
| Insufficient material | Draw |
| Repetition (simplified: tracked in training manager) | Draw |
| `max_moves_per_game` reached | Draw (training only) |

---

## Material Values

Used in fitness evaluation and board assessment:

| Piece | Value |
|-------|-------|
| Pawn | 1.0 |
| Knight | 3.0 |
| Bishop | 3.25 |
| Rook | 5.0 |
| Queen | 9.0 |
| King | 0.0 (not counted) |

`board_state.material_score(color)` returns the sum of all piece values for the given side.

---

## Board State Methods (key API)

| Method | Description |
|--------|------------|
| `setup_initial()` | Set up standard starting position |
| `generate_legal_moves() → PackedInt32Array` | All legal moves for side to move |
| `make_move(move: int)` | Apply move, update state |
| `is_game_over → bool` | Check termination |
| `result → int` | 0=ongoing, 1=white, -1=black, 2=draw |
| `material_score(color) → float` | Sum of piece values for color |
| `mobility_score(color) → int` | Count of legal moves for color |
| `king_safety_score(color) → float` | King exposure heuristic |

---

## Encoder: Board → Neural Network Inputs

`ChessEncoder.encode_board(state) → PackedFloat32Array[389]`

```
Indices 0–383:  6 piece types × 64 squares
                +1.0 = white piece of that type on that square
                -1.0 = black piece of that type on that square
                 0.0 = empty / different piece type

Index 384:      side_to_move (0.0 = white, 1.0 = black)
Indices 385–388: castling rights bits (K, Q, k, q) as 0.0/1.0
```

The encoder caches its result on the `BoardState` to avoid redundant recomputation when the same position is encoded multiple times (e.g., multiple network evaluations before a move).

### Move Decoder

`ChessEncoder.decode_move(outputs, legal_moves) → int`

```
For each legal move (from_sq, to_sq):
    score = outputs[from_sq * 64 + to_sq]
return legal_move with highest score
```

Each of the 4096 outputs corresponds to one (from_square, to_square) pair. Only legal moves are scored; illegal move outputs are ignored.

---

## Replay System

The `GameRecorder` captures full game histories as replay files (saved to `user://replays/`). The `ReplayViewer` UI component loads these files and allows stepping through moves.

Launch with:
```bash
godot --path . -- --replay <filename>
```

The replay viewer searches for files in (in order):
1. Absolute path as given
2. `user://<filename>`
3. `user://replays/<filename>`
