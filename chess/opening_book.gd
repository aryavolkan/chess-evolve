class_name OpeningBook
extends RefCounted

## Embedded chess opening book. Maps board positions to known good moves.
## Uses move-sequence approach: openings defined as [from_sq, to_sq] pair sequences,
## replayed on a fresh board to build position -> move mapping.

const BoardStateScript = preload("res://chess/board_state.gd")

static var _book: Dictionary = {}  # position_key -> Array of {move: int, weight: float}
static var _initialized: bool = false


static func lookup(state) -> int:
	## Look up a position in the opening book. Returns an encoded move or -1.
	_ensure_initialized()
	var key := state.get_position_key()
	if not _book.has(key):
		return -1
	var candidates: Array = _book[key]
	if candidates.is_empty():
		return -1
	# Weighted random selection
	var total_weight := 0.0
	for c in candidates:
		total_weight += c.weight
	var roll := randf() * total_weight
	var cumulative := 0.0
	for c in candidates:
		cumulative += c.weight
		if roll <= cumulative:
			return c.move
	return candidates[0].move


static func has_position(state) -> bool:
	_ensure_initialized()
	return _book.has(state.get_position_key())


static func get_candidates(state) -> Array:
	## Returns all book moves for a position as Array of {move: int, weight: float}.
	_ensure_initialized()
	var key := state.get_position_key()
	if not _book.has(key):
		return []
	return _book[key]


static func size() -> int:
	_ensure_initialized()
	return _book.size()


static func _ensure_initialized() -> void:
	if _initialized:
		return
	_initialized = true
	_build_book()


static func _build_book() -> void:
	# Each opening: [name, weight, [[from, to], [from, to], ...]]
	# Higher weight = more likely to be chosen
	# Squares: a1=0, b1=1, ..., h1=7, a2=8, ..., h8=63
	var openings := [
		# === e4 openings (e2=12, e4=28) ===
		# Italian Game: 1.e4 e5 2.Nf3 Nc6 3.Bc4
		[1.0, [[12,28], [52,36], [6,21], [57,42], [5,26]]],
		# Ruy Lopez: 1.e4 e5 2.Nf3 Nc6 3.Bb5
		[1.0, [[12,28], [52,36], [6,21], [57,42], [5,33]]],
		# Scotch Game: 1.e4 e5 2.Nf3 Nc6 3.d4
		[0.7, [[12,28], [52,36], [6,21], [57,42], [11,27]]],
		# King's Gambit: 1.e4 e5 2.f4
		[0.5, [[12,28], [52,36], [13,29]]],
		# Sicilian Defense: 1.e4 c5
		[1.0, [[12,28], [50,34]]],
		# Sicilian Najdorf: 1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 a6
		[0.8, [[12,28], [50,34], [6,21], [51,43], [11,27], [34,27], [21,27], [62,45], [1,18], [48,40]]],
		# French Defense: 1.e4 e6
		[0.9, [[12,28], [52,44]]],
		# French Advance: 1.e4 e6 2.d4 d5 3.e5
		[0.7, [[12,28], [52,44], [11,27], [51,35], [28,36]]],
		# Caro-Kann: 1.e4 c6
		[0.9, [[12,28], [50,42]]],
		# Caro-Kann Classical: 1.e4 c6 2.d4 d5 3.Nc3
		[0.7, [[12,28], [50,42], [11,27], [51,35], [1,18]]],
		# Scandinavian: 1.e4 d5
		[0.6, [[12,28], [51,35]]],
		# Pirc Defense: 1.e4 d6 2.d4 Nf6
		[0.6, [[12,28], [51,43], [11,27], [62,45]]],

		# === d4 openings (d2=11, d4=27) ===
		# Queen's Gambit: 1.d4 d5 2.c4
		[1.0, [[11,27], [51,35], [10,26]]],
		# QGD: 1.d4 d5 2.c4 e6
		[0.9, [[11,27], [51,35], [10,26], [52,44]]],
		# QGA: 1.d4 d5 2.c4 dxc4
		[0.7, [[11,27], [51,35], [10,26], [35,26]]],
		# Slav: 1.d4 d5 2.c4 c6
		[0.8, [[11,27], [51,35], [10,26], [50,42]]],
		# King's Indian: 1.d4 Nf6 2.c4 g6
		[0.9, [[11,27], [62,45], [10,26], [54,46]]],
		# King's Indian Classical: 1.d4 Nf6 2.c4 g6 3.Nc3 Bg7 4.e4 d6
		[0.7, [[11,27], [62,45], [10,26], [54,46], [1,18], [61,46], [12,28], [51,43]]],
		# Nimzo-Indian: 1.d4 Nf6 2.c4 e6 3.Nc3 Bb4
		[0.8, [[11,27], [62,45], [10,26], [52,44], [1,18], [61,25]]],
		# Queen's Indian: 1.d4 Nf6 2.c4 e6 3.Nf3 b6
		[0.7, [[11,27], [62,45], [10,26], [52,44], [6,21], [49,41]]],
		# Grunfeld: 1.d4 Nf6 2.c4 g6 3.Nc3 d5
		[0.7, [[11,27], [62,45], [10,26], [54,46], [1,18], [51,35]]],
		# Dutch Defense: 1.d4 f5
		[0.5, [[11,27], [53,37]]],
		# London System: 1.d4 d5 2.Bf4
		[0.8, [[11,27], [51,35], [2,29]]],

		# === Other first moves ===
		# English: 1.c4
		[0.7, [[10,26]]],
		# English with e5: 1.c4 e5
		[0.6, [[10,26], [52,36]]],
		# Reti: 1.Nf3 d5 2.g3
		[0.6, [[6,21], [51,35], [14,22]]],
		# Reti with c4: 1.Nf3 d5 2.c4
		[0.6, [[6,21], [51,35], [10,26]]],

		# === Black responses to 1.e4 (give e4 more weight as initial move) ===
		# Alekhine's Defense: 1.e4 Nf6
		[0.4, [[12,28], [62,45]]],
		# Modern Defense: 1.e4 g6
		[0.4, [[12,28], [54,46]]],
	]

	for opening in openings:
		var weight: float = opening[0]
		var moves: Array = opening[1]
		_add_opening_sequence(moves, weight)


static func _add_opening_sequence(move_pairs: Array, weight: float) -> void:
	## Replay a move sequence, recording each position -> next move.
	var state := BoardStateScript.new()
	state.setup_initial()

	for i in move_pairs.size():
		var pair: Array = move_pairs[i]
		var from_sq: int = int(pair[0])
		var to_sq: int = int(pair[1])
		var encoded_move := BoardStateScript.encode_move(from_sq, to_sq)

		# Record this position -> move in the book
		var key := state.get_position_key()
		if not _book.has(key):
			_book[key] = []

		# Check if move already exists for this position
		var found := false
		for entry in _book[key]:
			if entry.move == encoded_move:
				entry.weight += weight
				found = true
				break
		if not found:
			_book[key].append({"move": encoded_move, "weight": weight})

		# Apply the move to advance position
		var legal_moves := state.generate_legal_moves()
		if not legal_moves.has(encoded_move):
			push_warning("OpeningBook: Illegal move in sequence at step %d: %d->%d" % [i, from_sq, to_sq])
			break
		state.make_move(encoded_move)
