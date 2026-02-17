extends Node

## Chess constants and piece definitions.

enum Piece { NONE = 0, PAWN = 1, KNIGHT = 2, BISHOP = 3, ROOK = 4, QUEEN = 5, KING = 6 }
enum Side { WHITE = 0, BLACK = 1 }

const BOARD_SIZE := 8

# Material values for fitness evaluation
const PIECE_VALUES := {
	Piece.PAWN: 1.0,
	Piece.KNIGHT: 3.0,
	Piece.BISHOP: 3.25,
	Piece.ROOK: 5.0,
	Piece.QUEEN: 9.0,
	Piece.KING: 0.0,  # King value not used for material counting
}

# Same values indexed by Piece enum (0-6) — O(1) array access vs dict lookup.
const PIECE_VALUES_ARRAY := [0.0, 1.0, 3.0, 3.25, 5.0, 9.0, 0.0]

# Piece symbols for display
const PIECE_SYMBOLS := {
	Piece.NONE: ".",
	Piece.PAWN: "P",
	Piece.KNIGHT: "N",
	Piece.BISHOP: "B",
	Piece.ROOK: "R",
	Piece.QUEEN: "Q",
	Piece.KING: "K",
}
