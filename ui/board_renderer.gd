extends Control

## Renders a chess board with pieces using Godot's drawing primitives.

const SQUARE_SIZE := 60
const BOARD_OFFSET := Vector2(20, 20)

const LIGHT_COLOR := Color(0.94, 0.85, 0.71)
const DARK_COLOR := Color(0.71, 0.53, 0.39)
const HIGHLIGHT_COLOR := Color(0.8, 0.8, 0.2, 0.5)

const PIECE_CHARS := {
	1: "♙", 2: "♘", 3: "♗", 4: "♖", 5: "♕", 6: "♔",  # White
	-1: "♟", -2: "♞", -3: "♝", -4: "♜", -5: "♛", -6: "♚",  # Black
}

var board_state: RefCounted = null  # BoardState
var last_move: Vector2i = Vector2i(-1, -1)


func set_state(state: RefCounted) -> void:
	board_state = state
	queue_redraw()


func set_last_move(move: Vector2i) -> void:
	last_move = move
	queue_redraw()


func _draw() -> void:
	# Draw board squares
	for r in 8:
		for f in 8:
			var is_light := (f + r) % 2 == 1
			var color := LIGHT_COLOR if is_light else DARK_COLOR
			var rect := Rect2(BOARD_OFFSET + Vector2(f, 7 - r) * SQUARE_SIZE, Vector2(SQUARE_SIZE, SQUARE_SIZE))
			draw_rect(rect, color)

	# Highlight last move
	if last_move.x >= 0:
		for sq in [last_move.x, last_move.y]:
			var f := sq % 8
			var r := sq / 8
			var rect := Rect2(BOARD_OFFSET + Vector2(f, 7 - r) * SQUARE_SIZE, Vector2(SQUARE_SIZE, SQUARE_SIZE))
			draw_rect(rect, HIGHLIGHT_COLOR)

	# Draw pieces
	if board_state:
		for sq in 64:
			var piece: int = board_state.board[sq]
			if piece == 0: continue
			var f := sq % 8
			var r := sq / 8
			var pos := BOARD_OFFSET + Vector2(f, 7 - r) * SQUARE_SIZE + Vector2(SQUARE_SIZE * 0.5, SQUARE_SIZE * 0.7)
			var ch: String = PIECE_CHARS.get(piece, "?")
			var font := ThemeDB.fallback_font
			draw_string(font, pos, ch, HORIZONTAL_ALIGNMENT_CENTER, -1, SQUARE_SIZE * 0.6)

	# Draw coordinates
	var font := ThemeDB.fallback_font
	for f in 8:
		var label := char("a".unicode_at(0) + f)
		draw_string(font, BOARD_OFFSET + Vector2(f * SQUARE_SIZE + SQUARE_SIZE * 0.4, 8 * SQUARE_SIZE + 35), label, HORIZONTAL_ALIGNMENT_LEFT, -1, 14)
	for r in 8:
		draw_string(font, BOARD_OFFSET + Vector2(-15, (7 - r) * SQUARE_SIZE + SQUARE_SIZE * 0.6), str(r + 1), HORIZONTAL_ALIGNMENT_LEFT, -1, 14)


func get_board_size() -> Vector2:
	return BOARD_OFFSET * 2 + Vector2(8, 8) * SQUARE_SIZE
