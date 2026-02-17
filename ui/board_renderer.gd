extends Control

## Renders a chess board with pieces using Godot's drawing primitives.

const DEFAULT_SQUARE_SIZE := 60
const DEFAULT_BOARD_PADDING := 20.0

const LIGHT_COLOR := Color(0.94, 0.85, 0.71)
const DARK_COLOR := Color(0.71, 0.53, 0.39)
const HIGHLIGHT_COLOR := Color(0.8, 0.8, 0.2, 0.5)

const PIECE_TEXTURE_PATHS := {
	1: "res://assets/pieces/wP.png",
	2: "res://assets/pieces/wN.png",
	3: "res://assets/pieces/wB.png",
	4: "res://assets/pieces/wR.png",
	5: "res://assets/pieces/wQ.png",
	6: "res://assets/pieces/wK.png",
	-1: "res://assets/pieces/bP.png",
	-2: "res://assets/pieces/bN.png",
	-3: "res://assets/pieces/bB.png",
	-4: "res://assets/pieces/bR.png",
	-5: "res://assets/pieces/bQ.png",
	-6: "res://assets/pieces/bK.png",
}

var board_state: RefCounted = null  # BoardState
var last_move: Vector2i = Vector2i(-1, -1)
var _piece_textures: Dictionary = {}


func set_state(state: RefCounted) -> void:
	board_state = state
	queue_redraw()


func set_last_move(move: Vector2i) -> void:
	last_move = move
	queue_redraw()


func _notification(what: int) -> void:
	if what == NOTIFICATION_RESIZED:
		queue_redraw()


func _ensure_piece_textures() -> void:
	if not _piece_textures.is_empty():
		return
	for piece in PIECE_TEXTURE_PATHS.keys():
		_piece_textures[piece] = load(PIECE_TEXTURE_PATHS[piece])


func _draw() -> void:
	var board_pixel_size: float = min(size.x, size.y)
	if board_pixel_size <= 0.0:
		board_pixel_size = DEFAULT_SQUARE_SIZE * 8.0 + DEFAULT_BOARD_PADDING * 2.0
	var coord_margin: float = max(12.0, board_pixel_size * 0.04)
	var padding: float = max(8.0, board_pixel_size * 0.03)
	var board_size: float = board_pixel_size - coord_margin * 2.0
	board_size = max(board_size, 8.0 + padding * 2.0)
	var square_size: float = (board_size - padding * 2.0) / 8.0
	var board_origin: Vector2 = Vector2(coord_margin + padding, coord_margin + padding)
	var board_rect: Rect2 = Rect2(board_origin, Vector2(square_size, square_size) * 8.0)

	# Draw board squares
	for r in range(8):
		for f in range(8):
			var is_light := (f + r) % 2 == 1
			var color := LIGHT_COLOR if is_light else DARK_COLOR
			var rect := Rect2(
				board_origin + Vector2(f, 7 - r) * square_size,
				Vector2(square_size, square_size)
			)
			draw_rect(rect, color)

	# Highlight last move
	if last_move.x >= 0:
		for sq in [last_move.x, last_move.y]:
			var f: int = sq % 8
			var r: int = sq / 8
			var rect := Rect2(
				board_origin + Vector2(f, 7 - r) * square_size,
				Vector2(square_size, square_size)
			)
			draw_rect(rect, HIGHLIGHT_COLOR)

	# Draw pieces
	if board_state:
		_ensure_piece_textures()
		var piece_scale := 0.9
		var piece_size := square_size * piece_scale
		var piece_offset := (square_size - piece_size) * 0.5
		for sq in range(64):
			var piece: int = board_state.board[sq]
			if piece == 0:
				continue
			var f := sq % 8
			var r := sq / 8
			var top_left := board_origin + Vector2(f, 7 - r) * square_size
			var rect := Rect2(
				top_left + Vector2(piece_offset, piece_offset),
				Vector2(piece_size, piece_size)
			)
			var texture: Texture2D = _piece_textures.get(piece, null)
			if texture:
				draw_texture_rect(texture, rect, false)

	# Draw coordinates
	var font := ThemeDB.fallback_font
	var coord_font_size: float = max(10.0, square_size * 0.25)
	for f in range(8):
		var label := char("a".unicode_at(0) + f)
		draw_string(
			font,
			board_origin + Vector2(
				f * square_size + square_size * 0.4,
				board_rect.end.y + coord_margin * 0.4
			),
			label,
			HORIZONTAL_ALIGNMENT_LEFT,
			-1,
			coord_font_size
		)
	for r in range(8):
		draw_string(
			font,
			board_origin + Vector2(
				-coord_margin * 0.8,
				(7 - r) * square_size + square_size * 0.6
			),
			str(r + 1),
			HORIZONTAL_ALIGNMENT_LEFT,
			-1,
			coord_font_size
		)


func get_board_size() -> Vector2:
	return (
		Vector2(DEFAULT_SQUARE_SIZE, DEFAULT_SQUARE_SIZE) * 8.0
		+ Vector2.ONE * DEFAULT_BOARD_PADDING * 2.0
	)
