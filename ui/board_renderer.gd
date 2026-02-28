extends Control

## Renders a chess board with pieces using Godot's drawing primitives.
## Supports move animation (tween-based) and interactive click handling.

const DEFAULT_SQUARE_SIZE := 60
const DEFAULT_BOARD_PADDING := 20.0

const LIGHT_COLOR := Color(0.94, 0.85, 0.71)
const DARK_COLOR := Color(0.71, 0.53, 0.39)
const HIGHLIGHT_COLOR := Color(0.8, 0.8, 0.2, 0.5)
const SELECTED_COLOR := Color(0.2, 0.6, 0.2, 0.6)
const LEGAL_MOVE_COLOR := Color(0.3, 0.3, 0.8, 0.4)
const LEGAL_CAPTURE_COLOR := Color(0.8, 0.3, 0.3, 0.5)

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

const BoardStateScript = preload("res://chess/board_state.gd")

var board_state: RefCounted = null  # BoardState
var last_move: Vector2i = Vector2i(-1, -1)
var _piece_textures: Dictionary = {}

# Animation state
var enable_animation: bool = false
var _animating: bool = false
var _anim_from_sq: int = -1
var _anim_to_sq: int = -1
var _anim_piece: int = 0
var _anim_progress: float = 0.0
var _anim_duration: float = 0.3
var _pending_state: RefCounted = null
var _prev_board: Array[int] = []
var _captured_piece: int = 0
var _captured_alpha: float = 1.0
signal animation_complete

# Interactive mode (for human play)
var interactive: bool = false
var selected_square: int = -1
var legal_moves_from_selected: PackedInt32Array = PackedInt32Array()
signal move_made(from_sq: int, to_sq: int)


func set_state(state: RefCounted) -> void:
	if not enable_animation or board_state == null or state == null or _animating:
		board_state = state
		queue_redraw()
		return

	# Detect which piece moved
	var old_board: Array[int] = board_state.board
	var new_board: Array[int] = state.board

	var moved_from := -1
	var moved_to := -1

	for sq in range(64):
		if old_board[sq] != 0 and new_board[sq] != old_board[sq]:
			if moved_from == -1 and (new_board[sq] == 0 or (old_board[sq] > 0) != (new_board[sq] > 0)):
				moved_from = sq
		if new_board[sq] != 0 and old_board[sq] != new_board[sq]:
			if old_board[sq] == 0 or (old_board[sq] > 0) != (new_board[sq] > 0):
				if moved_to == -1:
					moved_to = sq

	if moved_from != -1 and moved_to != -1:
		_start_animation(moved_from, moved_to, old_board[moved_from], old_board)
		_pending_state = state
	else:
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


func _start_animation(from_sq: int, to_sq: int, piece: int, old_board: Array[int]) -> void:
	_animating = true
	_anim_from_sq = from_sq
	_anim_to_sq = to_sq
	_anim_piece = piece
	_anim_progress = 0.0
	_prev_board = old_board.duplicate()
	_captured_piece = old_board[to_sq] if old_board[to_sq] != 0 else 0
	_captured_alpha = 1.0

	var tween := create_tween()
	tween.tween_method(_update_animation, 0.0, 1.0, _anim_duration)
	tween.tween_callback(_finish_animation)


func _update_animation(progress: float) -> void:
	_anim_progress = progress
	_captured_alpha = 1.0 - progress
	queue_redraw()


func _finish_animation() -> void:
	_animating = false
	if _pending_state:
		board_state = _pending_state
		_pending_state = null
	_anim_from_sq = -1
	_anim_to_sq = -1
	queue_redraw()
	animation_complete.emit()


func _gui_input(event: InputEvent) -> void:
	if not interactive or board_state == null:
		return
	if event is InputEventMouseButton and event.pressed and event.button_index == MOUSE_BUTTON_LEFT:
		var sq := _screen_to_square(event.position)
		if sq == -1:
			_deselect()
			return
		if selected_square != -1:
			var move_enc := BoardStateScript.encode_move(selected_square, sq)
			if legal_moves_from_selected.has(move_enc):
				move_made.emit(selected_square, sq)
				_deselect()
				return
		if board_state.is_friendly(sq):
			_select_square(sq)
		else:
			_deselect()


func _screen_to_square(pos: Vector2) -> int:
	var board_pixel_size: float = min(size.x, size.y)
	if board_pixel_size <= 0.0:
		return -1
	var coord_margin: float = max(12.0, board_pixel_size * 0.04)
	var padding: float = max(8.0, board_pixel_size * 0.03)
	var board_size_f: float = board_pixel_size - coord_margin * 2.0
	board_size_f = max(board_size_f, 8.0 + padding * 2.0)
	var square_size: float = (board_size_f - padding * 2.0) / 8.0
	var board_origin: Vector2 = Vector2(coord_margin + padding, coord_margin + padding)

	var local := pos - board_origin
	if local.x < 0.0 or local.y < 0.0:
		return -1
	var file := int(local.x / square_size)
	var rank := 7 - int(local.y / square_size)
	if file < 0 or file > 7 or rank < 0 or rank > 7:
		return -1
	return rank * 8 + file


func _select_square(sq: int) -> void:
	selected_square = sq
	var all_legal = board_state.generate_legal_moves()
	legal_moves_from_selected.clear()
	for move in all_legal:
		if BoardStateScript.move_from(move) == sq:
			legal_moves_from_selected.append(move)
	queue_redraw()


func _deselect() -> void:
	selected_square = -1
	legal_moves_from_selected.clear()
	queue_redraw()


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

	# Highlight selected square and legal moves (interactive mode)
	if interactive and selected_square >= 0:
		var sf := selected_square % 8
		var sr := selected_square / 8
		var sel_rect := Rect2(
			board_origin + Vector2(sf, 7 - sr) * square_size,
			Vector2(square_size, square_size)
		)
		draw_rect(sel_rect, SELECTED_COLOR)

		for move in legal_moves_from_selected:
			var to_sq := BoardStateScript.move_to(move)
			var tf := to_sq % 8
			var tr := to_sq / 8
			var is_capture: bool = board_state.board[to_sq] != 0
			var move_rect := Rect2(
				board_origin + Vector2(tf, 7 - tr) * square_size,
				Vector2(square_size, square_size)
			)
			draw_rect(move_rect, LEGAL_CAPTURE_COLOR if is_capture else LEGAL_MOVE_COLOR)

	# Draw pieces
	var draw_board_arr = _prev_board if _animating and not _prev_board.is_empty() else (board_state.board if board_state else null)
	if draw_board_arr:
		_ensure_piece_textures()
		var piece_scale := 0.9
		var piece_size := square_size * piece_scale
		var piece_offset := (square_size - piece_size) * 0.5
		for sq in range(64):
			var piece: int = draw_board_arr[sq]
			if piece == 0:
				continue

			# Skip the moving piece during animation (drawn separately)
			if _animating and sq == _anim_from_sq:
				continue

			# Draw captured piece with fade-out during animation
			if _animating and sq == _anim_to_sq and _captured_piece != 0:
				var f := sq % 8
				var r := sq / 8
				var top_left := board_origin + Vector2(f, 7 - r) * square_size
				var rect := Rect2(
					top_left + Vector2(piece_offset, piece_offset),
					Vector2(piece_size, piece_size)
				)
				var texture: Texture2D = _piece_textures.get(piece, null)
				if texture:
					draw_texture_rect(texture, rect, false, Color(1, 1, 1, _captured_alpha))
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

		# Draw the animating piece at interpolated position
		if _animating and _anim_piece != 0:
			var from_f := _anim_from_sq % 8
			var from_r := _anim_from_sq / 8
			var to_f := _anim_to_sq % 8
			var to_r := _anim_to_sq / 8

			var from_pos := board_origin + Vector2(from_f, 7 - from_r) * square_size
			var to_pos := board_origin + Vector2(to_f, 7 - to_r) * square_size
			var current_pos := from_pos.lerp(to_pos, _anim_progress)

			var rect := Rect2(
				current_pos + Vector2(piece_offset, piece_offset),
				Vector2(piece_size, piece_size)
			)
			var texture: Texture2D = _piece_textures.get(_anim_piece, null)
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
