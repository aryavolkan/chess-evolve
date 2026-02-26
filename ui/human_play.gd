extends Control

## Human vs AI game controller.
## Allows a human to play chess against an evolved neural network.

signal game_finished(result: int)

const BoardStateScript = preload("res://chess/board_state.gd")
const BoardRenderer = preload("res://ui/board_renderer.gd")
const ChessEncoderScript = preload("res://chess/encoder.gd")
const MinimaxPlayerScript = preload("res://ai/minimax_player.gd")

var board_state = null
var _board_renderer: Control = null
var _status_label: Label = null
var _result_label: Label = null
var _new_game_button: Button = null
var _back_button: Button = null
var _move_count_label: Label = null

var human_color: int = 0  # 0=white, 1=black
var ai_network = null
var ai_use_minimax: bool = false
var ai_minimax_depth: int = 2
var _ai_player = null

var _game_active: bool = false
var _waiting_for_ai: bool = false


func _ready() -> void:
	_build_ui()


func setup_game(network, color: int = 0, use_minimax: bool = false, minimax_depth: int = 2) -> void:
	ai_network = network
	human_color = color
	ai_use_minimax = use_minimax
	ai_minimax_depth = minimax_depth
	if ai_use_minimax and ai_network:
		_ai_player = MinimaxPlayerScript.new(ai_network, ai_minimax_depth)
	_start_new_game()


func _start_new_game() -> void:
	board_state = BoardStateScript.new()
	board_state.setup_initial()
	_game_active = true
	_waiting_for_ai = false
	_board_renderer.set_state(board_state)
	_board_renderer.interactive = true
	if _result_label:
		_result_label.text = ""
	_update_status()

	# If AI plays first (human is black), make AI move
	if human_color == 1:
		_waiting_for_ai = true
		_board_renderer.interactive = false
		get_tree().create_timer(0.3).timeout.connect(_make_ai_move, CONNECT_ONE_SHOT)


func _build_ui() -> void:
	var main_hbox := HBoxContainer.new()
	main_hbox.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	add_child(main_hbox)

	# Board on the left
	_board_renderer = BoardRenderer.new()
	_board_renderer.custom_minimum_size = Vector2(500, 500)
	_board_renderer.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_board_renderer.size_flags_vertical = Control.SIZE_EXPAND_FILL
	_board_renderer.interactive = true
	_board_renderer.enable_animation = true
	_board_renderer.move_made.connect(_on_human_move)
	main_hbox.add_child(_board_renderer)

	# Controls panel on the right
	var panel := VBoxContainer.new()
	panel.custom_minimum_size = Vector2(220, 0)
	panel.add_theme_constant_override("separation", 12)
	main_hbox.add_child(panel)

	var title := Label.new()
	title.text = "Human vs AI"
	title.add_theme_font_size_override("font_size", 24)
	panel.add_child(title)

	_status_label = Label.new()
	_status_label.add_theme_font_size_override("font_size", 16)
	_status_label.autowrap_mode = TextServer.AUTOWRAP_WORD
	panel.add_child(_status_label)

	_move_count_label = Label.new()
	panel.add_child(_move_count_label)

	_result_label = Label.new()
	_result_label.add_theme_font_size_override("font_size", 20)
	panel.add_child(_result_label)

	# Spacer
	var spacer := Control.new()
	spacer.size_flags_vertical = Control.SIZE_EXPAND_FILL
	panel.add_child(spacer)

	_new_game_button = Button.new()
	_new_game_button.text = "New Game"
	_new_game_button.pressed.connect(_start_new_game)
	panel.add_child(_new_game_button)

	_back_button = Button.new()
	_back_button.text = "Back to Training"
	_back_button.pressed.connect(func(): game_finished.emit(0))
	panel.add_child(_back_button)


func _on_human_move(from_sq: int, to_sq: int) -> void:
	if not _game_active or _waiting_for_ai:
		return
	if board_state.side_to_move != human_color:
		return

	var move := BoardStateScript.encode_move(from_sq, to_sq)
	var legal := board_state.generate_legal_moves()
	if not legal.has(move):
		return

	board_state.make_move(move)
	_board_renderer.set_state(board_state)
	_board_renderer.set_last_move(Vector2i(from_sq, to_sq))
	_update_status()

	if board_state.is_game_over:
		_end_game()
		return

	# AI responds after a short delay
	_waiting_for_ai = true
	_board_renderer.interactive = false
	get_tree().create_timer(0.5).timeout.connect(_make_ai_move, CONNECT_ONE_SHOT)


func _make_ai_move() -> void:
	if not _game_active or board_state.is_game_over:
		_waiting_for_ai = false
		_board_renderer.interactive = true
		return

	var legal_moves := board_state.generate_legal_moves()
	if legal_moves.is_empty():
		_end_game()
		return

	var chosen := -1

	if ai_use_minimax and _ai_player:
		chosen = _ai_player.choose_move(board_state)
	elif ai_network:
		var inputs := ChessEncoderScript.encode_board(board_state)
		var outputs := ai_network.forward(inputs)
		chosen = ChessEncoderScript.decode_move(outputs, legal_moves)

	if chosen == -1:
		# Fallback to random legal move
		chosen = legal_moves[randi() % legal_moves.size()]

	var from_sq := BoardStateScript.move_from(chosen)
	var to_sq := BoardStateScript.move_to(chosen)

	board_state.make_move(chosen)
	_board_renderer.set_state(board_state)
	_board_renderer.set_last_move(Vector2i(from_sq, to_sq))

	_waiting_for_ai = false
	_board_renderer.interactive = true
	_update_status()

	if board_state.is_game_over:
		_end_game()


func _end_game() -> void:
	_game_active = false
	_board_renderer.interactive = false
	var result_text := ""
	match board_state.result:
		1: result_text = "White wins!"
		-1: result_text = "Black wins!"
		2: result_text = "Draw"
	if board_state.result != 0:
		var human_won := (board_state.result == 1 and human_color == 0) or \
			(board_state.result == -1 and human_color == 1)
		if human_won:
			result_text += "\nYou win!"
		elif board_state.result == 2:
			result_text += ""
		else:
			result_text += "\nAI wins!"
	_result_label.text = result_text
	game_finished.emit(board_state.result)


func _update_status() -> void:
	if not _game_active:
		return
	var turn := "White" if board_state.side_to_move == 0 else "Black"
	var is_human := board_state.side_to_move == human_color
	_status_label.text = "%s to move (%s)" % [turn, "You" if is_human else "AI"]
	if board_state.is_in_check():
		_status_label.text += " - CHECK!"
	_move_count_label.text = "Move %d" % board_state.fullmove_number
