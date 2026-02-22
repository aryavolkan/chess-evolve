extends Control

## Replay viewer for recorded chess games.
## Provides playback controls (play/pause, step, speed) and displays game metadata.
## Reuses BoardRenderer for board display.

signal replay_closed
const BoardRenderer = preload("res://ui/board_renderer.gd")
const GameRecorderScript = preload("res://ai/game_recorder.gd")


var _board_renderer: Control = null
var _states: Array = []  # Array of BoardState from replay
var _metadata: Dictionary = {}
var _current_index: int = 0
var _playing := false
var _playback_speed: float = 1.0  # Moves per second
var _elapsed: float = 0.0

# UI elements
var _meta_label: Label
var _move_label: Label
var _play_button: Button
var _prev_button: Button
var _next_button: Button
var _speed_label: Label
var _speed_up_button: Button
var _speed_down_button: Button
var _close_button: Button


func _ready() -> void:
    _build_ui()
    _update_display()


func load_replay_file(path: String) -> bool:
    ## Load a replay from file path. Returns true on success.
    var data := GameRecorderScript.load_from_file(path)
    if data.is_empty():
        return false
    return load_replay_data(data["moves"], data["metadata"])


func load_replay_data(moves: PackedInt32Array, metadata: Dictionary = {}) -> bool:
    ## Load replay from move list and optional metadata.
    _metadata = metadata
    _states = GameRecorderScript.replay_moves(moves)
    _current_index = 0
    _playing = false
    _update_display()
    return _states.size() > 0


func get_current_index() -> int:
    return _current_index


func get_total_states() -> int:
    return _states.size()


func is_playing() -> bool:
    return _playing


func _build_ui() -> void:
    var main_hbox := HBoxContainer.new()
    main_hbox.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
    add_child(main_hbox)

    # Board on the left
    _board_renderer = BoardRenderer.new()
    _board_renderer.custom_minimum_size = Vector2(400, 400)
    _board_renderer.size_flags_horizontal = Control.SIZE_EXPAND_FILL
    _board_renderer.size_flags_vertical = Control.SIZE_EXPAND_FILL
    main_hbox.add_child(_board_renderer)

    # Controls panel on the right
    var panel := VBoxContainer.new()
    panel.custom_minimum_size = Vector2(250, 0)
    panel.add_theme_constant_override("separation", 8)
    main_hbox.add_child(panel)

    _meta_label = Label.new()
    _meta_label.autowrap_mode = TextServer.AUTOWRAP_WORD
    panel.add_child(_meta_label)

    _move_label = Label.new()
    _move_label.add_theme_font_size_override("font_size", 18)
    panel.add_child(_move_label)

    # Playback controls
    var controls := HBoxContainer.new()
    panel.add_child(controls)

    _prev_button = Button.new()
    _prev_button.text = "◀"
    _prev_button.pressed.connect(_step_back)
    controls.add_child(_prev_button)

    _play_button = Button.new()
    _play_button.text = "▶"
    _play_button.pressed.connect(_toggle_play)
    controls.add_child(_play_button)

    _next_button = Button.new()
    _next_button.text = "▶▶"
    _next_button.pressed.connect(_step_forward)
    controls.add_child(_next_button)

    # Speed controls
    var speed_row := HBoxContainer.new()
    panel.add_child(speed_row)

    _speed_down_button = Button.new()
    _speed_down_button.text = "-"
    _speed_down_button.pressed.connect(_decrease_speed)
    speed_row.add_child(_speed_down_button)

    _speed_label = Label.new()
    _speed_label.text = "1.0x"
    speed_row.add_child(_speed_label)

    _speed_up_button = Button.new()
    _speed_up_button.text = "+"
    _speed_up_button.pressed.connect(_increase_speed)
    speed_row.add_child(_speed_up_button)

    _close_button = Button.new()
    _close_button.text = "Close"
    _close_button.pressed.connect(func(): replay_closed.emit())
    panel.add_child(_close_button)


func _process(delta: float) -> void:
    if not _playing or _states.is_empty():
        return

    _elapsed += delta
    var interval := 1.0 / _playback_speed
    if _elapsed >= interval:
        _elapsed -= interval
        if _current_index < _states.size() - 1:
            _current_index += 1
            _update_display()
        else:
            _playing = false
            _update_play_button()


func _step_forward() -> void:
    if _current_index < _states.size() - 1:
        _current_index += 1
        _update_display()


func _step_back() -> void:
    if _current_index > 0:
        _current_index -= 1
        _update_display()


func _toggle_play() -> void:
    _playing = not _playing
    _elapsed = 0.0
    _update_play_button()


func _increase_speed() -> void:
    _playback_speed = min(_playback_speed * 2.0, 16.0)
    _speed_label.text = "%.1fx" % _playback_speed


func _decrease_speed() -> void:
    _playback_speed = max(_playback_speed / 2.0, 0.25)
    _speed_label.text = "%.1fx" % _playback_speed


func _update_display() -> void:
    if _states.is_empty():
        return
    if _board_renderer:
        _board_renderer.set_state(_states[_current_index])
    if _move_label:
        _move_label.text = "Move %d / %d" % [_current_index, _states.size() - 1]
    if _meta_label:
        _meta_label.text = _format_metadata()
    _update_play_button()


func _update_play_button() -> void:
    if _play_button:
        _play_button.text = "⏸" if _playing else "▶"


func _format_metadata() -> String:
    if _metadata.is_empty():
        return "No metadata"
    var lines: Array[String] = []
    if _metadata.has("generation"):
        lines.append("Generation: %d" % _metadata["generation"])
    if _metadata.has("white_id"):
        lines.append("White: #%d" % _metadata["white_id"])
    if _metadata.has("black_id"):
        lines.append("Black: #%d" % _metadata["black_id"])
    if _metadata.has("result"):
        var r: int = _metadata["result"]
        var result_str := (
            "Draw" if r == 2
            else ("White wins" if r == 1 else ("Black wins" if r == -1 else "Unknown"))
        )
        lines.append("Result: %s" % result_str)
    if _metadata.has("total_moves"):
        lines.append("Total moves: %d" % _metadata["total_moves"])
    return "\n".join(lines)
