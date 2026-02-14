extends Control

## Training dashboard showing stats and controls.

const TrainingManager = preload("res://ai/training_manager.gd")

const SPEED_OPTIONS = [1, 2, 4, 8]

var _gen_label: Label
var _white_best_label: Label
var _black_best_label: Label
var _white_avg_label: Label
var _black_avg_label: Label
var _games_label: Label
var _start_button: Button
var _speed_selector: OptionButton
var _stats_panel: VBoxContainer
var _pending_stats_refresh := false

var is_training := false
var train_speed := 1  # Generations per frame

var training_manager: TrainingManager = null:
	set(value):
		training_manager = value
		if training_manager:
			training_manager.training_step_complete.connect(_on_training_step_complete)
			if is_node_ready():
				_update_stats()
			else:
				_pending_stats_refresh = true


func _ready() -> void:
	_build_ui()
	Engine.time_scale = 1.0
	if _pending_stats_refresh:
		_pending_stats_refresh = false
		_update_stats()


func _exit_tree() -> void:
	Engine.time_scale = 1.0


func _build_ui() -> void:
	var vbox := VBoxContainer.new()
	vbox.position = Vector2(10, 10)
	vbox.custom_minimum_size = Vector2(320, 0)
	add_child(vbox)

	_gen_label = Label.new()
	_gen_label.add_theme_font_size_override("font_size", 20)
	_gen_label.text = "Generation: 0"
	vbox.add_child(_gen_label)

	_stats_panel = VBoxContainer.new()
	_stats_panel.add_theme_constant_override("separation", 6)
	vbox.add_child(_stats_panel)

	_white_best_label = _build_stat_label("White Best: 0.00")
	_white_avg_label = _build_stat_label("White Avg: 0.00")
	_black_best_label = _build_stat_label("Black Best: 0.00")
	_black_avg_label = _build_stat_label("Black Avg: 0.00")
	_games_label = _build_stat_label("Games Played: 0")

	_start_button = Button.new()
	_start_button.text = "Start Training"
	_start_button.pressed.connect(_on_start_pressed)
	vbox.add_child(_start_button)

	var speed_row := HBoxContainer.new()
	var speed_label := Label.new()
	speed_label.text = "Speed"
	speed_row.add_child(speed_label)

	_speed_selector = OptionButton.new()
	for idx in SPEED_OPTIONS.size():
		var multiplier: int = SPEED_OPTIONS[idx]
		_speed_selector.add_item("%dx" % multiplier, multiplier)
	_speed_selector.select(0)
	_speed_selector.item_selected.connect(_on_speed_selected)
	speed_row.add_child(_speed_selector)
	vbox.add_child(speed_row)


func _build_stat_label(text: String) -> Label:
	var label := Label.new()
	label.text = text
	_stats_panel.add_child(label)
	return label


func _on_start_pressed() -> void:
	is_training = not is_training
	_start_button.text = "Pause" if is_training else "Resume"
	if is_training:
		_update_stats()


func _on_speed_selected(index: int) -> void:
	var multiplier: int = SPEED_OPTIONS[index]
	train_speed = multiplier
	Engine.time_scale = float(multiplier)


func _on_training_step_complete(_generation: int, stats: Dictionary) -> void:
	_update_stats(stats)


func _process(_delta: float) -> void:
	if is_training and training_manager:
		# Run multiple game steps per frame based on speed
		var games_to_run := train_speed * 2  # 2 games per speed multiplier
		for _i in range(games_to_run):
			var gen_complete := training_manager.run_one_game_step()
			if gen_complete:
				_update_stats()
				break
		# Update stats periodically even mid-generation
		if training_manager.total_games_played % 5 == 0:
			_update_stats()


func _update_stats(stats: Dictionary = {}) -> void:
	if not training_manager:
		return
	var latest: Dictionary = stats if not stats.is_empty() else training_manager.get_stats()
	_gen_label.text = "Generation: %d" % latest.get("generation", 0)
	_white_best_label.text = "White Best: %.2f" % latest.get("white_best", 0.0)
	_white_avg_label.text = "White Avg: %.2f" % latest.get("white_avg", 0.0)
	_black_best_label.text = "Black Best: %.2f" % latest.get("black_best", 0.0)
	_black_avg_label.text = "Black Avg: %.2f" % latest.get("black_avg", 0.0)
	_games_label.text = "Games Played: %d" % int(latest.get("games_played", 0))
