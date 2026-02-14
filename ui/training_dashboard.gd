extends Control

## Training dashboard showing stats and controls.

var _stats_label: Label
var _gen_label: Label
var _start_button: Button
var _speed_slider: HSlider

var training_manager: TrainingManager = null
var is_training := false
var train_speed := 1  # Generations per frame


func _ready() -> void:
	_build_ui()


func _build_ui() -> void:
	var vbox := VBoxContainer.new()
	vbox.position = Vector2(10, 10)
	add_child(vbox)

	_gen_label = Label.new()
	_gen_label.text = "Generation: 0"
	vbox.add_child(_gen_label)

	_stats_label = Label.new()
	_stats_label.text = "Awaiting training..."
	_stats_label.custom_minimum_size = Vector2(250, 100)
	vbox.add_child(_stats_label)

	_start_button = Button.new()
	_start_button.text = "Start Training"
	_start_button.pressed.connect(_on_start_pressed)
	vbox.add_child(_start_button)

	var speed_label := Label.new()
	speed_label.text = "Speed:"
	vbox.add_child(speed_label)

	_speed_slider = HSlider.new()
	_speed_slider.min_value = 1
	_speed_slider.max_value = 10
	_speed_slider.value = 1
	_speed_slider.custom_minimum_size = Vector2(200, 20)
	_speed_slider.value_changed.connect(func(v): train_speed = int(v))
	vbox.add_child(_speed_slider)


func _on_start_pressed() -> void:
	is_training = not is_training
	_start_button.text = "Pause" if is_training else "Resume"


func _process(_delta: float) -> void:
	if is_training and training_manager:
		for _i in train_speed:
			training_manager.run_generation()
		_update_stats()


func _update_stats() -> void:
	if not training_manager: return
	var stats := training_manager.get_stats()
	_gen_label.text = "Generation: %d" % stats["generation"]
	_stats_label.text = "White Best: %.2f  Avg: %.2f\nBlack Best: %.2f  Avg: %.2f\nPop Size: %d" % [
		stats["white_best"], stats["white_avg"],
		stats["black_best"], stats["black_avg"],
		stats["population_size"],
	]
