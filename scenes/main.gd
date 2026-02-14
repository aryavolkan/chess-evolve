extends Control

## Main scene: training dashboard + board viewers showing live games.

const BoardStateScript = preload("res://chess/board_state.gd")
const BoardRenderer = preload("res://ui/board_renderer.gd")
const Dashboard = preload("res://ui/training_dashboard.gd")
const ChessEvolution = preload("res://ai/evolution.gd")
const TrainingManager = preload("res://ai/training_manager.gd")
const ChessEncoder = preload("res://chess/encoder.gd")

var training_manager: TrainingManager
var dashboard: Control
var board_viewers: Array[Control] = []
var _showcase_timer := 0.0
const SHOWCASE_INTERVAL := 2.0  # Seconds between showcase game updates

var _is_wide_layout := true
var _current_state: RefCounted = null
var _pending_moves: Array = []  # Queue of individual moves to display
var _game_display_idx := 0  # Which board to update next
var _time_since_last_update := 0.0
const UPDATE_INTERVAL := 0.3  # Seconds between move updates (slows down visualization)


func _ready() -> void:
	# Check for auto-train mode (headless training)
	var auto_train := _check_auto_train_arg()
	
	if auto_train:
		_run_auto_train()
		return
	
	# Make this control fill the viewport
	set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	
	# Initialize training with UI
	var config := _load_config()
	var evo := ChessEvolution.new(
		config.get("population_size", 30),
		config.get("input_size", 389),
		config.get("hidden_size", 32),
		config.get("output_size", 128),
		config.get("elite_count", 3)
	)
	training_manager = TrainingManager.new(
		evo,
		config.get("games_per_individual", 2),
		config.get("max_moves_per_game", 100)
	)

	# Dashboard on the right side
	dashboard = Dashboard.new()
	dashboard.training_manager = training_manager
	add_child(dashboard)
	
	_layout_ui()

	# Show initial position on all boards
	_current_state = BoardStateScript.new()
	_current_state.setup_initial()
	for viewer in board_viewers:
		viewer.set_state(_current_state)

	# Connect training signals
	training_manager.evolution.generation_complete.connect(_on_generation_complete)
	training_manager.game_complete.connect(_on_game_complete)


func _notification(what: int) -> void:
	if what == NOTIFICATION_RESIZED:
		_layout_ui()


func _layout_ui() -> void:
	if not is_inside_tree() or dashboard == null:
		return
	var vp_size := get_viewport_rect().size
	var is_wide := vp_size.x >= 800
	if is_wide != _is_wide_layout:
		_is_wide_layout = is_wide
		_clear_board_viewers()

	if is_wide:
		dashboard.position = Vector2(vp_size.x - 340, 10)
		dashboard.custom_minimum_size = Vector2(320, 0)
		dashboard.size = Vector2(330, vp_size.y - 20)
		_ensure_board_count(4)
		_layout_board_grid(Vector2(10, 10), vp_size.x - 360, vp_size.y - 20)
	else:
		dashboard.position = Vector2(10, 10)
		dashboard.custom_minimum_size = Vector2(320, 0)
		dashboard.size = Vector2(vp_size.x - 20, 200)
		_ensure_board_count(1)
		var dashboard_height := 200
		var board_area_y := dashboard_height + 20
		var board_size: float = min(vp_size.x - 20, vp_size.y - board_area_y - 10)
		_layout_single_board(Vector2(10, board_area_y), board_size)

	if _current_state:
		for viewer in board_viewers:
			viewer.set_state(_current_state)


func _ensure_board_count(count: int) -> void:
	if board_viewers.size() == count:
		return
	_clear_board_viewers()
	for _i in count:
		var board_view: Control = BoardRenderer.new()
		add_child(board_view)
		board_viewers.append(board_view)


func _clear_board_viewers() -> void:
	for viewer in board_viewers:
		viewer.queue_free()
	board_viewers.clear()


func _layout_board_grid(start_pos: Vector2, available_width: float, available_height: float) -> void:
	"""Layout a 2x2 grid of boards in the available space."""
	var board_size: float = min(available_width / 2 - 15, available_height / 2 - 15)
	board_size = max(board_size, 100)

	for i in range(board_viewers.size()):
		var board_view: Control = board_viewers[i]
		var col := i % 2
		var row := i / 2
		board_view.position = start_pos + Vector2(col * (board_size + 10), row * (board_size + 10))
		board_view.custom_minimum_size = Vector2(board_size, board_size)
		board_view.size = Vector2(board_size, board_size)


func _layout_single_board(start_pos: Vector2, board_size: float) -> void:
	"""Layout a single centered board for narrow viewports."""
	if board_viewers.is_empty():
		return
	var board_view: Control = board_viewers[0]
	board_view.position = start_pos
	board_view.custom_minimum_size = Vector2(board_size, board_size)
	board_view.size = Vector2(board_size, board_size)


func _process(delta: float) -> void:
	_time_since_last_update += delta
	
	# Show moves one at a time across all boards
	if _time_since_last_update >= UPDATE_INTERVAL and not _pending_moves.is_empty():
		_time_since_last_update = 0.0
		if board_viewers.size() > 0:
			var move_state = _pending_moves.pop_front()
			var board_idx := _game_display_idx % board_viewers.size()
			board_viewers[board_idx].set_state(move_state)
			_game_display_idx += 1


func _on_game_complete(_white_idx: int, _black_idx: int, _result: int) -> void:
	# Queue all moves from the completed game for display
	if training_manager and not training_manager.last_game_history.is_empty():
		# Only keep last 60 moves in queue (prevent memory buildup)
		if _pending_moves.size() < 60:
			for state in training_manager.last_game_history:
				_pending_moves.append(state)
				if _pending_moves.size() >= 60:
					break


func _on_generation_complete(gen: int, _wb: float, _bb: float) -> void:
	# Play a showcase game between best networks and display it
	if gen % 5 == 0:
		_play_showcase_game()


func _play_showcase_game() -> void:
	## Play a game between the best white and a random black, show on first board.
	var evo: ChessEvolution = training_manager.evolution
	if evo.all_time_best_white == null: return

	var state := BoardStateScript.new()
	state.setup_initial()
	var white_net = evo.all_time_best_white
	var black_net = evo.get_network(1, randi() % evo.population_size)

	var moves_played := 0
	while not state.is_game_over and moves_played < 80:
		var net = white_net if state.side_to_move == 0 else black_net
		var inputs: PackedFloat32Array = ChessEncoder.encode_board(state)
		var outputs: PackedFloat32Array = net.forward(inputs)
		var legal := state.generate_legal_moves()
		if legal.is_empty(): break
		var move: Vector2i = ChessEncoder.decode_move(outputs, legal)
		state.make_move(move)
		moves_played += 1

	# Show final position
	_current_state = state
	if board_viewers.size() > 0:
		board_viewers[0].set_state(state)


func _check_auto_train_arg() -> bool:
	## Check if --auto-train command-line argument is present
	var args := OS.get_cmdline_user_args()
	return "--auto-train" in args


func _load_config() -> Dictionary:
	## Load training config from user:// or use defaults
	var config_path := "user://sweep_config.json"
	var config := {}
	
	if FileAccess.file_exists(config_path):
		var file := FileAccess.open(config_path, FileAccess.READ)
		if file:
			var json := JSON.new()
			var parse_result := json.parse(file.get_as_text())
			if parse_result == OK:
				config = json.get_data()
				print("Loaded config from %s" % config_path)
			file.close()
	
	return config


func _run_auto_train() -> void:
	## Run headless training without UI (for sweep/overnight runs)
	print("Starting auto-train mode...")
	
	var config := _load_config()
	var evo := ChessEvolution.new(
		config.get("population_size", 30),
		config.get("input_size", 389),
		config.get("hidden_size", 64),
		config.get("output_size", 128),
		config.get("elite_count", 3)
	)
	
	# Set mutation params
	if config.has("mutation_rate"):
		evo.mutation_rate = config["mutation_rate"]
	if config.has("mutation_strength"):
		evo.mutation_strength = config["mutation_strength"]
	
	training_manager = TrainingManager.new(
		evo,
		config.get("games_per_individual", 2),
		config.get("max_moves_per_game", 100)
	)
	
	var max_generations := int(config.get("max_generations", 100))
	
	print("Config: pop=%d, hidden=%d, elite=%d, games_per=%d, max_gen=%d" % [
		evo.population_size, evo.hidden_size, evo.elite_count,
		training_manager.games_per_individual, max_generations
	])
	
	# Run training loop
	var total_games := 0
	for gen in max_generations:
		training_manager.run_generation()
		total_games += evo.population_size * config.get("games_per_individual", 2)
		
		# Get stats and write metrics.json for W&B polling
		var stats := training_manager.get_stats()
		_write_metrics(gen, stats, total_games)
		
		if gen % 10 == 0:
			print("Gen %d: W_best=%.1f, W_avg=%.1f, B_best=%.1f, B_avg=%.1f" % [
				gen, stats["white_best"], stats["white_avg"],
				stats["black_best"], stats["black_avg"]
			])
	
	print("Auto-train complete!")
	get_tree().quit()


func _write_metrics(gen: int, stats: Dictionary, total_games: int) -> void:
	## Write metrics.json for W&B polling
	var metrics := {
		"generation": gen,
		"white_best": stats.get("white_best", 0.0),
		"white_avg": stats.get("white_avg", 0.0),
		"black_best": stats.get("black_best", 0.0),
		"black_avg": stats.get("black_avg", 0.0),
		"games_played": total_games
	}
	
	var metrics_path := "user://metrics.json"
	var file := FileAccess.open(metrics_path, FileAccess.WRITE)
	if file:
		file.store_string(JSON.stringify(metrics, "\t"))
		file.close()
