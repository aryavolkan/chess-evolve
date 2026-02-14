extends Control

## Main scene: training dashboard + board viewers showing live games.

const BoardStateScript = preload("res://chess/board_state.gd")
const BoardRenderer = preload("res://ui/board_renderer.gd")
const Dashboard = preload("res://ui/training_dashboard.gd")

var training_manager: TrainingManager
var dashboard: Control
var board_viewers: Array = []
var _showcase_timer := 0.0
const SHOWCASE_INTERVAL := 2.0  # Seconds between showcase game updates


func _ready() -> void:
	# Initialize training
	var evo := ChessEvolution.new(30, 389, 32, 128, 3)
	training_manager = TrainingManager.new(evo, 2, 100)

	# Dashboard on the right
	dashboard = Dashboard.new()
	dashboard.position = Vector2(780, 10)
	dashboard.training_manager = training_manager
	add_child(dashboard)

	# Create board viewers (2x2 grid showing showcase games)
	for i in 4:
		var board_view: Control = BoardRenderer.new()
		board_view.position = Vector2((i % 2) * 370 + 10, (i / 2) * 370 + 10)
		board_view.custom_minimum_size = Vector2(360, 360)
		add_child(board_view)
		board_viewers.append(board_view)

	# Show initial position on all boards
	var init_state := BoardStateScript.new()
	init_state.setup_initial()
	for viewer in board_viewers:
		viewer.set_state(init_state)

	# Connect training signal to update showcase
	training_manager.evolution.generation_complete.connect(_on_generation_complete)


func _on_generation_complete(gen: int, _wb: float, _bb: float) -> void:
	# Play a showcase game between best networks and display it
	if gen % 5 == 0:
		_play_showcase_game()


func _play_showcase_game() -> void:
	## Play a game between the best white and a random black, show on first board.
	var evo := training_manager.evolution
	if evo.all_time_best_white == null: return

	var state := BoardStateScript.new()
	state.setup_initial()
	var white_net = evo.all_time_best_white
	var black_net = evo.get_network(1, randi() % evo.population_size)

	var moves_played := 0
	while not state.is_game_over and moves_played < 80:
		var net = white_net if state.side_to_move == 0 else black_net
		var inputs := ChessEncoder.encode_board(state)
		var outputs := net.forward(inputs)
		var legal := state.generate_legal_moves()
		if legal.is_empty(): break
		var move := ChessEncoder.decode_move(outputs, legal)
		state.make_move(move)
		moves_played += 1

	# Show final position
	if board_viewers.size() > 0:
		board_viewers[0].set_state(state)
