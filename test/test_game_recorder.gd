extends "res://test/test_base.gd"

## Tests for GameRecorder save/load functionality

const GameRecorder = preload("res://ai/game_recorder.gd")
const BoardState = preload("res://chess/board_state.gd")


func _run_tests() -> void:
	_test("save and load game", test_save_and_load_game)
	_test("replay moves", test_replay_moves)
	_test("save creates directory", test_save_creates_directory)
	_test("load nonexistent file", test_load_nonexistent_file)
	_test("list replays", test_list_replays)


func test_save_and_load_game() -> void:
	var moves: Array[Vector2i] = [
		Vector2i(12, 28),  # e2->e4
		Vector2i(52, 36),  # e7->e5
	]

	var metadata := {
		"generation": 5,
		"white_id": 10,
		"black_id": 20,
		"result": 2,  # Draw
		"total_moves": 2
	}

	var filename := "test_replay.json"

	# Create a recorder and save the game
	var recorder := GameRecorder.new()
	recorder.start_recording(metadata)
	var board := BoardState.new()
	board.setup_initial()
	for move in moves:
		board.make_move(move)
		recorder.record_move(move, board)
	recorder.stop_recording(2)
	recorder.save_to_file(filename)

	# Load it back
	var loaded := GameRecorder.load_from_file("user://replays/" + filename)

	assert_true(not loaded.is_empty(), "Loaded data should not be empty")
	assert_eq(loaded["moves"].size(), moves.size(), "Move count should match")
	assert_eq(loaded["metadata"]["generation"], 5, "Generation should match")
	assert_eq(loaded["metadata"]["white_id"], 10, "White ID should match")
	assert_eq(loaded["metadata"]["result"], 2, "Result should match")

	# Clean up
	DirAccess.remove_absolute("user://replays/" + filename)


func test_replay_moves() -> void:
	var moves: Array[Vector2i] = [
		Vector2i(52, 36),  # e2->e4 (52=e2, 36=e4 in 0-63 board indexing)
		Vector2i(12, 28),  # e7->e5 (12=e7, 28=e5)
	]

	var states := GameRecorder.replay_moves(moves)

	# Should have initial state + 2 moves = 3 states
	assert_eq(states.size(), 3, "Should have 3 states (initial + 2 moves)")

	# First state should be initial position
	var state0: BoardState = states[0]
	assert_eq(state0.side_to_move, 0, "Initial state should be white's turn")

	# Second state should have e4 played
	var state1: BoardState = states[1]
	assert_eq(state1.side_to_move, 1, "After white's move, black to move")

	# Third state should have e5 played
	var state2: BoardState = states[2]
	assert_eq(state2.side_to_move, 0, "After black's move, white to move")


func test_save_creates_directory() -> void:
	# Remove replays directory if it exists
	if DirAccess.dir_exists_absolute("user://replays"):
		var dir := DirAccess.open("user://replays")
		if dir:
			for file in dir.get_files():
				dir.remove(file)
		DirAccess.remove_absolute("user://replays")

	# Now save should create it
	var recorder := GameRecorder.new()
	recorder.start_recording({})
	var board := BoardState.new()
	board.setup_initial()
	recorder.record_move(Vector2i(0, 0), board)
	recorder.stop_recording(2)
	recorder.save_to_file("dir_test.json")

	assert_true(DirAccess.dir_exists_absolute("user://replays"), "Replays directory should be created")

	# Clean up
	DirAccess.remove_absolute("user://replays/dir_test.json")


func test_load_nonexistent_file() -> void:
	var loaded := GameRecorder.load_from_file("user://replays/nonexistent.json")
	assert_true(loaded.is_empty(), "Loading nonexistent file should return empty dict")


func test_list_replays() -> void:
	# Save a couple of test replays
	var recorder1 := GameRecorder.new()
	recorder1.start_recording({"test": 1})
	var board1 := BoardState.new()
	board1.setup_initial()
	recorder1.record_move(Vector2i(0, 0), board1)
	recorder1.stop_recording(2)
	recorder1.save_to_file("test1.json")

	var recorder2 := GameRecorder.new()
	recorder2.start_recording({"test": 2})
	var board2 := BoardState.new()
	board2.setup_initial()
	recorder2.record_move(Vector2i(1, 1), board2)
	recorder2.stop_recording(2)
	recorder2.save_to_file("test2.json")

	var files := GameRecorder.list_replays()
	assert_true(files.size() >= 2, "Should have at least 2 replay files")
	assert_true("test1.json" in files, "Should find test1.json")
	assert_true("test2.json" in files, "Should find test2.json")

	# Clean up
	DirAccess.remove_absolute("user://replays/test1.json")
	DirAccess.remove_absolute("user://replays/test2.json")
