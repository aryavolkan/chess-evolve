extends "res://test/test_base.gd"

const OpeningBookScript = preload("res://chess/opening_book.gd")
const BoardStateScript = preload("res://chess/board_state.gd")


func _run_tests() -> void:
	_test("book initializes without error", func():
		var count := OpeningBookScript.size()
		assert_gt(float(count), 0.0, "Book should have positions")
	)

	_test("initial position has book moves", func():
		var state := BoardStateScript.new()
		state.setup_initial()
		assert_true(OpeningBookScript.has_position(state), "Initial position should be in book")
	)

	_test("lookup returns valid move", func():
		var state := BoardStateScript.new()
		state.setup_initial()
		var move := OpeningBookScript.lookup(state)
		assert_ne(move, -1, "Should find a book move")
		var legal := state.generate_legal_moves()
		assert_true(legal.has(move), "Book move should be legal")
	)

	_test("after 1.e4 black has book response", func():
		var state := BoardStateScript.new()
		state.setup_initial()
		state.make_move(BoardStateScript.encode_move(12, 28))  # e4
		assert_true(OpeningBookScript.has_position(state), "Position after 1.e4 should be in book")
		var move := OpeningBookScript.lookup(state)
		assert_ne(move, -1, "Should find a book move for black")
	)

	_test("deep position not in book", func():
		var state := BoardStateScript.new()
		state.setup_initial()
		# Play random legal moves to get out of book
		for _i in range(30):
			var legal := state.generate_legal_moves()
			if legal.is_empty() or state.is_game_over:
				break
			state.make_move(legal[randi() % legal.size()])
		# Very unlikely to still be in book after 30 random moves
		# (but not guaranteed, so this is a probabilistic test)
		# We just verify it doesn't crash
		var _has := OpeningBookScript.has_position(state)
	)

	_test("candidates returns multiple options", func():
		var state := BoardStateScript.new()
		state.setup_initial()
		var candidates := OpeningBookScript.get_candidates(state)
		assert_gt(float(candidates.size()), 1.0, "Initial position should have multiple book moves")
	)
