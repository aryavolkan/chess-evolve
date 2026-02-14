extends "res://test/test_base.gd"


func _run_tests() -> void:
	_test("training manager initializes", func():
		var tm := TrainingManager.new()
		assert_not_null(tm.evolution)
		assert_eq(tm.evolution.generation, 0)
	)

	_test("run generation completes", func():
		var evo := ChessEvolution.new(4, 389, 16, 128, 1, 0.15, 0.2, 0.7)
		var tm := TrainingManager.new(evo, 1, 20)  # 1 game each, max 20 moves
		tm.run_generation()
		assert_eq(evo.generation, 1)
	)

	_test("stats returned correctly", func():
		var evo := ChessEvolution.new(4, 389, 16, 128)
		var tm := TrainingManager.new(evo, 1, 10)
		tm.run_generation()
		var stats := tm.get_stats()
		assert_eq(stats["generation"], 1)
		assert_true(stats.has("white_best"))
		assert_true(stats.has("black_best"))
	)
