extends "res://test/test_base.gd"

const ChessEvolutionScript = preload("res://ai/evolution.gd")
const ChessNeatEvolutionScript = preload("res://ai/chess_neat_evolution.gd")
const TrainingManagerScript = preload("res://ai/training_manager.gd")


func _run_tests() -> void:
    _test("training manager initializes", func():
        var tm = TrainingManagerScript.new()
        tm.use_minimax = false
        tm.use_tournament = false
        assert_not_null(tm.evolution)
        assert_eq(tm.evolution.generation, 0)
    )

    _test("run generation completes", func():
        var evo = ChessEvolutionScript.new(4, 389, 16, 128, 1, 0.15, 0.2, 0.7)
        var tm = TrainingManagerScript.new()
        tm.use_minimax = false
        tm.use_tournament = false
        tm.evolution = evo
        tm.games_per_individual = 1
        tm.max_moves_per_game = 20
        tm.run_generation()
        assert_eq(evo.generation, 1)
    )

    _test("stats returned correctly", func():
        var evo = ChessEvolutionScript.new(4, 389, 16, 128)
        var tm = TrainingManagerScript.new()
        tm.use_minimax = false
        tm.use_tournament = false
        tm.evolution = evo
        tm.games_per_individual = 1
        tm.max_moves_per_game = 10
        tm.run_generation()
        var stats: Dictionary = tm.get_stats()
        assert_eq(stats["generation"], 1)
        assert_true(stats.has("white_best"))
        assert_true(stats.has("black_best"))
    )

    _test("training manager enables NEAT when requested", func():
        var tm = TrainingManagerScript.new(null, 1, 10, "", true)
        assert_true(tm.use_neat)
        assert_true(tm.evolution is ChessNeatEvolutionScript)
        assert_false(tm.use_rust_batch_sim)
    )

    _test("training manager detects NEAT evolution instance", func():
        var evo = ChessNeatEvolutionScript.new(4)
        var tm = TrainingManagerScript.new(evo, 1, 10, "", false)
        assert_true(tm.use_neat)
        assert_true(tm.evolution is ChessNeatEvolutionScript)
    )
