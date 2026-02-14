extends "res://test/test_base.gd"

const ChessEvolutionScript = preload("res://ai/evolution.gd")


func _run_tests() -> void:
	_test("evolution initializes two populations", func():
		var evo = ChessEvolutionScript.new(10, 20, 8, 16)
		assert_eq(evo.white_pop.size(), 10)
		assert_eq(evo.black_pop.size(), 10)
	)

	_test("get_network returns valid network", func():
		var evo = ChessEvolutionScript.new(10, 20, 8, 16)
		var net = evo.get_network(0, 0)
		assert_not_null(net)
		assert_eq(net.input_size, 20)
	)

	_test("set and get fitness", func():
		var evo = ChessEvolutionScript.new(10, 20, 8, 16)
		evo.set_fitness(0, 3, 5.0)
		evo.set_fitness(1, 7, 8.0)
		assert_eq(evo.white_fitness[3], 5.0)
		assert_eq(evo.black_fitness[7], 8.0)
	)

	_test("evolve increments generation", func():
		var evo = ChessEvolutionScript.new(10, 20, 8, 16)
		for i in 10:
			evo.set_fitness(0, i, randf() * 10)
			evo.set_fitness(1, i, randf() * 10)
		evo.evolve()
		assert_eq(evo.generation, 1)
		assert_eq(evo.white_pop.size(), 10)
	)

	_test("evolution preserves population size", func():
		var evo = ChessEvolutionScript.new(20, 20, 8, 16)
		for i in 20:
			evo.set_fitness(0, i, randf() * 10)
			evo.set_fitness(1, i, randf() * 10)
		evo.evolve()
		assert_eq(evo.white_pop.size(), 20)
		assert_eq(evo.black_pop.size(), 20)
	)

	_test("avg fitness computed", func():
		var evo = ChessEvolutionScript.new(4, 20, 8, 16)
		evo.set_fitness(0, 0, 2.0)
		evo.set_fitness(0, 1, 4.0)
		evo.set_fitness(0, 2, 6.0)
		evo.set_fitness(0, 3, 8.0)
		assert_eq(evo.get_avg_fitness(0), 5.0)
	)
