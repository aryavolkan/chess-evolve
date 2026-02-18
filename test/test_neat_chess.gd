extends "res://test/test_base.gd"

const NeatConfigScript = preload("res://ai/neat_config.gd")
const NeatGenomeScript = preload("res://ai/neat_genome.gd")
const NeatInnovationScript = preload("res://ai/neat_innovation.gd")
const NeatSpeciesScript = preload("res://ai/neat_species.gd")
const ChessNeatPlayerScript = preload("res://ai/chess_neat_player.gd")
const ChessNeatEvolutionScript = preload("res://ai/chess_neat_evolution.gd")
const BoardStateScript = preload("res://chess/board_state.gd")


func _run_tests() -> void:
	_test("genome creation and mutation", func():
		var config := NeatConfigScript.new()
		config.input_count = 10
		config.output_count = 4
		config.initial_connections_per_output = 3
		var tracker := NeatInnovationScript.new(config.input_count + config.output_count)
		var genome := NeatGenomeScript.create(config, tracker)
		genome.create_basic()
		assert_gt(genome.connection_genes.size(), 0)
		genome.mutate(config)
		assert_gt(genome.node_genes.size(), 0)
	)

	_test("speciation with multiple genomes", func():
		var config := NeatConfigScript.new()
		config.input_count = 6
		config.output_count = 3
		config.initial_connections_per_output = 2
		var tracker := NeatInnovationScript.new(config.input_count + config.output_count)
		var g1 := NeatGenomeScript.create(config, tracker)
		g1.create_basic()
		var g2 := NeatGenomeScript.create(config, tracker)
		g2.create_basic()
		g2.mutate(config)
		var pop := [g1, g2]
		var result := NeatSpeciesScript.speciate(pop, [], config, 0)
		assert_gt(result.species.size(), 0)
	)

	_test("select_move returns valid legal move", func():
		var config := NeatConfigScript.new()
		config.input_count = 389
		config.output_count = 218
		var tracker := NeatInnovationScript.new(config.input_count + config.output_count)
		var genome := NeatGenomeScript.create(config, tracker)
		genome.create_basic()
		var player := ChessNeatPlayerScript.new(genome)
		var state := BoardStateScript.new()
		state.setup_initial()
		var legal_moves := state.generate_legal_moves()
		var chosen := player.select_move(state)
		assert_true(legal_moves.has(chosen))
	)

	_test("short evolution loop runs", func():
		var config := NeatConfigScript.new()
		config.population_size = 10
		var evo := ChessNeatEvolutionScript.new(10, config)
		for gen in 3:
			for i in evo.population_size:
				evo.set_fitness(0, i, randf())
				evo.set_fitness(1, i, randf())
			evo.evolve()
		assert_eq(evo.generation, 3)
	)
