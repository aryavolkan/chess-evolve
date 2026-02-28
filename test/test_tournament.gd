extends "res://test/test_base.gd"

## Unit tests for tournament-based fitness evaluation

const TrainingManager = preload("res://ai/training_manager.gd")
const ChessEvolution = preload("res://ai/evolution.gd")

func test_round_robin_pairings() -> void:
    ## Test that round-robin generates appropriate pairings
    var evo := ChessEvolution.new(10, 389, 64, 128, 2)
    var manager := TrainingManager.new(evo, 3, 50)

    # Enable tournament mode
    manager.use_tournament = true
    manager.tournament_mode = "round_robin"
    manager.tournament_opponents = 4

    # Generate pairings
    var pairings = manager.generate_round_robin_pairings(10)

    # Verify structure
    assert_equal(pairings.size(), 10, "Should have pairings for all individuals")

    # Verify each individual has correct number of opponents
    for i in range(10):
        assert_equal(pairings[i].size(), 4, "Each individual should have 4 opponents")

        # Verify no self-pairing
        assert_false(i in pairings[i], "Individual should not play against itself")

        # Verify no duplicates
        var unique_opponents = {}
        for opp in pairings[i]:
            assert_false(opp in unique_opponents, "No duplicate opponents")
            unique_opponents[opp] = true

func test_tournament_scoring() -> void:
    ## Test tournament score calculation
    var evo := ChessEvolution.new(4, 389, 64, 128, 1)
    var manager := TrainingManager.new(evo, 3, 50)

    # Enable tournament mode
    manager.use_tournament = true

    # Simulate some results (dict format with material_advantage for draws)
    manager.tournament_results = {
        "0_white": {"result": 1},    # Win
        "1_white": {"result": 0, "material_advantage": 0.0},    # Draw (equal material)
        "2_white": {"result": -1},   # Loss
        "3_white": {"result": 1},    # Win
        "0_black": {"result": 0, "material_advantage": 0.0},    # Draw (equal material)
        "1_black": {"result": 1},    # Win
        "2_black": {"result": -1},   # Loss
        "3_black": {"result": 0, "material_advantage": 0.0},    # Draw (equal material)
    }

    # Calculate scores
    var scores = manager.calculate_tournament_scores()

    # Verify scores (wins = 1.0, draws with equal material = 0.5, losses = 0.0)
    assert_almost_equal(scores[0], 1.0, 0.01, "Player 0 should have 1.0 points (1 win)")
    assert_almost_equal(scores[1], 0.5, 0.01, "Player 1 should have 0.5 points (1 draw, equal material)")
    assert_almost_equal(scores[2], 0.0, 0.01, "Player 2 should have 0.0 points (1 loss)")
    assert_almost_equal(scores[3], 1.0, 0.01, "Player 3 should have 1.0 points (1 win)")

func test_swiss_pairings_round1() -> void:
    ## Test Swiss system first round (should be random pairings)
    var evo := ChessEvolution.new(8, 389, 64, 128, 2)
    var manager := TrainingManager.new(evo, 3, 50)

    # Enable Swiss tournament
    manager.use_tournament = true
    manager.tournament_mode = "swiss"

    # Generate first round pairings
    var pairings = manager.generate_swiss_pairings(8, 0)

    # Verify everyone is paired exactly once
    var paired_count = {}
    for i in pairings:
        for opp in pairings[i]:
            paired_count[i] = paired_count.get(i, 0) + 1

    # Each player should have exactly 1 opponent in round 1
    for i in range(8):
        assert_equal(paired_count.get(i, 0), 1, "Each player should be paired once")

func test_material_weighted_draw_scoring() -> void:
    ## Test that material advantage affects draw scoring
    var evo := ChessEvolution.new(4, 389, 64, 128, 1)
    var manager := TrainingManager.new(evo, 3, 50)
    manager.use_tournament = true

    # Player 0: draw with +5 material advantage -> should score > 0.5
    # Player 1: draw with -5 material advantage -> should score < 0.5
    # Player 2: draw with 0 material -> should score exactly 0.5
    manager.tournament_results = {
        "0_white": {"result": 0, "material_advantage": 5.0},
        "1_white": {"result": 0, "material_advantage": -5.0},
        "2_white": {"result": 0, "material_advantage": 0.0},
        "3_white": {"result": 1},
    }

    var scores = manager.calculate_tournament_scores()

    # +5 material: 0.5 + clamp(5*0.02, -0.2, 0.2) = 0.5 + 0.1 = 0.6
    assert_almost_equal(scores[0], 0.6, 0.01, "Positive material draw should score > 0.5")
    # -5 material: 0.5 + clamp(-5*0.02, -0.2, 0.2) = 0.5 - 0.1 = 0.4
    assert_almost_equal(scores[1], 0.4, 0.01, "Negative material draw should score < 0.5")
    # 0 material: 0.5 + clamp(0, -0.2, 0.2) = 0.5
    assert_almost_equal(scores[2], 0.5, 0.01, "Equal material draw should score 0.5")
    # Win: 1.0
    assert_almost_equal(scores[3], 1.0, 0.01, "Win should still score 1.0")

func test_fitness_update_from_tournament() -> void:
    ## Test that tournament scores properly update fitness
    var evo := ChessEvolution.new(4, 389, 64, 128, 1)
    var manager := TrainingManager.new(evo, 3, 50)

    # Enable tournament mode
    manager.use_tournament = true

    # Set some accumulated fitness (for bonus calculation)
    evo.white_fitness[0] = 5.0
    evo.white_fitness[1] = 3.0
    evo.white_fitness[2] = 2.0
    evo.white_fitness[3] = 4.0

    # Set tournament results (dict format)
    manager.tournament_results = {
        "0_white": {"result": 1},    # Win = 1.0
        "1_white": {"result": 0, "material_advantage": 0.0},    # Draw = 0.5
        "2_white": {"result": -1},   # Loss = 0.0
        "3_white": {"result": 1},    # Win = 1.0
    }

    # Update fitness from tournament
    manager.update_fitness_from_tournament()

    # Verify fitness = tournament_score + 10% of accumulated fitness
    assert_almost_equal(evo.white_fitness[0], 1.0 + 0.5, 0.01, "Player 0: 1.0 + 10% of 5.0")
    assert_almost_equal(evo.white_fitness[1], 0.5 + 0.3, 0.01, "Player 1: 0.5 + 10% of 3.0")
    assert_almost_equal(evo.white_fitness[2], 0.0 + 0.2, 0.01, "Player 2: 0.0 + 10% of 2.0")
    assert_almost_equal(evo.white_fitness[3], 1.0 + 0.4, 0.01, "Player 3: 1.0 + 10% of 4.0")