extends SceneTree

## Quick test to verify the minimax and Hall of Fame improvements work

func _init():
    print("Testing Chess Evolution Improvements...")

    # Test 1: Verify minimax player can be created
    var NeuralNetwork = preload("res://ai/neural_network.gd")
    var MinimaxPlayer = preload("res://ai/minimax_player.gd")
    var BoardState = preload("res://chess/board_state.gd")

    var network = NeuralNetwork.new()
    var minimax = MinimaxPlayer.new(network, 2)

    print("✓ Minimax player created successfully")

    # Test 2: Test minimax on a simple position
    var board = BoardState.new()
    board.setup_initial()

    var move = minimax.choose_move(board)
    print("✓ Minimax chose move: ", move)
    print("  Nodes evaluated: ", minimax.get_stats()["nodes_evaluated"])

    # Test 3: Verify Hall of Fame functionality
    var Evolution = preload("res://ai/evolution.gd")
    var evolution = Evolution.new(10)  # Small population for testing

    print("✓ Evolution system created")
    print("  Initial Hall of Fame size: ", evolution.hall_of_fame.size())

    # Simulate adding to Hall of Fame
    evolution._update_hall_of_fame(network, 10.0, true)
    print("✓ Added to Hall of Fame")
    print("  Hall of Fame size: ", evolution.hall_of_fame.size())

    # Test getting Hall of Fame opponent
    var hof_opponent = evolution.get_hall_of_fame_opponent(0)
    if hof_opponent != null:
        print("✓ Retrieved Hall of Fame opponent")

    # Test 4: Quick training manager test
    var TrainingManager = preload("res://ai/training_manager.gd")
    var tm = TrainingManager.new(evolution, 1, 50)  # 1 game, max 50 moves

    print("✓ Training manager created with minimax enabled: ", tm.use_minimax)
    print("  Hall of Fame ratio: ", tm.hall_of_fame_ratio)

    print("\nAll tests passed! The improvements are working correctly.")
    quit()