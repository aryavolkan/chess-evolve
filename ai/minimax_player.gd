class_name MinimaxPlayer
extends RefCounted

## Minimax player with alpha-beta pruning that uses a neural network as the
## position evaluation function. Instead of the network directly picking moves,
## we search the game tree and use the network to evaluate leaf positions.
##
## Design decision: Since the current network architecture outputs 128 values
## (policy network for move preferences), we'll adapt it to work as a value
## network by summing the outputs to get a single position score. This avoids
## breaking the existing network structure while transitioning to search-based play.

const BoardStateScript = preload("res://chess/board_state.gd")
const ChessEncoderScript = preload("res://chess/encoder.gd")

var neural_network  # The neural network to evaluate positions
var search_depth: int = 2  # Configurable search depth (2-3 recommended)
var nodes_evaluated: int = 0  # Statistics counter

func _init(network, depth: int = 2):
    neural_network = network
    search_depth = depth

## Choose the best move using minimax with alpha-beta pruning
func choose_move(state: BoardStateScript) -> int:
    nodes_evaluated = 0
    var legal_moves := state.generate_legal_moves()

    if legal_moves.is_empty():
        return -1

    if legal_moves.size() == 1:
        return legal_moves[0]

    var best_move := legal_moves[0]
    var best_score := -INF
    var alpha := -INF
    var beta := INF

    # For each legal move, search the game tree
    for move in legal_moves:
        var state_copy := state.clone()
        state_copy.make_move(move)

        # Minimax search (negate because we're switching sides)
        var score := -_minimax(state_copy, search_depth - 1, -beta, -alpha)

        if score > best_score:
            best_score = score
            best_move = move

        alpha = maxf(alpha, score)
        if alpha >= beta:
            break  # Beta cutoff

    return best_move

## Minimax search with alpha-beta pruning
func _minimax(state: BoardStateScript, depth: int, alpha: float, beta: float) -> float:
    nodes_evaluated += 1

    # Terminal node: game over or max depth reached
    if state.is_game_over or depth <= 0:
        return _evaluate_position(state)

    var legal_moves := state.generate_legal_moves()
    if legal_moves.is_empty():
        # Stalemate or checkmate
        return _evaluate_position(state)

    var best_score := -INF

    for move in legal_moves:
        var state_copy := state.clone()
        state_copy.make_move(move)

        # Recursive minimax (negate for opponent's perspective)
        var score := -_minimax(state_copy, depth - 1, -beta, -alpha)
        best_score = maxf(best_score, score)

        alpha = maxf(alpha, score)
        if alpha >= beta:
            break  # Beta cutoff

    return best_score

## Evaluate a position using the neural network
func _evaluate_position(state: BoardStateScript) -> float:
    # Handle terminal states
    if state.is_game_over:
        if state.result == 2:  # Draw
            return 0.0
        elif state.result == 1:  # White wins
            return 100.0 if state.side_to_move == 1 else -100.0
        else:  # Black wins
            return -100.0 if state.side_to_move == 1 else 100.0

    # Use neural network for position evaluation
    var inputs := ChessEncoderScript.encode_board(state)
    var outputs: PackedFloat32Array = neural_network.forward(inputs)

    # Convert policy network output to position score
    # Since we have 128 outputs (from/to move preferences), we'll interpret
    # the sum of outputs as a position score. The network learns to output
    # higher values for better positions.
    var score := 0.0
    for i in outputs.size():
        score += outputs[i]

    # Normalize to reasonable range and adjust for side to move
    # Network outputs are tanh (-1 to 1), so sum is roughly -128 to 128
    score = score / 10.0  # Scale to roughly -12.8 to 12.8

    # Flip score if it's black's turn (since network sees from current player's view)
    if state.side_to_move == 1:
        score = -score

    return score

## Get search statistics
func get_stats() -> Dictionary:
    return {
        "depth": search_depth,
        "nodes_evaluated": nodes_evaluated
    }