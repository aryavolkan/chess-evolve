class_name ChessNeatPlayer
extends RefCounted

## Wraps a NEAT genome to select legal chess moves.

const BoardStateScript = preload("res://chess/board_state.gd")
const ChessEncoderScript = preload("res://chess/encoder.gd")
const NeatNetworkScript = preload("res://ai/neat_network.gd")

var genome: NeatGenome
var network: NeatNetwork


func _init(p_genome: NeatGenome) -> void:
	genome = p_genome
	network = NeatNetworkScript.from_genome(genome)


func refresh_network() -> void:
	network = NeatNetworkScript.from_genome(genome)


func select_move(board: BoardStateScript) -> int:
	var legal_moves := board.generate_legal_moves()
	if legal_moves.is_empty():
		return -1

	var inputs: PackedFloat32Array = ChessEncoderScript.encode_board(board)
	var outputs: PackedFloat32Array = network.forward(inputs)
	var count: int = mini(legal_moves.size(), outputs.size())
	if count <= 0:
		return legal_moves[0]

	var max_logit := -INF
	for i in count:
		max_logit = maxf(max_logit, outputs[i])

	var best_idx := 0
	var best_prob := -INF
	var exp_sum := 0.0
	var exp_vals := PackedFloat32Array()
	exp_vals.resize(count)
	for i in count:
		var exp_val := exp(outputs[i] - max_logit)
		exp_vals[i] = exp_val
		exp_sum += exp_val

	for i in count:
		var prob := exp_vals[i] / exp_sum if exp_sum > 0.0 else 0.0
		if prob > best_prob:
			best_prob = prob
			best_idx = i

	return legal_moves[best_idx]
