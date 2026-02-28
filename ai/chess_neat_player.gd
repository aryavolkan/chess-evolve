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


func select_move(board: BoardStateScript, temperature: float = 0.0) -> int:
    var legal_moves := board.generate_legal_moves()
    if legal_moves.is_empty():
        return -1

    var inputs: PackedFloat32Array = ChessEncoderScript.encode_board(board)
    var outputs: PackedFloat32Array = network.forward(inputs)
    return ChessEncoderScript.decode_move(outputs, legal_moves, temperature)
