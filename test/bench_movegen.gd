extends SceneTree

const BoardStateScript = preload("res://chess/board_state.gd")
const BitboardStateScript = preload("res://chess/bitboard_state.gd")


func _init() -> void:
	var iterations := 1000

	var board := BoardStateScript.new()
	board.use_bitboard_movegen = false
	board.setup_initial()
	_bench_board_state(board, iterations)

	var bb := BitboardStateScript.from_board_state(board)
	_bench_bitboard_state(bb, iterations)

	quit()


func _bench_board_state(state: BoardStateScript, iterations: int) -> void:
	var start := Time.get_ticks_usec()
	for i in iterations:
		state.generate_legal_moves()
	var elapsed := Time.get_ticks_usec() - start
	print("BoardState generate_legal_moves:", float(elapsed) / float(iterations), " usec/call (", elapsed, " usec total)")


func _bench_bitboard_state(state: BitboardStateScript, iterations: int) -> void:
	var start := Time.get_ticks_usec()
	for i in iterations:
		state.get_legal_moves()
	var elapsed := Time.get_ticks_usec() - start
	print("BitboardState get_legal_moves:", float(elapsed) / float(iterations), " usec/call (", elapsed, " usec total)")
