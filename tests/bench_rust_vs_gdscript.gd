extends SceneTree

const BoardStateScript = preload("res://chess/board_state.gd")


func _init() -> void:
	var iterations := 1000

	var gd_state := BoardStateScript.new()
	gd_state.setup_initial()

	var gd_start := Time.get_ticks_usec()
	for i in iterations:
		gd_state.generate_legal_moves()
	var gd_end := Time.get_ticks_usec()
	var gd_elapsed_s := (gd_end - gd_start) / 1_000_000.0
	var gd_mps := iterations / max(gd_elapsed_s, 0.000001)

	print("GDScript BoardState: %.0f moves/sec" % gd_mps)

	if ClassDB.class_exists(&"RustChessBoard"):
		var rust_board := RustChessBoard.new()
		var rust_start := Time.get_ticks_usec()
		for i in iterations:
			rust_board.get_legal_moves()
		var rust_end := Time.get_ticks_usec()
		var rust_elapsed_s := (rust_end - rust_start) / 1_000_000.0
		var rust_mps := iterations / max(rust_elapsed_s, 0.000001)
		print("RustChessBoard: %.0f moves/sec" % rust_mps)
		print("Speedup: %.2fx" % (rust_mps / max(gd_mps, 0.000001)))
	else:
		print("RustChessBoard not available; build the extension to benchmark")

	quit()
