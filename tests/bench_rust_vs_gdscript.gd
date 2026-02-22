extends SceneTree

const BoardStateScript = preload("res://chess/board_state.gd")


func _init() -> void:
    var iterations := 1000

    var gd_state := BoardStateScript.new()
    gd_state.setup_initial()

    var gd_start := Time.get_ticks_usec()
    for i in iterations:
        gd_state.generate_legal_moves()
    var gd_elapsed_ms := (Time.get_ticks_usec() - gd_start) / 1000.0

    if ClassDB.class_exists(&"RustChessBoard"):
        var rust_board := RustChessBoard.new()
        var rust_start := Time.get_ticks_usec()
        for i in iterations:
            rust_board.get_legal_moves()
        var rust_elapsed_ms := (Time.get_ticks_usec() - rust_start) / 1000.0
        var speedup := gd_elapsed_ms / max(rust_elapsed_ms, 0.0001)
        print("GDScript: %.2fms, Rust: %.2fms, speedup: %.2fx" % [gd_elapsed_ms, rust_elapsed_ms, speedup])
    else:
        print("GDScript: %.2fms, Rust: (unavailable), speedup: 0.00x" % gd_elapsed_ms)

    quit()
