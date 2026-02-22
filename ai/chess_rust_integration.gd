class_name ChessRustIntegration
extends RefCounted

static var _rust_board = null
static var _checked := false

static func _check_available() -> void:
    if _checked:
        return
    _checked = true
    if ClassDB.class_exists(&"RustChessBoard"):
        _rust_board = RustChessBoard.new()
        print("[Chess] ✓ Using Rust board (fast bitboards)")
    else:
        print("[Chess] ⚠ Rust extension not loaded, using GDScript")

static func is_rust_available() -> bool:
    _check_available()
    return _rust_board != null
