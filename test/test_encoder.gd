extends "res://test/test_base.gd"

const BoardStateScript = preload("res://chess/board_state.gd")
const ChessEncoderScript = preload("res://chess/encoder.gd")
const INPUT_SIZE := 389
const MOVE_OUTPUT_SIZE := 128


func _run_tests() -> void:
    var encoder = ChessEncoderScript.new()
    _test("encode produces correct size", func():
        var b := BoardStateScript.new()
        b.setup_initial()
        var encoded: PackedFloat32Array = encoder.call("encode_board", b)
        assert_eq(encoded.size(), INPUT_SIZE)
    )

    _test("encoding has non-zero values", func():
        var b := BoardStateScript.new()
        b.setup_initial()
        var encoded: PackedFloat32Array = encoder.call("encode_board", b)
        var non_zero := 0
        for v in encoded:
            if v != 0.0: non_zero += 1
        assert_gt(non_zero, 30, "Should have many non-zero inputs")
    )

    _test("decode selects legal move", func():
        var b := BoardStateScript.new()
        b.setup_initial()
        var legal := b.generate_legal_moves()
        # Random output
        var outputs := PackedFloat32Array()
        outputs.resize(MOVE_OUTPUT_SIZE)
        for i in outputs.size():
            outputs[i] = randf_range(-1.0, 1.0)
        var move: int = encoder.call("decode_move", outputs, legal)
        assert_true(legal.has(move), "Decoded move should be legal")
    )

    _test("decode with empty moves returns -1", func():
        var outputs := PackedFloat32Array()
        outputs.resize(128)
        var empty := PackedInt32Array()
        var move: int = encoder.call("decode_move", outputs, empty)
        assert_eq(move, -1)
    )

    _test("encoding differs for different positions", func():
        var b1 := BoardStateScript.new()
        b1.setup_initial()
        var b2 := BoardStateScript.new()
        b2.setup_initial()
        b2.make_move(BoardStateScript.encode_move(12, 28))  # e2-e4
        var e1: PackedFloat32Array = encoder.call("encode_board", b1)
        var e2: PackedFloat32Array = encoder.call("encode_board", b2)
        var diffs := 0
        for i in e1.size():
            if e1[i] != e2[i]: diffs += 1
        assert_gt(diffs, 0, "Different positions should encode differently")
    )
