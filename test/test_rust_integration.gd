extends "res://test/test_base.gd"

const BoardStateScript = preload("res://chess/board_state.gd")
const NeuralNetworkScript = preload("res://ai/neural_network.gd")
const ChessEncoderScript = preload("res://chess/encoder.gd")


func _run_tests() -> void:
    _test("rust availability check does not crash", func():
        var available := ChessRustIntegration.is_rust_available()
        assert_eq(typeof(available), TYPE_BOOL)
    )

    _test("gds fallback board has 20 legal moves at start", func():
        var state := BoardStateScript.new()
        state.setup_initial()
        assert_eq(state.generate_legal_moves().size(), 20)
    )

    _test("rust legal move count matches GDScript", func():
        if ChessRustIntegration.is_rust_available():
            var state := BoardStateScript.new()
            state.setup_initial()
            var rust_board := RustChessBoard.new()
            var rust_moves := rust_board.get_legal_moves()
            var gd_moves := state.generate_legal_moves()
            assert_eq(rust_moves.size(), gd_moves.size())
        else:
            assert_true(true)
    )

    _test("rust make_move flips side_to_move", func():
        if ChessRustIntegration.is_rust_available():
            var rust_board := RustChessBoard.new()
            var start_side := rust_board.side_to_move()
            var moves := rust_board.get_legal_moves()
            assert_gt(moves.size(), 0)
            rust_board.make_move(moves[0])
            assert_ne(rust_board.side_to_move(), start_side)
        else:
            assert_true(true)
    )

    _test("rust dense network forward returns correct output size", func():
        if ClassDB.class_exists(&"RustDenseNetwork"):
            var net := RustDenseNetwork.new()
            net.setup(4, 8, 218)
            var inputs := PackedFloat32Array()
            inputs.resize(4)
            inputs.fill(0.0)
            var out := net.forward(inputs)
            assert_eq(out.size(), 218)
        else:
            assert_true(true)
    )

    _test("rust dense network zero weights returns zeros", func():
        if ClassDB.class_exists(&"RustDenseNetwork"):
            var net := RustDenseNetwork.new()
            net.setup(4, 8, 218)
            var weights_ih := PackedFloat32Array()
            weights_ih.resize(4 * 8)
            weights_ih.fill(0.0)
            var biases_h := PackedFloat32Array()
            biases_h.resize(8)
            biases_h.fill(0.0)
            var weights_ho := PackedFloat32Array()
            weights_ho.resize(8 * 218)
            weights_ho.fill(0.0)
            var biases_o := PackedFloat32Array()
            biases_o.resize(218)
            biases_o.fill(0.0)
            net.set_weights(weights_ih, biases_h, weights_ho, biases_o)
            var inputs := PackedFloat32Array()
            inputs.resize(4)
            inputs.fill(0.0)
            var out := net.forward(inputs)
            for v in out:
                assert_true(absf(v) < 0.0001)
        else:
            assert_true(true)
    )

    _test("rust batch simulator matches gdscript game", func():
        if ClassDB.class_exists(&"RustBatchSimulator"):
            var input_size := ChessEncoderScript.INPUT_SIZE
            var hidden_size := 8
            var output_size := ChessEncoderScript.MOVE_OUTPUT_SIZE
            var net := NeuralNetworkScript.new(input_size, hidden_size, output_size, false)
            net.weights_ih.fill(0.0)
            net.bias_h.fill(0.0)
            net.weights_ho.fill(0.0)
            net.bias_o.fill(0.0)
            net.bias_o[12] = 2.0
            net.bias_o[64 + 28] = 2.0
            net.bias_o[52] = 2.0
            net.bias_o[64 + 36] = 2.0
            var weights := net.get_weights()
            var rust_sim := RustBatchSimulator.new()
            var rust_stats: Dictionary = rust_sim.simulate_game(
                weights,
                weights,
                input_size,
                hidden_size,
                output_size,
                2
            )
            var gd_stats := _simulate_gd(weights, input_size, hidden_size, output_size, 2)
            assert_eq(rust_stats.get("result", 999), gd_stats.get("result", -999))
            assert_eq(rust_stats.get("move_count", -1), gd_stats.get("move_count", -2))
            var float_keys := [
                "white_material",
                "black_material",
                "white_mobility",
                "black_mobility",
                "white_king_safety",
                "black_king_safety",
            ]
            for key in float_keys:
                var a: float = float(rust_stats.get(key, -999.0))
                var b: float = float(gd_stats.get(key, 999.0))
                assert_true(absf(a - b) < 0.0001, "%s mismatch: %s vs %s" % [key, a, b])
        else:
            assert_true(true)
    )

    _test("rust availability check stays safe on repeat", func():
        var available := ChessRustIntegration.is_rust_available()
        assert_eq(typeof(available), TYPE_BOOL)
    )

func _simulate_gd(
        weights: PackedFloat32Array,
        input_size: int,
        hidden_size: int,
        output_size: int,
        max_moves: int
    ) -> Dictionary:
    var net := NeuralNetworkScript.new(input_size, hidden_size, output_size, false)
    net.set_weights(weights)
    var state := BoardStateScript.new()
    state.setup_initial()
    var move_count := 0
    var result := 2
    while move_count < max_moves:
        var legal_moves := state.generate_legal_moves()
        if legal_moves.is_empty():
            if state.is_in_check():
                result = 1 if state.side_to_move == 1 else -1
            else:
                result = 2
            break
        var inputs := ChessEncoderScript.encode_board(state)
        var outputs := net.forward(inputs)
        var chosen := ChessEncoderScript.decode_move(outputs, legal_moves)
        if chosen == -1:
            break
        state.make_move(chosen)
        move_count += 1
        if state.halfmove_clock >= 100:
            result = 2
            break
    return {
        "result": result,
        "move_count": move_count,
        "white_material": state.material_score(0),
        "black_material": state.material_score(1),
        "white_mobility": _mobility_from_state(state, 0),
        "black_mobility": _mobility_from_state(state, 1),
        "white_king_safety": state.king_safety_score(0),
        "black_king_safety": state.king_safety_score(1),
    }

func _mobility_from_state(state: BoardState, color: int) -> int:
    var copy := state.clone()
    copy.side_to_move = color
    return copy.generate_legal_moves().size()
