extends "res://test/test_base.gd"

const BoardStateScript = preload("res://chess/board_state.gd")
const PIECE = ChessConstants.Piece


func _run_tests() -> void:
    _test("initial position has 32 pieces", func():
        var b := BoardStateScript.new()
        b.setup_initial()
        var count := 0
        for sq in range(64):
            if b.board[sq] != 0: count += 1
        assert_eq(count, 32)
    )

    _test("initial position white to move", func():
        var b := BoardStateScript.new()
        b.setup_initial()
        assert_eq(b.side_to_move, 0)
    )

    _test("white has 20 opening moves", func():
        var b := BoardStateScript.new()
        b.setup_initial()
        var moves := b.generate_legal_moves()
        assert_eq(moves.size(), 20, "Expected 20 moves, got %d" % moves.size())
    )

    _test("pawn can move forward", func():
        var b := BoardStateScript.new()
        b.setup_initial()
        var moves := b.generate_legal_moves()
        # e2-e4 should be legal (square 12 to 28)
        var e2e4 := BoardStateScript.encode_move(12, 28)
        assert_true(moves.has(e2e4), "e2-e4 should be legal")
    )

    _test("make move changes side", func():
        var b := BoardStateScript.new()
        b.setup_initial()
        b.make_move(BoardStateScript.encode_move(12, 28))  # e2-e4
        assert_eq(b.side_to_move, 1, "Should be black's turn")
    )

    _test("king not in check initially", func():
        var b := BoardStateScript.new()
        b.setup_initial()
        assert_false(b.is_in_check())
    )

    _test("material score correct at start", func():
        var b := BoardStateScript.new()
        b.setup_initial()
        var white_mat := b.material_score(0)
        var black_mat := b.material_score(1)
        assert_eq(white_mat, black_mat, "Material should be equal")
        # 8 pawns(8) + 2 knights(6) + 2 bishops(6.5) + 2 rooks(10) + 1 queen(9) = 39.5
        assert_gt(white_mat, 39.0)
    )

    _test("clone produces independent copy", func():
        var b := BoardStateScript.new()
        b.setup_initial()
        var c := b.clone()
        c.make_move(BoardStateScript.encode_move(12, 28))
        assert_eq(b.side_to_move, 0, "Original should be unchanged")
        assert_eq(c.side_to_move, 1, "Clone should have moved")
    )

    _test("to_string_board works", func():
        var b := BoardStateScript.new()
        b.setup_initial()
        var s := b.to_string_board()
        assert_true(s.length() > 0)
        assert_true(s.contains("K"))
    )

    _test("en passant square set on double pawn push", func():
        var b := BoardStateScript.new()
        b.setup_initial()
        b.make_move(BoardStateScript.encode_move(12, 28))  # e2-e4
        assert_eq(b.en_passant_square, 20, "EP square should be e3 (20)")
    )

    _test("en passant capture is legal", func():
        var b := BoardStateScript.new()
        b.setup_initial()
        b.make_move(BoardStateScript.encode_move(12, 28))  # e2-e4
        b.make_move(BoardStateScript.encode_move(48, 40))  # a7-a6
        b.make_move(BoardStateScript.encode_move(28, 36))  # e4-e5
        b.make_move(BoardStateScript.encode_move(51, 35))  # d7-d5
        assert_eq(b.en_passant_square, 43, "EP square should be d6 (43)")
        var moves := b.generate_legal_moves()
        var ep := BoardStateScript.encode_move(36, 43)  # e5xd6 en passant
        assert_true(moves.has(ep), "En passant capture should be legal")
    )

    _test("fool's mate produces checkmate", func():
        var b := BoardStateScript.new()
        b.setup_initial()
        b.make_move(BoardStateScript.encode_move(13, 21))  # f2-f3
        b.make_move(BoardStateScript.encode_move(52, 36))  # e7-e5
        b.make_move(BoardStateScript.encode_move(14, 30))  # g2-g4
        b.make_move(BoardStateScript.encode_move(59, 31))  # Qd8-h4
        assert_true(b.is_game_over, "Game should be over (checkmate)")
        assert_eq(b.result, -1, "Black should win by checkmate")
    )

    _test("stalemate detected", func():
        # Set up a known stalemate: white king a1, white queen b3, black king a8... (simplified)
        # For now just test that game_over works with no legal moves
        var b := BoardStateScript.new()
        b.board.fill(0)
        b.board[0] = PIECE.KING  # White king a1
        b.board[58] = PIECE.QUEEN  # White queen c8... actually let's do:
        # Black king h8 (63), white king f6 (45), white queen g6 (46)
        b.board.fill(0)
        b.board[63] = -PIECE.KING  # Black king h8
        b.board[45] = PIECE.KING   # White king f6
        b.board[46] = PIECE.QUEEN  # White queen g6
        b.rebuild_piece_lists()
        b.side_to_move = 1  # Black to move
        var moves := b.generate_legal_moves()
        # Black king is on h8, queen on g6 controls g7,g8,h7. King on f6 controls g7,e7,f7
        # h8 king can go to g8(controlled by Qg6), h7(controlled by Qg6), g7(controlled by both)
        # This is stalemate if not in check
        # Is black in check? Qg6 doesn't attack h8 directly. Kf6 doesn't attack h8.
        # So it's stalemate!
        assert_eq(moves.size(), 0, "Should be stalemate (0 legal moves)")
    )

    _test("50-move rule draw", func():
        var b := BoardStateScript.new()
        b.setup_initial()
        b.halfmove_clock = 100
        b.make_move(BoardStateScript.encode_move(1, 18))  # Knight move (no pawn/capture)
        assert_true(b.is_game_over, "Should be game over after 50 moves")
        assert_eq(b.result, 2, "Should be draw")
    )

    _test("castling kingside white", func():
        var b := BoardStateScript.new()
        b.board.fill(0)
        b.board[4] = PIECE.KING
        b.board[7] = PIECE.ROOK
        b.board[60] = -PIECE.KING
        b.rebuild_piece_lists()
        b.castling_rights = 0b0001  # White kingside only
        b.side_to_move = 0
        var moves := b.generate_legal_moves()
        var castle := BoardStateScript.encode_move(4, 6)
        assert_true(moves.has(castle), "Kingside castling should be legal")
        b.make_move(castle)
        assert_eq(b.board[6], PIECE.KING, "King should be on g1")
        assert_eq(b.board[5], PIECE.ROOK, "Rook should be on f1")
    )

    _test("profile generate_legal_moves", func():
        var b := BoardStateScript.new()
        b.setup_initial()
        var iterations := 1000
        var start := Time.get_ticks_usec()
        for i in iterations:
            b.generate_legal_moves()
        var total := Time.get_ticks_usec() - start
        var avg := float(total) / float(iterations)
        print("generate_legal_moves %d %d usec total, %s usec/call" % [
            iterations, total, avg])

        var types := [PIECE.PAWN, PIECE.KNIGHT, PIECE.BISHOP, PIECE.ROOK, PIECE.QUEEN, PIECE.KING]
        var type_names := {
            PIECE.PAWN: "pawn",
            PIECE.KNIGHT: "knight",
            PIECE.BISHOP: "bishop",
            PIECE.ROOK: "rook",
            PIECE.QUEEN: "queen",
            PIECE.KING: "king"
        }
        var squares_by_type := {}
        for t in types:
            squares_by_type[t] = []
        for sq in range(64):
            var p := b.board[sq]
            if p > 0:
                var t := absi(p)
                if squares_by_type.has(t):
                    squares_by_type[t].append(sq)

        var slowest_name := ""
        var slowest_time := -1.0
        for t in types:
            var moves_buf := PackedInt32Array()
            var per_start := Time.get_ticks_usec()
            for i in iterations:
                moves_buf.clear()
                for sq in squares_by_type[t]:
                    b.add_piece_moves(sq, moves_buf)
            var per_total := Time.get_ticks_usec() - per_start
            var per_avg := float(per_total) / float(iterations)
            print("piece moves %s %d usec total, %s usec/call" % [
                type_names[t], per_total, per_avg])
            if per_avg > slowest_time:
                slowest_time = per_avg
                slowest_name = type_names[t]
        print("slowest piece type: ", slowest_name)
    )
