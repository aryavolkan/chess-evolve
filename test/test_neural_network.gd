extends "res://test/test_base.gd"

var NNScript = preload("res://ai/neural_network.gd")


func _run_tests() -> void:
    _test("network creates with correct sizes", func():
        var nn = NNScript.new(10, 5, 3)
        assert_eq(nn.input_size, 10)
        assert_eq(nn.hidden_size, 5)
        assert_eq(nn.output_size, 3)
    )

    _test("forward produces correct output size", func():
        var nn = NNScript.new(10, 5, 3)
        var inputs := PackedFloat32Array()
        inputs.resize(10)
        inputs.fill(0.5)
        var out := nn.forward(inputs)
        assert_eq(out.size(), 3)
    )

    _test("outputs are in tanh range", func():
        var nn = NNScript.new(10, 5, 3)
        var inputs := PackedFloat32Array()
        inputs.resize(10)
        for i in 10: inputs[i] = randf_range(-1.0, 1.0)
        var out := nn.forward(inputs)
        for v in out:
            assert_true(v >= -1.0 and v <= 1.0, "Output %f out of range" % v)
    )

    _test("clone produces independent copy", func():
        var nn = NNScript.new(10, 5, 3)
        var clone = nn.clone()
        clone.mutate(1.0, 1.0)  # Mutate everything
        var w1 := nn.get_weights()
        var w2: PackedFloat32Array = clone.get_weights()
        var diffs := 0
        for i in w1.size():
            if w1[i] != w2[i]: diffs += 1
        assert_gt(diffs, 0, "Clone should be independent after mutation")
    )

    _test("mutation changes weights", func():
        var nn = NNScript.new(10, 5, 3)
        var before := nn.get_weights().duplicate()
        nn.mutate(1.0, 0.5)
        var after := nn.get_weights()
        var changed := 0
        for i in before.size():
            if before[i] != after[i]: changed += 1
        assert_gt(changed, 0, "Mutation should change weights")
    )

    _test("crossover produces child", func():
        var a = NNScript.new(10, 5, 3)
        var b = NNScript.new(10, 5, 3)
        var child = a.crossover_with(b)
        assert_eq(child.input_size, 10)
        assert_eq(child.get_weight_count(), a.get_weight_count())
    )

    _test("weight count matches expected", func():
        var nn = NNScript.new(389, 64, 4096)
        # ih: 389*64 = 24896, bh: 64, ho: 64*4096 = 262144, bo: 4096
        var expected := 389 * 64 + 64 + 64 * 4096 + 4096
        assert_eq(nn.get_weight_count(), expected)
    )

    _test("set_weights restores exact values", func():
        var nn = NNScript.new(10, 5, 3)
        var weights := nn.get_weights().duplicate()
        var nn2 = NNScript.new(10, 5, 3)
        nn2.set_weights(weights)
        var w2 := nn2.get_weights()
        for i in weights.size():
            assert_true(weights[i] == w2[i], "Weight mismatch at %d" % i)
    )
