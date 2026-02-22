extends SceneTree

## Quick benchmark for neural network forward pass performance

func _init() -> void:
    var nn = load("res://ai/neural_network.gd").new(389, 64, 128)
    var inputs := PackedFloat32Array()
    inputs.resize(389)
    for i in inputs.size():
        inputs[i] = randf_range(-1.0, 1.0)

    var iterations := 10000
    var start_time := Time.get_ticks_usec()

    for i in iterations:
        nn.forward(inputs)

    var end_time := Time.get_ticks_usec()
    var elapsed_ms := (end_time - start_time) / 1000.0
    var avg_us := (end_time - start_time) / float(iterations)

    print("Benchmark Results:")
    print("  Total iterations: %d" % iterations)
    print("  Total time: %.2f ms" % elapsed_ms)
    print("  Avg per forward pass: %.2f µs" % avg_us)
    print("  Forward passes/sec: %.0f" % (iterations / (elapsed_ms / 1000.0)))

    quit()
