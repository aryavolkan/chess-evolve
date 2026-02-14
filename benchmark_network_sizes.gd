extends SceneTree

## Benchmark different network sizes to find speed/accuracy tradeoff

func _init() -> void:
	var sizes := [32, 48, 64, 80, 96, 128]
	var iterations := 10000
	
	print("Neural Network Size Benchmark")
	print("Architecture: 389 inputs → [hidden] → 128 outputs")
	print("Testing %d iterations per size\n" % iterations)
	
	for hidden_size in sizes:
		var nn = load("res://ai/neural_network.gd").new(389, hidden_size, 128)
		var inputs := PackedFloat32Array()
		inputs.resize(389)
		for i in inputs.size():
			inputs[i] = randf_range(-1.0, 1.0)
		
		var start_time := Time.get_ticks_usec()
		
		for i in iterations:
			var _output: PackedFloat32Array = nn.forward(inputs)
		
		var end_time := Time.get_ticks_usec()
		var elapsed_ms := (end_time - start_time) / 1000.0
		var avg_us := (end_time - start_time) / float(iterations)
		var fps := iterations / (elapsed_ms / 1000.0)
		var weight_count: int = nn.get_weight_count()
		
		print("Hidden size: %d" % hidden_size)
		print("  Weights: %d" % weight_count)
		print("  Avg: %.2f µs/forward" % avg_us)
		print("  Throughput: %.0f forwards/sec" % fps)
		print()
	
	quit()
