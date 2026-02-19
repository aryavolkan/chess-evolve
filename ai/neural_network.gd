extends RefCounted

## Feedforward neural network for chess move selection.
## Architecture: inputs -> hidden (tanh) -> outputs (tanh)
## Adapted from Evolve's neural_network.gd.

var input_size: int
var hidden_size: int
var output_size: int

var weights_ih: PackedFloat32Array
var bias_h: PackedFloat32Array
var weights_ho: PackedFloat32Array
var bias_o: PackedFloat32Array

var _hidden: PackedFloat32Array
var _output: PackedFloat32Array
var _ih_offsets: PackedInt32Array
var _ho_offsets: PackedInt32Array


func _init(
		p_input_size: int = 389,
		p_hidden_size: int = 64,
		p_output_size: int = 128,
		p_randomize: bool = true
	) -> void:
	input_size = p_input_size
	hidden_size = p_hidden_size
	output_size = p_output_size

	weights_ih.resize(input_size * hidden_size)
	bias_h.resize(hidden_size)
	weights_ho.resize(hidden_size * output_size)
	bias_o.resize(output_size)
	_hidden.resize(hidden_size)
	_output.resize(output_size)
	_ih_offsets.resize(hidden_size)
	_ho_offsets.resize(output_size)
	for h in hidden_size:
		_ih_offsets[h] = h * input_size
	for o in output_size:
		_ho_offsets[o] = o * hidden_size

	if p_randomize:
		randomize_weights()


func randomize_weights() -> void:
	var ih_scale := sqrt(2.0 / input_size)
	var ho_scale := sqrt(2.0 / hidden_size)
	for i in weights_ih.size():
		weights_ih[i] = randf_range(-ih_scale, ih_scale)
	for i in bias_h.size():
		bias_h[i] = randf_range(-0.1, 0.1)
	for i in weights_ho.size():
		weights_ho[i] = randf_range(-ho_scale, ho_scale)
	for i in bias_o.size():
		bias_o[i] = randf_range(-0.1, 0.1)


func forward(inputs: PackedFloat32Array) -> PackedFloat32Array:
	var ih := weights_ih
	var bh := bias_h
	var ho := weights_ho
	var bo := bias_o
	var hidden := _hidden
	var output := _output
	var input_len := input_size
	var hidden_len := hidden_size
	var output_len := output_size

	var ih_offsets := _ih_offsets
	var ho_offsets := _ho_offsets

	for h in hidden_len:
		var sum := bh[h]
		var offset := ih_offsets[h]
		for i in input_len:
			sum += ih[offset + i] * inputs[i]
		hidden[h] = tanh(sum)

	for o in output_len:
		var sum := bo[o]
		var offset := ho_offsets[o]
		for h in hidden_len:
			sum += ho[offset + h] * hidden[h]
		output[o] = tanh(sum)

	return output


func benchmark_forward(iterations: int = 1000) -> void:
	var input := PackedFloat32Array()
	input.resize(input_size)
	for i in input.size():
		input[i] = randf()
	var start := Time.get_ticks_usec()
	for i in iterations:
		forward(input)
	var elapsed := Time.get_ticks_usec() - start
	print("forward ", iterations, " ", elapsed, " usec total, ", float(elapsed) / float(iterations), " usec/call")


func get_weights() -> PackedFloat32Array:
	var all_weights := PackedFloat32Array()
	all_weights.append_array(weights_ih)
	all_weights.append_array(bias_h)
	all_weights.append_array(weights_ho)
	all_weights.append_array(bias_o)
	return all_weights


func set_weights(weights: PackedFloat32Array) -> void:
	var idx := 0
	for i in weights_ih.size():
		weights_ih[i] = weights[idx]; idx += 1
	for i in bias_h.size():
		bias_h[i] = weights[idx]; idx += 1
	for i in weights_ho.size():
		weights_ho[i] = weights[idx]; idx += 1
	for i in bias_o.size():
		bias_o[i] = weights[idx]; idx += 1


func get_weight_count() -> int:
	return weights_ih.size() + bias_h.size() + weights_ho.size() + bias_o.size()


func clone():
	## Direct array copy: avoids the ~33K-float intermediate allocation from
	## get_weights()/set_weights(), and skips the wasted randomize_weights() in new().
	var copy = get_script().new(input_size, hidden_size, output_size, false)
	copy.weights_ih = weights_ih.duplicate()
	copy.bias_h = bias_h.duplicate()
	copy.weights_ho = weights_ho.duplicate()
	copy.bias_o = bias_o.duplicate()
	return copy


func mutate(mutation_rate: float = 0.1, mutation_strength: float = 0.3) -> void:
	_mutate_array(weights_ih, mutation_rate, mutation_strength)
	_mutate_array(bias_h, mutation_rate, mutation_strength)
	_mutate_array(weights_ho, mutation_rate, mutation_strength)
	_mutate_array(bias_o, mutation_rate, mutation_strength)


func _mutate_array(arr: PackedFloat32Array, mutation_rate: float, mutation_strength: float) -> void:
	## Geometric-skip mutation: jumps directly to the next mutated index using the
	## inverse CDF of the geometric distribution. O(k) randf() calls where k is the
	## number of actual mutations, vs O(n) for the naive per-element approach.
	## Produces an identical distribution to: if randf() < rate: arr[i] += randfn(...)
	if mutation_rate <= 0.0 or arr.is_empty():
		return
	if mutation_rate >= 1.0:
		for i in arr.size():
			arr[i] += randfn(0.0, mutation_strength)
		return
	var log1mp := log(1.0 - mutation_rate)
	var i := int(floor(log(maxf(randf(), 1e-15)) / log1mp))
	while i < arr.size():
		arr[i] += randfn(0.0, mutation_strength)
		i += 1 + int(floor(log(maxf(randf(), 1e-15)) / log1mp))


func crossover_with(other):
	## Two-point crossover operating directly on weight sub-arrays.
	## Avoids the 3 x ~33K-float intermediate allocation from get_weights().
	var child = get_script().new(input_size, hidden_size, output_size, false)
	var total := get_weight_count()
	var p1 := randi() % total
	var p2 := randi() % total
	if p1 > p2:
		var tmp := p1; p1 = p2; p2 = tmp

	var offset := 0
	_crossover_segment(child.weights_ih, weights_ih, other.weights_ih, p1, p2, offset)
	offset += weights_ih.size()
	_crossover_segment(child.bias_h, bias_h, other.bias_h, p1, p2, offset)
	offset += bias_h.size()
	_crossover_segment(child.weights_ho, weights_ho, other.weights_ho, p1, p2, offset)
	offset += weights_ho.size()
	_crossover_segment(child.bias_o, bias_o, other.bias_o, p1, p2, offset)
	return child


func _crossover_segment(
		dst: PackedFloat32Array,
		a: PackedFloat32Array,
		b: PackedFloat32Array,
		p1: int, p2: int, offset: int
	) -> void:
	for i in a.size():
		var gi := offset + i
		dst[i] = b[i] if (gi >= p1 and gi < p2) else a[i]
