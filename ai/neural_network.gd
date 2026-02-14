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


func _init(p_input_size: int = 389, p_hidden_size: int = 64, p_output_size: int = 128) -> void:
	input_size = p_input_size
	hidden_size = p_hidden_size
	output_size = p_output_size

	weights_ih.resize(input_size * hidden_size)
	bias_h.resize(hidden_size)
	weights_ho.resize(hidden_size * output_size)
	bias_o.resize(output_size)
	_hidden.resize(hidden_size)
	_output.resize(output_size)

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
	for h in hidden_size:
		var sum := bias_h[h]
		var offset := h * input_size
		for i in input_size:
			sum += weights_ih[offset + i] * inputs[i]
		_hidden[h] = tanh(sum)

	for o in output_size:
		var sum := bias_o[o]
		var offset := o * hidden_size
		for h in hidden_size:
			sum += weights_ho[offset + h] * _hidden[h]
		_output[o] = tanh(sum)

	return _output


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
	var copy = get_script().new(input_size, hidden_size, output_size)
	copy.set_weights(get_weights())
	return copy


func mutate(mutation_rate: float = 0.1, mutation_strength: float = 0.3) -> void:
	for i in weights_ih.size():
		if randf() < mutation_rate:
			weights_ih[i] += randfn(0.0, mutation_strength)
	for i in bias_h.size():
		if randf() < mutation_rate:
			bias_h[i] += randfn(0.0, mutation_strength)
	for i in weights_ho.size():
		if randf() < mutation_rate:
			weights_ho[i] += randfn(0.0, mutation_strength)
	for i in bias_o.size():
		if randf() < mutation_rate:
			bias_o[i] += randfn(0.0, mutation_strength)


func crossover_with(other):
	var child = get_script().new(input_size, hidden_size, output_size)
	var wa := get_weights()
	var wb := other.get_weights()
	var wc := PackedFloat32Array()
	wc.resize(wa.size())
	var p1 := randi() % wa.size()
	var p2 := randi() % wa.size()
	if p1 > p2: var tmp := p1; p1 = p2; p2 = tmp
	for i in wa.size():
		wc[i] = wb[i] if (i >= p1 and i < p2) else wa[i]
	child.set_weights(wc)
	return child
