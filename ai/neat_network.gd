extends RefCounted
class_name NeatNetwork

## Phenotype network built from a NeatGenome.

var _node_order: PackedInt32Array
var _input_ids: PackedInt32Array
var _output_ids: PackedInt32Array
var _biases: Dictionary = {}
var _connections: Array = []
var _activations: Dictionary = {}
var _incoming: Dictionary = {}
var _is_input: Dictionary = {}
var _cached_outputs: PackedFloat32Array


static func from_genome(genome: NeatGenome) -> NeatNetwork:
    var net := NeatNetwork.new()
    net.build(genome)
    return net


func build(genome: NeatGenome) -> void:
    for node in genome.node_genes:
        _biases[node.id] = node.bias
        _incoming[node.id] = []
        if node.type == 0:
            _input_ids.append(node.id)
            _is_input[node.id] = true
        elif node.type == 2:
            _output_ids.append(node.id)

    for conn in genome.connection_genes:
        if conn.enabled:
            _connections.append({
                "in_id": conn.in_id,
                "out_id": conn.out_id,
                "weight": conn.weight,
            })
            if _incoming.has(conn.out_id):
                _incoming[conn.out_id].append({"in_id": conn.in_id, "weight": conn.weight})

    _node_order = _topological_sort(genome)

    for node in genome.node_genes:
        _activations[node.id] = 0.0

    _cached_outputs.resize(_output_ids.size())


func forward(inputs: PackedFloat32Array) -> PackedFloat32Array:
    for i in _input_ids.size():
        _activations[_input_ids[i]] = inputs[i] if i < inputs.size() else 0.0

    for node_id in _node_order:
        if _is_input.has(node_id):
            continue

        var sum: float = _biases.get(node_id, 0.0)
        var incoming: Array = _incoming.get(node_id, [])
        for edge in incoming:
            sum += _activations.get(edge.in_id, 0.0) * edge.weight

        _activations[node_id] = tanh(sum)

    for i in _output_ids.size():
        _cached_outputs[i] = _activations.get(_output_ids[i], 0.0)

    return _cached_outputs


func get_input_count() -> int:
    return _input_ids.size()


func get_output_count() -> int:
    return _output_ids.size()


func get_node_count() -> int:
    return _node_order.size()


func get_connection_count() -> int:
    return _connections.size()


func reset() -> void:
    for key in _activations:
        _activations[key] = 0.0


func _topological_sort(genome: NeatGenome) -> PackedInt32Array:
    var in_degree: Dictionary = {}
    var adjacency: Dictionary = {}

    for node in genome.node_genes:
        in_degree[node.id] = 0
        adjacency[node.id] = []

    for conn in genome.connection_genes:
        if not conn.enabled:
            continue
        if not in_degree.has(conn.out_id):
            continue
        in_degree[conn.out_id] = in_degree.get(conn.out_id, 0) + 1
        adjacency[conn.in_id].append(conn.out_id)

    var queue: Array[int] = []
    var queue_head: int = 0
    for node_id in in_degree:
        if in_degree[node_id] == 0:
            queue.append(node_id)

    var order := PackedInt32Array()
    while queue_head < queue.size():
        var node_id: int = queue[queue_head]
        queue_head += 1
        order.append(node_id)
        for downstream in adjacency.get(node_id, []):
            in_degree[downstream] -= 1
            if in_degree[downstream] == 0:
                queue.append(downstream)

    if order.size() < genome.node_genes.size():
        order = PackedInt32Array()
        for node in genome.node_genes:
            if node.type == 0:
                order.append(node.id)
        for node in genome.node_genes:
            if node.type == 1:
                order.append(node.id)
        for node in genome.node_genes:
            if node.type == 2:
                order.append(node.id)

    return order
