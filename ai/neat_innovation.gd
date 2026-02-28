extends RefCounted
class_name NeatInnovation

## Global innovation tracker for NEAT.

var _next_innovation: int = 0
var _next_node_id: int = 0
var _innovation_cache: Dictionary = {}


func _init(initial_node_id: int = 0) -> void:
    _next_node_id = initial_node_id


func get_innovation(in_id: int, out_id: int) -> int:
    var key := "%d:%d" % [in_id, out_id]
    if _innovation_cache.has(key):
        return _innovation_cache[key]
    var innov := _next_innovation
    _innovation_cache[key] = innov
    _next_innovation += 1
    return innov


func allocate_node_id() -> int:
    var id := _next_node_id
    _next_node_id += 1
    return id


func get_next_innovation() -> int:
    return _next_innovation


func get_next_node_id() -> int:
    return _next_node_id


func seed_from_genome(genome) -> void:
    ## Seed tracker state from an existing genome so new mutations get
    ## consistent innovation numbers that don't collide with the template.
    # Set _next_node_id past max node ID in genome
    for node in genome.node_genes:
        if node.id >= _next_node_id:
            _next_node_id = node.id + 1

    # Set _next_innovation past max innovation number and populate cache
    for conn in genome.connection_genes:
        if conn.innovation >= _next_innovation:
            _next_innovation = conn.innovation + 1
        var key := "%d:%d" % [conn.in_id, conn.out_id]
        _innovation_cache[key] = conn.innovation


func reset_generation_cache() -> void:
    _innovation_cache.clear()
