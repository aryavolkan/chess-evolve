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


func reset_generation_cache() -> void:
    _innovation_cache.clear()
