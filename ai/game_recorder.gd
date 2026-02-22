class_name GameRecorder
extends RefCounted

## Records complete chess games and saves them as JSON replay files.
## Each replay includes full move history, metadata, and board snapshots
## that can be loaded by the ReplayViewer for playback.

const BoardStateScript = preload("res://chess/board_state.gd")

const REPLAY_DIR := "user://replays/"

var _moves: PackedInt32Array = PackedInt32Array()  # Every move as encoded int
var _board_snapshots: Array = []  # BoardState clone after each move
var _metadata: Dictionary = {}
var _recording := false


func start_recording(metadata: Dictionary = {}) -> void:
    ## Begin recording a new game. Metadata should include generation, white_id, black_id, etc.
    _moves.clear()
    _board_snapshots.clear()
    _metadata = metadata.duplicate()
    _metadata["timestamp"] = Time.get_unix_time_from_system()
    _recording = true


func record_move(move: int, board_after) -> void:
    ## Record a single move and the resulting board state.
    if not _recording:
        return
    _moves.append(move)
    _board_snapshots.append(board_after.clone())


func stop_recording(result: int) -> void:
    ## Finish recording. Result: 1=white wins, -1=black wins, 2=draw.
    _metadata["result"] = result
    _metadata["total_moves"] = _moves.size()
    _recording = false


func is_recording() -> bool:
    return _recording


func get_moves() -> PackedInt32Array:
    return _moves


func get_metadata() -> Dictionary:
    return _metadata


func get_board_snapshots() -> Array:
    return _board_snapshots


func save_to_file(filename: String = "") -> String:
    ## Save the recorded game to a JSON file. Returns the full path.
    ## If filename is empty, generates one from metadata.
    if filename.is_empty():
        var gen: int = _metadata.get("generation", 0)
        var ts: int = int(_metadata.get("timestamp", 0))
        filename = "game_gen%d_%d.json" % [gen, ts]

    _ensure_replay_dir()
    var path := REPLAY_DIR + filename

    var data := {
        "metadata": _metadata,
        "moves": _serialize_moves(),
    }

    var file := FileAccess.open(path, FileAccess.WRITE)
    if file == null:
        push_error("GameRecorder: Failed to open %s for writing" % path)
        return ""
    file.store_string(JSON.stringify(data, "\t"))
    file.close()
    return path


static func load_from_file(path: String) -> Dictionary:
    ## Load a replay file. Returns {"metadata": Dict, "moves": PackedInt32Array} or empty on error.
    if not FileAccess.file_exists(path):
        push_error("GameRecorder: File not found: %s" % path)
        return {}

    var file := FileAccess.open(path, FileAccess.READ)
    if file == null:
        return {}

    var json := JSON.new()
    var err := json.parse(file.get_as_text())
    file.close()

    if err != OK:
        push_error("GameRecorder: JSON parse error in %s" % path)
        return {}

    var data: Dictionary = json.get_data()
    var result := {
        "metadata": data.get("metadata", {}),
        "moves": _deserialize_moves(data.get("moves", [])),
    }
    return result


static func list_replays() -> Array[String]:
    ## List all replay files in the replay directory.
    var files: Array[String] = []
    var dir := DirAccess.open(REPLAY_DIR)
    if dir == null:
        return files
    dir.list_dir_begin()
    var fname := dir.get_next()
    while fname != "":
        if fname.ends_with(".json"):
            files.append(fname)
        fname = dir.get_next()
    dir.list_dir_end()
    files.sort()
    return files


static func replay_moves(moves: PackedInt32Array) -> Array:
    ## Reconstruct board states from a move list. Returns Array of BoardState.
    var states: Array = []
    var board := BoardStateScript.new()
    board.setup_initial()
    states.append(board.clone())
    for move in moves:
        board.make_move(move)
        states.append(board.clone())
    return states


func _serialize_moves() -> Array:
    var arr: Array = []
    for m in _moves:
        arr.append([BoardStateScript.move_from(m), BoardStateScript.move_to(m)])
    return arr


static func _deserialize_moves(arr: Array) -> PackedInt32Array:
    var moves := PackedInt32Array()
    for item in arr:
        if item is Array and item.size() == 2:
            moves.append(BoardStateScript.encode_move(int(item[0]), int(item[1])))
    return moves


static func _ensure_replay_dir() -> void:
    if not DirAccess.dir_exists_absolute(REPLAY_DIR):
        DirAccess.make_dir_recursive_absolute(REPLAY_DIR)
