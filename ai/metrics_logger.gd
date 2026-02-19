class_name MetricsLogger
extends RefCounted

## Writes generation-by-generation training metrics to configurable path
## so that external tools (e.g. W&B bridge) can pick them up.
##
## Two outputs per write:
##   metrics_path        — latest gen snapshot (overwritten each gen, for compatibility)
##   metrics_path + "l"  — JSONL history file (appended each gen, for incremental polling)

var metrics_path: String = "user://metrics.json"

func _init(path: String = "") -> void:
	if not path.is_empty():
		metrics_path = path

func write_metrics(stats: Dictionary) -> void:
	var payload := stats.duplicate(true)
	payload["updated_at"] = Time.get_unix_time_from_system()

	print("MetricsLogger: Writing metrics to %s" % metrics_path)
	var absolute_path := ProjectSettings.globalize_path(metrics_path)
	print("MetricsLogger: Absolute path: %s" % absolute_path)
	dir_create_recursive(absolute_path)

	# Write latest-gen snapshot (overwrite)
	var file := FileAccess.open(metrics_path, FileAccess.WRITE)
	if file == null:
		push_warning("Failed to open metrics file at %s" % metrics_path)
		return
	file.store_string(JSON.new().stringify(payload, "\t"))
	file.close()

	# Append to JSONL history (one line per gen)
	var jsonl_path := metrics_path + "l"  # metrics.json → metrics.jsonl
	var jsonl_file := FileAccess.open(jsonl_path, FileAccess.READ_WRITE)
	if jsonl_file == null:
		jsonl_file = FileAccess.open(jsonl_path, FileAccess.WRITE)
	if jsonl_file != null:
		jsonl_file.seek_end(0)
		jsonl_file.store_line(JSON.new().stringify(payload))
		jsonl_file.close()


func dir_create_recursive(abs_path: String) -> void:
	var dir_path := abs_path.get_base_dir()
	DirAccess.make_dir_recursive_absolute(dir_path)
