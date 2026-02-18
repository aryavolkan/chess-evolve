class_name MetricsLogger
extends RefCounted

## Writes generation-by-generation training metrics to configurable path
## so that external tools (e.g. W&B bridge) can pick them up.

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

	var file := FileAccess.open(metrics_path, FileAccess.WRITE)
	if file == null:
		push_warning("Failed to open metrics file at %s" % metrics_path)
		return

	var json := JSON.new()
	file.store_string(json.stringify(payload, "\t"))
	file.close()


func dir_create_recursive(abs_path: String) -> void:
	var dir_path := abs_path.get_base_dir()
	DirAccess.make_dir_recursive_absolute(dir_path)
