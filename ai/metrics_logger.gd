extends RefCounted
class_name MetricsLogger

## Writes generation-by-generation training metrics to `user://metrics.json`
## so that external tools (e.g. W&B bridge) can pick them up.

const METRICS_PATH := "user://metrics.json"

func write_metrics(stats: Dictionary) -> void:
	var payload := stats.duplicate(true)
	payload["updated_at"] = Time.get_unix_time_from_system()

	var absolute_path := ProjectSettings.globalize_path(METRICS_PATH)
	dir_create_recursive(absolute_path)

	var file := FileAccess.open(METRICS_PATH, FileAccess.WRITE)
	if file == null:
		push_warning("Failed to open metrics file at %s" % METRICS_PATH)
		return

	var json := JSON.new()
	file.store_string(json.stringify(payload, "\t"))
	file.close()


func dir_create_recursive(abs_path: String) -> void:
	var dir_path := abs_path.get_base_dir()
	DirAccess.make_dir_recursive_absolute(dir_path)
