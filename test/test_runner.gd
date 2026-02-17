extends SceneTree
## Test runner for Chess Evolve. Run with: godot --headless --script test/test_runner.gd

var _tests_run := 0
var _tests_passed := 0
var _tests_failed := 0
var _current_test := ""
var _failure_messages: Array[String] = []


func _init() -> void:
	print("\n========================================")
	print("     CHESS EVOLVE TEST SUITE")
	print("========================================\n")

	_run_all_tests()

	print("\n========================================")
	print("       RESULTS")
	print("========================================")
	print("Tests run:    %d" % _tests_run)
	print("Tests passed: %d" % _tests_passed)
	print("Tests failed: %d" % _tests_failed)

	if _failure_messages.size() > 0:
		print("\nFAILURES:")
		for msg in _failure_messages:
			print("  - %s" % msg)

	print("========================================\n")
	quit(0 if _tests_failed == 0 else 1)


func _run_all_tests() -> void:
	var test_suites: Array = [
		preload("res://test/test_board_state.gd"),
		preload("res://test/test_encoder.gd"),
		preload("res://test/test_neural_network.gd"),
		preload("res://test/test_evolution.gd"),
		preload("res://test/test_fitness.gd"),
		preload("res://test/test_training.gd"),
		preload("res://test/test_game_recorder.gd"),
		preload("res://test/test_integration.gd"),
	]

	for suite_script in test_suites:
		var suite = suite_script.new()
		suite._runner = self
		print("--- %s ---" % suite_script.resource_path.get_file())
		suite._run_tests()


func _start_test(name: String) -> void:
	_current_test = name
	_tests_run += 1

func _pass_test() -> void:
	_tests_passed += 1
	print("  ✓ %s" % _current_test)

func _fail_test(message: String) -> void:
	_tests_failed += 1
	var full := "%s: %s" % [_current_test, message]
	_failure_messages.append(full)
	print("  ✗ %s - %s" % [_current_test, message])
