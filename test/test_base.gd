extends RefCounted
## Base class for test suites with assertion helpers.

var _runner = null
var _current_test_failed := false


func _run_tests() -> void:
    pass


func _test(name: String, callable: Callable) -> void:
    _runner.start_test(name)
    _current_test_failed = false
    callable.call()
    if not _current_test_failed:
        _runner.pass_test()


func assert_true(condition: bool, message: String = "Expected true") -> void:
    if not condition: _fail(message)

func assert_false(condition: bool, message: String = "Expected false") -> void:
    if condition: _fail(message)

func assert_eq(actual, expected, message: String = "") -> void:
    if actual != expected:
        _fail(message if message else "Expected %s but got %s" % [expected, actual])

func assert_ne(actual, not_expected, message: String = "") -> void:
    if actual == not_expected:
        _fail(message if message else "Expected value to not equal %s" % not_expected)

func assert_gt(actual: float, expected: float, message: String = "") -> void:
    if actual <= expected:
        _fail(message if message else "Expected %s > %s" % [actual, expected])

func assert_lt(actual: float, expected: float, message: String = "") -> void:
    if actual >= expected:
        _fail(message if message else "Expected %s < %s" % [actual, expected])

func assert_not_null(value, message: String = "Expected non-null") -> void:
    if value == null: _fail(message)

func _fail(message: String) -> void:
    if not _current_test_failed:
        _current_test_failed = true
        _runner.fail_test(message)
