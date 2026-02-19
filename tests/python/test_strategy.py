import runpy
import sys
import types


def _load(path: str):
    return runpy.run_path(path)


def _collect_values(param: dict) -> list[float]:
    if "values" in param:
        return list(param["values"])
    if "value" in param:
        return [param["value"]]
    if "min" in param and "max" in param:
        return [param["min"], param["max"]]
    return []


def test_sweep_config_parameter_ranges() -> None:
    module = _load("sweep_config.py")
    for name in ("sweep_config", "sweep_config_quick", "sweep_config_deep"):
        cfg = module[name]
        params = cfg["parameters"]

        pop_values = _collect_values(params["population_size"])
        elite_values = _collect_values(params["elite_count"])
        assert pop_values and elite_values
        assert max(elite_values) <= min(pop_values)

        tour_values = _collect_values(params["tournament_opponents"])
        assert tour_values
        assert max(tour_values) < min(pop_values)

        for rate_key in ("mutation_rate", "mutation_strength", "crossover_rate"):
            rate = params[rate_key]
            values = _collect_values(rate)
            assert values
            assert min(values) >= 0.0
            assert max(values) <= 1.0

        assert params["use_tournament"]["value"] is True
        assert "tournament_mode" in params


def test_wandb_log_keys_match_train_script() -> None:
    train = _load("train_wandb.py")
    bridge = _load("scripts/wandb_bridge.py")
    assert train["CHESS_LOG_KEYS"] == bridge["CHESS_LOG_KEYS"]


def test_chess_sweep_worker_imports() -> None:
    dummy_wandb = types.SimpleNamespace(
        config={},
        init=lambda **kwargs: types.SimpleNamespace(
            url="mock",
            log=lambda *a, **k: None,
            finish=lambda **k: None,
        ),
        agent=lambda *a, **k: None,
    )

    dummy_godot = types.SimpleNamespace(
        SweepWorker=object,
        define_step_metric=lambda *a, **k: None,
        godot_user_dir=lambda *a, **k: None,
        launch_godot=lambda *a, **k: None,
        log_final_summary=lambda *a, **k: None,
        read_metrics=lambda *a, **k: None,
        run_sweep_agent=lambda *a, **k: None,
        wait_for_metrics=lambda *a, **k: True,
    )

    original_wandb = sys.modules.get("wandb")
    original_godot = sys.modules.get("godot_wandb")
    sys.modules["wandb"] = dummy_wandb
    sys.modules["godot_wandb"] = dummy_godot
    try:
        _load("overnight-agent/chess_sweep_worker.py")
    finally:
        if original_wandb is not None:
            sys.modules["wandb"] = original_wandb
        else:
            sys.modules.pop("wandb", None)
        if original_godot is not None:
            sys.modules["godot_wandb"] = original_godot
        else:
            sys.modules.pop("godot_wandb", None)
