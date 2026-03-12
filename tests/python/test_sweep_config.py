import runpy

REQUIRED_TOP_LEVEL_KEYS = {"method", "metric", "parameters"}


def test_root_sweep_config_has_expected_shape() -> None:
    module = runpy.run_path("configs/sweep_config.py")
    sweep_config = module["sweep_config"]

    assert REQUIRED_TOP_LEVEL_KEYS.issubset(sweep_config)
    assert sweep_config["metric"]["name"] == "combined_best"


def test_scripts_sweep_config_has_expected_shape() -> None:
    module = runpy.run_path("scripts/sweep_config.py")
    sweep_config = module["sweep_config"]

    assert REQUIRED_TOP_LEVEL_KEYS.issubset(sweep_config)
    assert sweep_config["metric"]["goal"] == "maximize"
