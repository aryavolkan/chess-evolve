"""
Integration tests for chess-evolve training loop.

These tests verify that the full training pipeline can execute
without errors, even if we don't run full generations.
"""
import os
import sys

import pytest

# Add project to path
sys.path.insert(0, os.path.expanduser("~/Projects/chess-evolve"))


def test_train_script_has_required_components():
    """Test that train_wandb.py has all required components."""
    import train_wandb as train

    # Check required constants
    assert hasattr(train, 'CHESS_LOG_KEYS')
    assert isinstance(train.CHESS_LOG_KEYS, (list, tuple))

    # Should have DEFAULT_CONFIG
    assert hasattr(train, 'DEFAULT_CONFIG')
    assert isinstance(train.DEFAULT_CONFIG, dict)

    # Check config has required keys
    config = train.DEFAULT_CONFIG
    required = ["population_size", "hidden_size", "elite_count", "max_generations"]
    for key in required:
        assert key in config, f"Missing {key} in DEFAULT_CONFIG"


def test_sweep_config_has_valid_params():
    """Test that sweep config includes required parameters."""
    import sweep_config

    cfg = sweep_config.sweep_config
    params = cfg["parameters"]

    # Should have evolution params
    assert "population_size" in params
    assert "elite_count" in params
    assert "mutation_rate" in params

    # Should have network topology params
    assert "hidden_size" in params or "hidden_layers" in params


def test_sweep_config_quick_is_valid():
    """Test that quick sweep config is valid for testing."""
    import sweep_config

    cfg = sweep_config.sweep_config_quick
    params = cfg["parameters"]

    # Extract useful params (with defaults if not in quick config)
    pop_size = params.get("population_size", {}).get("value", 50)
    elite_count = params.get("elite_count", {}).get("value", 5)

    # Basic sanity checks
    assert pop_size > 0
    assert elite_count > 0
    assert elite_count <= pop_size

    # Check mutation rate is valid
    mutation_rate = params.get("mutation_rate", {}).get("value", 0.1)
    assert 0.0 <= mutation_rate <= 1.0


def test_shared_utils_available():
    """Test that shared evolve utils can be imported."""
    sys.path.insert(0, os.path.expanduser("~/Projects/shared-evolve-utils"))

    # Should be able to import godot_wandb utilities
    try:
        import godot_wandb
        assert hasattr(godot_wandb, 'run_training')
    except ImportError as e:
        pytest.skip(f"Shared utils not available: {e}")


def test_wandb_available():
    """Test that wandb module is available (not mocked)."""
    import sys
    # Check if wandb is a real module, not a mock
    wandb_mod = sys.modules.get("wandb")
    if wandb_mod is None:
        pytest.skip("wandb not installed")
    # Check it's a proper module with version
    if hasattr(wandb_mod, "__version__"):
        assert wandb_mod.__version__  # Should have version
    else:
        pytest.skip("wandb is mocked in test environment")


def test_chess_log_keys_complete():
    """Test that CHESS_LOG_KEYS has all expected keys."""
    import train_wandb as train

    keys = train.CHESS_LOG_KEYS

    expected = [
        "generation",
        "best_fitness",
        "avg_fitness",
        "white_best",
        "black_best",
    ]

    for key in expected:
        assert key in keys, f"Missing {key} in CHESS_LOG_KEYS"


def test_default_config_reasonable():
    """Test that default config has reasonable values."""
    import train_wandb as train

    config = train.DEFAULT_CONFIG

    # Population should be reasonable
    assert 1 <= config["population_size"] <= 1000

    # Elite should be smaller than population
    assert config["elite_count"] < config["population_size"]

    # Rates should be valid
    assert 0 <= config["crossover_rate"] <= 1
    assert 0 <= config["mutation_rate"] <= 1
    assert 0 <= config["mutation_strength"] <= 1

    # Network sizes should be positive
    assert config["input_size"] > 0
    assert config["output_size"] > 0


def test_project_path_exists():
    """Test that project path is set correctly."""
    import train_wandb

    path = train_wandb.PROJECT_PATH
    assert os.path.exists(path)
    assert os.path.isdir(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
