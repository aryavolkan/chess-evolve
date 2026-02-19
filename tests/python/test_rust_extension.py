from __future__ import annotations

import configparser
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GDEXTENSION_FILE = PROJECT_ROOT / "chess-native.gdextension"


def test_gdextension_file_exists() -> None:
    assert GDEXTENSION_FILE.exists(), "chess-native.gdextension is missing"


def test_gdextension_release_library_exists_for_platform() -> None:
    config = configparser.ConfigParser()
    config.read(GDEXTENSION_FILE)

    assert "libraries" in config, "gdextension file missing [libraries] section"

    platform_map = {"darwin": "macos", "linux": "linux", "win32": "windows"}
    platform_key = platform_map.get(sys.platform)
    if platform_key is None:
        pytest.skip(f"Unsupported platform: {sys.platform}")

    lib_key = f"{platform_key}.release"
    lib_path = config["libraries"].get(lib_key)
    if not lib_path:
        pytest.skip(f"No release library entry for {lib_key} in gdextension file")

    cleaned_path = lib_path.strip().strip("\"").strip("'")
    resolved_path = PROJECT_ROOT / cleaned_path.replace("res://", "")
    assert resolved_path.exists(), f"Expected built library at {resolved_path}"
