#!/bin/bash
set -e
source .venv/bin/activate
python3 -m pytest tests/python/ -v
cd rust/chess-native && cargo test 2>&1
