# Contributing to Chess Evolve

## Quick Start

```bash
# Build Rust crates
cd rust/chess-cpu && maturin develop --release && cd ../..
cd rust/evolve-ga && maturin develop --release && cd ../..
cd rust/neat-ga && maturin develop --release && cd ../..

# Run all tests
python -m pytest tests/python -q
godot --headless --path . -s test/test_runner.gd

# Run all lints + tests
./scripts/lint_and_test.sh
```

## Project Conventions

- **GDScript + Rust + Python** — Rust PyO3 crates for performance, Python for training orchestration, GDScript for Godot UI
- **RefCounted base** for non-scene GDScript classes (no manual memory management)
- **Tests for everything** — Python tests in `tests/python/`, GDScript tests in `test/`
- **Constants in one place** — piece types, values, and symbols live in `chess/constants.gd`

## Architecture Rules

1. **`chess/` has no AI dependencies** — board logic is self-contained
2. **`ai/` depends on `chess/`** — encoder reads BoardState, fitness evaluates it
3. **`ui/` depends on both** — rendering and dashboard only
4. **Rust PyO3 crates** (`chess-cpu`, `evolve-ga`, `neat-ga`) are independent of each other and of `chess-native`
5. **`test/` and `tests/python/`** can depend on everything

## Adding Tests

### Python
1. Create `tests/python/test_yourfeature.py`
2. Use standard pytest conventions

### GDScript
1. Create `test/test_yourfeature.gd` extending `test/test_base.gd`
2. Add methods prefixed with `test_` — the runner discovers them automatically
3. Use `assert_eq`, `assert_true`, `assert_false` from the base class

## Key Design Decisions

- **Signed piece planes** (+/-1) instead of 12 separate planes — keeps input size manageable at 389
- **4096 (64x64) output space** — one logit per from-to pair, masked to legal moves
- **Independent white/black evolution** — allows asymmetric specialization
- **Tournament evaluation** over multiple opponents — reduces variance from random pairings
- **`combined_best = min(white_best, black_best)`** — balanced optimization metric

## Pull Request Process

Every pull request is validated by CI with separate quality gates for Python and GDScript.
Before opening a PR, run the same checks locally:

```bash
./scripts/lint_and_test.sh
```

You can also run each check independently:

```bash
# Python
ruff check scripts/ train_wandb.py sweep_config.py
python -m pytest tests/python -q

# GDScript
./scripts/lint_gdscript.sh

# Rust
cargo fmt --check --manifest-path rust/chess-native/Cargo.toml
cargo clippy --manifest-path rust/chess-native/Cargo.toml -- -D warnings
```

PRs must pass all checks before they can be merged.
