# Contributing to Chess Evolve

## Quick Start

```bash
# Run all tests (must pass before any changes)
godot --headless --script test/test_runner.gd
```

## Project Conventions

- **GDScript + optional Rust GDExtension** — no C#, no external GDScript dependencies. Rust crates in `rust/` provide optional acceleration.
- **RefCounted base** for non-scene classes (no manual memory management)
- **Tests for everything** — if you add logic, add tests in `test/`
- **Constants in one place** — piece types, values, and symbols live in `chess/constants.gd`

## Architecture Rules

1. **`chess/` has no AI dependencies** — board logic is self-contained
2. **`ai/` depends on `chess/`** — encoder reads BoardState, fitness evaluates it
3. **`ui/` depends on both** — rendering and dashboard only
4. **`test/` can depend on everything**

## Adding a Test

1. Create `test/test_yourfeature.gd` extending `test/test_base.gd`
2. Add methods prefixed with `test_` — the runner discovers them automatically
3. Use `assert_eq`, `assert_true`, `assert_false` from the base class

## Key Design Decisions

- **Signed piece planes** (±1) instead of 12 separate planes — keeps input size manageable at 389
- **4096 (64×64) output space** — one logit per from-to pair, masked to legal moves
- **Independent white/black evolution** — allows asymmetric specialization
- **Tournament evaluation** over multiple opponents — reduces variance from random pairings

## Pull Request Process

Every pull request is validated by CI with separate quality gates for Python and GDScript.
Before opening a PR, run the same checks locally:

```bash
./scripts/lint_and_test.sh
```

You can also run each language-specific check independently:

```bash
./scripts/lint_python.sh
./scripts/lint_gdscript.sh
```

PRs must pass all of the checks above before they can be merged.

