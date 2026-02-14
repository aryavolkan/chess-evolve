# Contributing to Chess Evolve

## Quick Start

```bash
# Run all tests (must pass before any changes)
godot --headless --script test/test_runner.gd
```

## Project Conventions

- **Pure GDScript** — no C#, no GDExtension, no external dependencies
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
- **From+To output decoding** — simpler than 4096 (64×64) output; works with legal move filtering
- **Independent white/black evolution** — allows asymmetric specialization
- **Fitness averaging** over multiple games — reduces variance from random pairings
