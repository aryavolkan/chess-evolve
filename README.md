# Chess Evolve

[![tests](https://github.com/aryavolkan/chess-evolve/actions/workflows/tests.yml/badge.svg)](https://github.com/aryavolkan/chess-evolve/actions/workflows/tests.yml)
[![PR Quality](https://github.com/aryavolkan/chess-evolve/actions/workflows/pr-quality.yml/badge.svg)](https://github.com/aryavolkan/chess-evolve/actions/workflows/pr-quality.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](requirements.txt)
[![Rust](https://img.shields.io/badge/rust-stable-DEA584?logo=rust&logoColor=white)](rust/)

Coevolutionary neuroevolution for chess. Two populations of neural networks, one
playing White and one playing Black, evolve against each other with no gradients,
no game database and no hand-written evaluation: the only training signal is the
outcome of games between evolving opponents. A Rust simulation core plays the
games in parallel; a Python harness runs the evolutionary loop, the curriculum
and the experiment tracking.

## Why this project

- **A real arms race, measured honestly.** Coevolution hides progress (both sides
  improve at once), so every run is also scored against a fixed random benchmark
  population, an Elo-ranked Hall of Fame and, optionally, Stockfish. The results
  that did *not* work are written down too ([below](#results-and-what-we-learned)).
- **Throughput engineering.** The per-generation pipeline is
  `pairings → parallel Rust game simulation → fitness → Rust GA operators → W&B`.
  Populations are flat `float32` arrays handed to Rust as bytes; games run on all
  cores with rayon while the GIL is released.
- **Reproducible from a clean machine.** One `docker build`, pinned Python and
  Cargo dependencies, a 3-generation smoke run and a throughput benchmark that
  both run in CI.

## At a glance

| | |
|---|---|
| Network | 389 → 64 → 4096 feed-forward (tanh); **291,200 weights** per fixed-topology genome, or variable-topology NEAT genomes |
| Input | 6 signed piece planes × 64 squares (+1 White, −1 Black), side to move, 4 castling rights; the Rust encoder also offers a 391-float layout with en-passant file and halfmove clock |
| Output | one logit per (from, to) square pair, masked to legal moves |
| Simulation core | `rust/chess-cpu` (PyO3): bitboard move generation, NN forward pass, material / mobility / king-safety / king-danger metrics, mercy rule, parallel over games with rayon |
| Throughput | **~140 games/s, ~13,500 moves/s** on a 4-vCPU x86_64 container (random genomes, 100-move cap); 37 games/s single-threaded, so 3.7× on 4 threads. Reproduce with `scripts/bench_throughput.py` |
| Evolution | tournament selection (k = 2), two-point crossover (70 %), Gaussian mutation (rate 0.25, σ 0.12), elitism (2), fitness sharing (σ 0.08), 10 % immigration; NEAT with speciation and add-node / add-connection mutation |
| Curriculum | 5 stages: tactical puzzles → guided play → opponent ladder → Stockfish shaping → coevolution refinement (`python/curriculum.py`) |
| Evaluation | fixed random benchmark population, a Hall of Fame of historical opponents, Stockfish centipawn-loss fitness signal, a Lichess bot that plays the evolved genomes online |
| Tests and CI | 334 Python tests, 15 GDScript suites; ruff, gdlint, rustfmt and `clippy -D warnings` on every crate, pytest with the Rust crates built, an end-to-end smoke training run |

## Quickstart

### Docker (no toolchain needed)

```bash
docker build -t chess-evolve .
docker run --rm chess-evolve                               # 3-generation smoke run, W&B offline
docker run --rm chess-evolve scripts/bench_throughput.py   # games/s and moves/s on your machine
docker run --rm -e WANDB_MODE=online -e WANDB_API_KEY=... \
    chess-evolve train_wandb.py --config configs/steady_progress_config.json
```

### Native

Requires Python 3.11+ and a stable Rust toolchain.

```bash
pip install -r requirements.txt -r requirements-dev.txt

# Build the three PyO3 crates (chess_cpu, evolve_ga, neat_ga) and install them
for crate in chess-cpu evolve-ga neat-ga; do
  maturin build --release --locked --manifest-path rust/$crate/Cargo.toml --out dist
done
pip install dist/*.whl

# Smoke run: pop 10, 3 generations, finishes in seconds
WANDB_MODE=offline python train_wandb.py --config configs/smoke_config.json
```

Inside a virtualenv, `cd rust/chess-cpu && maturin develop --release` (and the same
for the other two crates) builds and installs in one step.

### Training

```bash
python train_wandb.py                                             # single run, auto-detects backend
python train_wandb.py --config configs/steady_progress_config.json # 5-stage curriculum pipeline
python train_wandb.py --config configs/neat_config.json           # NEAT, pop 500
python train_wandb.py --sweep <sweep-id>                          # join a W&B sweep
python train_wandb.py --chain 10                                  # chained runs, each seeded from the previous best
```

Backend selection: the Rust CPU backend is used whenever the `chess_cpu` and
`evolve_ga` / `neat_ga` modules import; otherwise training falls back to the
original Godot path (`godot --headless`). Stockfish-based stages need a `stockfish`
binary on `PATH` or `STOCKFISH_PATH`.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       Python harness                             │
│  train_wandb.py   backend detection, W&B logging, sweeps, chains │
│  cpu_trainer.py / neat_cpu_trainer.py   generation loops         │
│  fitness.py   curriculum.py   puzzles.py   lichess_bot.py        │
│  overnight-agent/   sweep workers, monitor, global elite pool    │
└───────────────┬──────────────────────────────────────────────────┘
                │ numpy float32 populations as bytes / PyO3
┌───────────────▼──────────────────────────────────────────────────┐
│                      Rust PyO3 crates                            │
│  chess-cpu    parallel game simulation, bitboards, NN forward    │
│  evolve-ga    selection, crossover, mutation, speciation, sharing│
│  neat-ga      NEAT genomes, innovation tracking, topology ops    │
└──────────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────────┐
│  Godot 4 (optional)  training dashboard, board viewers, replays, │
│  human-vs-AI play; chess-native GDExtension accelerates it       │
└──────────────────────────────────────────────────────────────────┘
```

One generation on the Rust backend:

1. Python builds the pairings (each individual meets `tournament_opponents`
   opponents from the other colour's population and the Hall of Fame).
2. `chess_cpu.simulate_games_batch()` plays every pairing in parallel: encode the
   board, forward pass, mask to legal moves, sample a move at the configured
   temperature, repeat to the move cap or a result; returns per-game outcome and
   fitness components.
3. `python/fitness.py` turns the components into fitness and outcome rates.
4. `evolve_ga` / `neat_ga` produce the next populations (elites, tournament
   selection, crossover, mutation, immigration, fitness sharing).
5. Benchmark games, Hall of Fame updates and metrics go to W&B; best genomes are
   saved per colour.

### Fitness

Defaults from `python/fitness.py`, overridable per config (`train_wandb.py`'s
default config, for example, raises `draw_bonus` to 3.0):

| Component | Weight | Notes |
|---|---|---|
| Win | 15.0 | plus 10.0 `checkmate_bonus` |
| Draw | 2.0 | scaled by material advantage, so a draw a piece up beats a dead-level one |
| Loss | −10.0 | |
| Material | 0.5× | net material difference |
| Mobility | 0.3× | legal-move count difference |
| Own king safety | 0.5× | |
| King danger | 1.0× | attack signals against the opponent's king |
| Captures | 0.2× | value of pieces taken |
| Move count | −0.002× | nudges towards decisive games |

The sweep metric is `combined_best = min(white_best, black_best)`, so a run only
scores well when both colours improve.

### Benchmark population and Hall of Fame

A fixed, never-evolving random population (20 genomes by default, 50 in the
curriculum configs) measures absolute progress as `bench_win_rate`. The Hall of
Fame keeps the strongest historical genomes per colour (Elo-ranked on the Godot
path) and feeds them back in as opponents so the populations cannot forget how to beat earlier
strategies.

## Results and what we learned

Throughput, measured with `scripts/bench_throughput.py` (150 games per round,
389 → 64 → 4096 genomes, 100-move cap, 4-vCPU x86_64 container):

| Threads | games/s | moves/s |
|---|---|---|
| 1 | 37 | 3,570 |
| 4 | 141 | 13,750 |

Learning, from the sweep notes in `configs/optimized_sweep_v*.yaml` and `docs/`:

- **Plain coevolution plateaus.** NEAT runs of 2,000 generations with populations
  of 200–500 reached only ~7 % win rate against the random benchmark; most
  mutations do not change game outcomes, so the fitness landscape is flat
  (`docs/curriculum_learning_plan.md`). That finding is why the curriculum exists.
- **The curriculum is the bottleneck, not the GA.** In the first 100-run Bayesian
  sweep 77 % of runs never left the puzzle stages; the best run reached a 45 %
  benchmark win rate. Later sweeps start directly at the opponent ladder.
- **Output encoding matters more than mutation rates.** For NEAT, a 128-output
  factored head averaged 0.244 benchmark win rate against 0.177 for 384 outputs
  (31-run grid), and the 200-run follow-up set the record at 0.444 with a
  population of 200 over 300 generations.
- **Sweep-derived defaults** (`docs/IMPROVING_TRAINING.md`): `elite_count = 2`
  beats 3 and 5; 128 hidden units underperform 32 and 64; crossover above 0.85
  hurts; minimax during training is 20–50× slower per move for less
  generation-level progress, so search is reserved for play, not training.
- **Playing strength is still modest.** The steady-progress pipeline spec puts
  the baseline at roughly 400–600 Elo and targets 1200+; the Lichess bot exists
  to measure that against real opponents rather than our own benchmark.

## Project layout

```
chess-evolve/
├── train_wandb.py          entry point: backend detection, W&B, sweeps, chained runs
├── python/
│   ├── cpu_trainer.py      fixed-topology generation loop (Rust backend)
│   ├── neat_cpu_trainer.py NEAT generation loop, Stockfish signal, puzzle stages
│   ├── fitness.py          fitness weights, outcome rates, tournament scores
│   ├── curriculum.py       5-stage curriculum manager
│   ├── puzzles.py          Lichess puzzle loading for stage 0
│   ├── lichess_bot.py      plays Hall-of-Fame genomes on Lichess (ensemble vote)
│   └── godot_wandb.py      Godot subprocess backend
├── rust/
│   ├── chess-cpu/          PyO3: game simulation, bitboards, NN forward pass
│   ├── evolve-ga/          PyO3: GA operators, speciation, islands
│   ├── neat-ga/            PyO3: NEAT genomes and evolution
│   └── chess-native/       gdext: GDExtension for the Godot path
├── configs/                training configs and W&B sweep definitions (with notes per iteration)
├── overnight-agent/        sweep workers, worker monitor, cross-run global elite pool
├── scripts/                lint/test runners, bench_throughput.py, puzzle preparation
├── tests/python/           pytest suite        tests/integration/  longer loops
├── ai/ chess/ ui/ scenes/  Godot: networks + evolution, chess rules + encoder, dashboard + board
├── test/                   GDScript tests (headless runner)
├── monitor/                local sweep-monitoring API + React dashboard
├── Dockerfile              reproducible Rust-backend training image
└── docs/                   architecture, training, tuning and game-system docs
```

Dependency rules: `chess/` has no AI dependencies, `ai/` depends on `chess/`,
`ui/` depends on both; the three PyO3 crates are independent of each other and
of `chess-native`.

## Tests and CI

```bash
python -m pytest tests/python -q                       # Python (Rust-backed tests run when the crates are installed)
godot --headless --path . -s test/test_runner.gd       # GDScript
./scripts/lint_and_test.sh                             # ruff + gdlint + pytest
cargo clippy --manifest-path rust/chess-cpu/Cargo.toml -- -D warnings   # per crate
```

| Workflow | Jobs |
|---|---|
| `tests.yml` | ruff, gdlint, rustfmt and clippy (`-D warnings`) on all four crates; pytest with the PyO3 wheels built; 3-generation smoke training on the Rust backend; throughput benchmark sanity run; Godot headless tests |
| `pr-quality.yml` | blocking Python lint + tests and GDScript lint on every PR |

## Lichess bot

```bash
python python/lichess_bot.py --test                   # dry run against itself
python python/lichess_bot.py --games 5                # accept challenges
python python/lichess_bot.py --challenge <bot-name>   # challenge a specific bot
```

Needs a Lichess bot account with `LICHESS_TOKEN` set. Moves are chosen by an
ensemble vote over the top genomes in `neat_best_genomes.json`.

## Godot dashboard

Open the project in Godot 4.2+ to watch training live: **Start Training**, per-colour
best and average fitness, games played, a speed selector (1×–8× generations per
frame) and showcase games between the best networks every 5 generations. The
`chess-native` GDExtension (`cargo build --release --manifest-path
rust/chess-native/Cargo.toml`) accelerates move generation and NN evaluation
on this path.

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — components and data flow
- [Training](docs/TRAINING.md) — configs, metrics, sweeps, CI
- [Improving Training](docs/IMPROVING_TRAINING.md) — tuning guide and diagnosis
- [AI System](docs/AI_SYSTEM.md) — network, evolution, fitness, Hall of Fame
- [Game System](docs/GAME_SYSTEM.md) — rules, board representation, encoder
- [Curriculum plan](docs/curriculum_learning_plan.md) and
  [bitboard + NEAT plan](docs/bitboard-and-neat-plan.md) — design notes
- [CHANGES.md](CHANGES.md) — changelog, [CONTRIBUTING.md](CONTRIBUTING.md) — conventions

## License

[MIT](LICENSE).
