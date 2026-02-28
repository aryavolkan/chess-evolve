# Changelog

Notable changes to chess-evolve, newest first. For detailed documentation see `docs/`.

## 2026-02-27

- **Elo tracking**: 14 Elo distribution metrics (min/p25/median/p75/max per color) synced to W&B
- **Documentation consolidation**: new [Improving Training](docs/IMPROVING_TRAINING.md) guide, updated all docs to reflect current 4096-output architecture

## 2026-02-15 — Phase 2 Complete

All Phase 2 roadmap items shipped:

- **NEAT topology evolution** — `chess_neat_evolution.gd`, `neat_genome.gd`, `neat_network.gd`
- **Hall of Fame** — top 20 per color, Elo-ranked, weighted opponent selection
- **Opening book** — ~30 embedded openings with configurable depth
- **Endgame hints** — K vs K, K+Q/R vs K pattern recognition in fitness
- **Human play mode** — `--human-play` CLI flag
- **Move animation** — tween-based transitions with capture fade
- **PGN export** — Standard Algebraic Notation with disambiguation

## 2026-02-12

- **Rust GDExtension** — complete Linux integration for move generation and NN eval

## 2026-02-10

- **Balanced optimization** — `combined_best = min(white_best, black_best)` as sweep metric
- **Immigration** — 10% random replacement per generation for diversity
- **Fitness sharing** — genetic distance reward (`sigma=0.08`)
- **Reduced selection pressure** — `tournament_k=2` (down from 3)

## Earlier

- **4096-output architecture** — replaced 64+64 factored encoder with full from-to output space
- **Tournament evaluation** — round-robin and Swiss-system opponent pairing
- **Minimax player** — optional alpha-beta search wrapper over NN evaluation
- **Hall of Fame coevolution** — historical opponents prevent cycling
- **Curriculum learning** — 4-stage fitness weight schedule
- **Global Elite Pool** — cross-run genome sharing across parallel workers
- **Adaptive mutation** — auto-tighten on improvement, loosen on stagnation
- **Rust GA operators** — PyO3 bindings for selection, mutation, crossover
- **Worker health monitor** — auto-spawn with WhatsApp alerts
