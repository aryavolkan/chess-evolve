class_name TrainingModeBase
extends RefCounted

## Base class for all training modes.
## Defines shared scoring constants and lifecycle hooks.

## DESIGN NOTE: Reduced from 1000 to 500. At 1000 pts/sec, 60s survival = 60,000 pts
## which dwarfs a pawn kill (1000 pts). Agents evolve to hide rather than engage.
## At 500 pts/sec, survival and active combat are balanced — agents must fight to win.
const SURVIVAL_UNIT_SCORE: float = 500.0
