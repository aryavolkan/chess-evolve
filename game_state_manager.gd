class_name GameStateManager
extends RefCounted

## Tracks per-episode game state: kills, score, combos.

var kills: int = 0
var score: float = 0.0
var combo: int = 0


func add_kill() -> void:
	kills += 1

	# First-kill bonus: strongly rewards the first kill to bootstrap combat learning.
	# DESIGN NOTE: Getting the first kill requires positioning + aiming + timing to
	# coincide — extremely sparse for random networks. This 2000pt bonus gives strong
	# selection signal distinguishing "can kill" from "merely survives."
	var first_kill_bonus: float = 2000.0 if kills == 1 else 0.0

	combo += 1
	var combo_multiplier := 1.0 + (combo - 1) * 0.1
	var bonus := ScoreManager.KILL_MULTIPLIER * combo_multiplier + first_kill_bonus
	score += bonus


func reset() -> void:
	kills = 0
	score = 0.0
	combo = 0
