class_name CurriculumManager
extends RefCounted

## Manages curriculum learning stages for neuroevolution training.
## Each stage increases difficulty as agents improve.

var stages: Array = [
	{
		"label": "Beginner",
		"min_gen": 0,
		"advancement_threshold": 3000.0,
		"max_moves": 60,
		"weights": {
			"win_bonus": 6.0,
			"draw_bonus": 1.0,
			"material_weight": 1.2,
			"mobility_weight": 0.1,
			"king_safety_weight": 0.2,
			"move_count_bonus": 0.02
		}
	},
	{
		"label": "Intermediate",
		"min_gen": 3,
		"advancement_threshold": 8000.0,
		"max_moves": 100,
		"weights": {
			"win_bonus": 8.0,
			"draw_bonus": 2.0,
			"material_weight": 1.0,
			"mobility_weight": 0.08,
			"king_safety_weight": 0.35,
			"move_count_bonus": 0.015
		}
	},
	{
		"label": "Advanced",
		"min_gen": 8,
		# Raised from 12000 to match Stage 2 — agents must maintain competence at harder difficulty
		"advancement_threshold": 15000.0,
		"max_moves": 150,
		"weights": {
			"win_bonus": 10.0,
			"draw_bonus": 3.0,
			"material_weight": 1.0,
			"mobility_weight": 0.05,
			"king_safety_weight": 0.5,
			"move_count_bonus": 0.01
		}
	},
	{
		"label": "Expert",
		"min_gen": 15,
		"advancement_threshold": 999999.0,
		"max_moves": 200,
		"weights": {
			"win_bonus": 12.0,
			"draw_bonus": 3.0,
			"material_weight": 1.0,
			"mobility_weight": 0.05,
			"king_safety_weight": 0.6,
			"move_count_bonus": 0.01
		}
	}
]

var current_stage_index: int = 0


func get_current_stage() -> Dictionary:
	return stages[current_stage_index]


func try_advance(best_score: float, current_gen: int) -> bool:
	## Advance to next stage if score threshold met and generation requirement satisfied.
	if current_stage_index >= stages.size() - 1:
		return false
	var stage := stages[current_stage_index]
	var next_stage := stages[current_stage_index + 1]
	if best_score >= stage["advancement_threshold"] and current_gen >= next_stage["min_gen"]:
		current_stage_index += 1
		print("Curriculum: Advanced to stage '%s'" % next_stage["label"])
		return true
	return false
