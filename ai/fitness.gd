extends RefCounted
class_name ChessFitness

## Evaluates fitness of a chess player based on game outcome and play quality.

# Weights for fitness components
const WIN_BONUS := 10.0
const DRAW_BONUS := 3.0
const LOSS_PENALTY := 0.0
const MATERIAL_WEIGHT := 1.0
const MOBILITY_WEIGHT := 0.05
const KING_SAFETY_WEIGHT := 0.5
const MOVE_COUNT_BONUS := 0.01  # Reward for surviving longer
const CHECKMATE_BONUS := 5.0


static func evaluate(state, color: int, move_count: int) -> float:
	## Compute fitness for a player after a game.
	var fitness := 0.0

	# Game result
	if state.is_game_over:
		if state.result == 2:  # Draw
			fitness += DRAW_BONUS
		elif (state.result == 1 and color == 0) or (state.result == -1 and color == 1):
			fitness += WIN_BONUS
			# Extra bonus for checkmate (vs timeout/stalemate)
			fitness += CHECKMATE_BONUS
		else:
			fitness += LOSS_PENALTY

	# Material advantage
	var my_material: float = state.material_score(color)
	var opp_material: float = state.material_score(1 - color)
	fitness += (my_material - opp_material) * MATERIAL_WEIGHT

	# Mobility (at end of game)
	# Only count if game isn't over to avoid 0-move stalemate confusion
	if not state.is_game_over:
		var mobility: float = state.mobility_score(color)
		fitness += mobility * MOBILITY_WEIGHT

	# King safety
	fitness += state.king_safety_score(color) * KING_SAFETY_WEIGHT

	# Reward for longer games (encourages learning to play)
	fitness += move_count * MOVE_COUNT_BONUS

	return maxf(fitness, 0.0)
