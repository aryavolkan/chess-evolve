class_name ChessFitness
extends RefCounted

## Evaluates fitness of a chess player based on game outcome and play quality.

const EndgameHintsScript = preload("res://chess/endgame_hints.gd")

# Weights for fitness components (mutable for curriculum)
static var win_bonus: float = 10.0
static var draw_bonus: float = 1.5
static var loss_penalty: float = -2.0
static var material_weight: float = 1.0
static var mobility_weight: float = 0.05
static var king_safety_weight: float = 0.5
static var move_count_bonus: float = -0.005  # Penalize longer games to discourage stalling
static var checkmate_bonus: float = 5.0
static var endgame_weight: float = 1.0

static func set_weights(weights: Dictionary) -> void:
    if weights.has("win_bonus"): win_bonus = weights["win_bonus"]
    if weights.has("draw_bonus"): draw_bonus = weights["draw_bonus"]
    if weights.has("loss_penalty"): loss_penalty = weights["loss_penalty"]
    if weights.has("material_weight"): material_weight = weights["material_weight"]
    if weights.has("mobility_weight"): mobility_weight = weights["mobility_weight"]
    if weights.has("king_safety_weight"): king_safety_weight = weights["king_safety_weight"]
    if weights.has("move_count_bonus"): move_count_bonus = weights["move_count_bonus"]
    if weights.has("checkmate_bonus"): checkmate_bonus = weights["checkmate_bonus"]
    if weights.has("endgame_weight"): endgame_weight = weights["endgame_weight"]


static func evaluate(state, color: int, move_count: int) -> float:
    ## Compute fitness for a player after a game.
    var fitness := 0.0

    # Material advantage
    var my_material: float = state.material_score(color)
    var opp_material: float = state.material_score(1 - color)

    # Game result
    if state.is_game_over:
        if state.result == 2:  # Draw — scale by material advantage
            var mat_adv: float = clampf((my_material - opp_material) / 10.0, -1.0, 1.0)
            fitness += draw_bonus * (0.5 + 0.5 * mat_adv)
        elif (state.result == 1 and color == 0) or (state.result == -1 and color == 1):
            fitness += win_bonus
            # Extra bonus for checkmate (vs timeout/stalemate)
            fitness += checkmate_bonus
        else:
            fitness += loss_penalty
    fitness += (my_material - opp_material) * material_weight

    # Mobility (at end of game)
    # Only count if game isn't over to avoid 0-move stalemate confusion
    if not state.is_game_over:
        var mobility: float = state.mobility_score(color)
        fitness += mobility * mobility_weight

    # King safety
    fitness += state.king_safety_score(color) * king_safety_weight

    # Endgame hints
    var endgame := EndgameHintsScript.evaluate_endgame(state)
    if not endgame.is_empty():
        var eg_bonus: float = endgame.get("eval_bonus", 0.0)
        # eval_bonus is positive for white advantage, negative for black
        if color == 1:
            eg_bonus = -eg_bonus
        fitness += eg_bonus * endgame_weight

    # Reward for longer games (encourages learning to play)
    fitness += move_count * move_count_bonus

    return fitness


static func evaluate_from_metrics(
        result: int,
        color: int,
        move_count: int,
        my_material: float,
        opp_material: float,
        my_mobility: float,
        my_king_safety: float,
        is_game_over: bool = true
    ) -> float:
    var fitness := 0.0
    if is_game_over:
        if result == 2:  # Draw — scale by material advantage
            var mat_adv: float = clampf((my_material - opp_material) / 10.0, -1.0, 1.0)
            fitness += draw_bonus * (0.5 + 0.5 * mat_adv)
        elif (result == 1 and color == 0) or (result == -1 and color == 1):
            fitness += win_bonus
            fitness += checkmate_bonus
        else:
            fitness += loss_penalty

    fitness += (my_material - opp_material) * material_weight
    if not is_game_over:
        fitness += my_mobility * mobility_weight
    fitness += my_king_safety * king_safety_weight
    fitness += move_count * move_count_bonus
    return fitness
