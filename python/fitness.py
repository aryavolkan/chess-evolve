"""Shared fitness computation for chess neuroevolution trainers.

Used by both CPUTrainer (fixed-topology) and NeatCPUTrainer (NEAT).
"""

# Default fitness weights — tuned to prioritize winning over material hoarding.
# Win + checkmate bonus (20.0) dominates material signal (~10 for 10-pt lead).
FITNESS_DEFAULTS = {
    "win_bonus": 10.0,
    "draw_bonus": 3.0,
    "loss_penalty": -5.0,
    "capture_weight": 0.2,
    "material_weight": 1.0,
    "mobility_weight": 0.3,
    "king_safety_weight": 0.5,
    "opp_king_safety_weight": 0.0,
    "king_danger_weight": 1.0,
    "move_count_penalty": -0.002,
    "checkmate_bonus": 10.0,
}


def compute_fitness(
    results: list[dict],
    pop_size: int,
    color: int,
    weights: dict,
) -> list[float]:
    """Compute fitness for each individual of the given color.

    Args:
        results: List of game result dicts from chess_cpu simulation.
        pop_size: Population size.
        color: 0 for white, 1 for black.
        weights: Dict with fitness weight keys (see FITNESS_DEFAULTS).

    Returns:
        List of average fitness per individual.
    """
    fitness = [0.0] * pop_size
    game_counts = [0] * pop_size

    win_bonus = weights["win_bonus"]
    draw_bonus = weights["draw_bonus"]
    loss_penalty = weights["loss_penalty"]
    checkmate_bonus = weights["checkmate_bonus"]
    material_weight = weights["material_weight"]
    mobility_weight = weights["mobility_weight"]
    king_safety_weight = weights["king_safety_weight"]
    opp_king_safety_weight = weights["opp_king_safety_weight"]
    king_danger_weight = weights["king_danger_weight"]
    capture_weight = weights["capture_weight"]
    move_count_penalty = weights["move_count_penalty"]

    for game in results:
        if color == 0:
            idx = game["white_idx"]
            my_material = game["white_material"]
            opp_material = game["black_material"]
            my_king_safety = game["white_king_safety"]
            opp_king_safety = game["black_king_safety"]
            my_mobility = game["white_mobility"]
            opp_mobility = game["black_mobility"]
            my_king_danger = game.get("white_king_danger", 0.0)
            my_captures = game.get("white_captures_value", 0.0)
        else:
            idx = game["black_idx"]
            my_material = game["black_material"]
            opp_material = game["white_material"]
            my_king_safety = game["black_king_safety"]
            opp_king_safety = game["white_king_safety"]
            my_mobility = game["black_mobility"]
            opp_mobility = game["white_mobility"]
            my_king_danger = game.get("black_king_danger", 0.0)
            my_captures = game.get("black_captures_value", 0.0)

        result = game["result"]
        move_count = game["move_count"]
        f = 0.0

        is_win = (result == 1 and color == 0) or (result == -1 and color == 1)
        is_loss = (result == -1 and color == 0) or (result == 1 and color == 1)

        if result == 2:
            mat_adv = max(-1.0, min(1.0, (my_material - opp_material) / 10.0))
            f += draw_bonus * (0.5 + 0.5 * mat_adv)
        elif is_win:
            f += win_bonus + checkmate_bonus
        elif is_loss:
            f += loss_penalty

        f += (my_material - opp_material) * material_weight
        f += (my_mobility - opp_mobility) * mobility_weight
        f += my_king_safety * king_safety_weight
        f -= opp_king_safety * opp_king_safety_weight
        f += my_king_danger * king_danger_weight
        f += my_captures * capture_weight
        f += move_count * move_count_penalty

        fitness[idx] += f
        game_counts[idx] += 1

    for i in range(pop_size):
        if game_counts[i] > 0:
            fitness[i] /= game_counts[i]

    return fitness


def compute_fitness_breakdown(
    results: list[dict],
    color: int,
    weights: dict,
) -> dict[str, float]:
    """Compute average contribution of each fitness component across all games."""
    totals = {
        "outcome": 0.0, "material": 0.0, "mobility": 0.0,
        "king_safety": 0.0, "opp_king_safety": 0.0,
        "king_danger": 0.0, "captures": 0.0, "move_penalty": 0.0,
    }
    n = len(results)
    if n == 0:
        return totals

    for game in results:
        if color == 0:
            my_mat = game["white_material"]
            opp_mat = game["black_material"]
            my_ks = game["white_king_safety"]
            opp_ks = game["black_king_safety"]
            my_mob = game["white_mobility"]
            opp_mob = game["black_mobility"]
        else:
            my_mat = game["black_material"]
            opp_mat = game["white_material"]
            my_ks = game["black_king_safety"]
            opp_ks = game["white_king_safety"]
            my_mob = game["black_mobility"]
            opp_mob = game["white_mobility"]

        result = game["result"]
        is_win = (result == 1 and color == 0) or (result == -1 and color == 1)
        is_loss = (result == -1 and color == 0) or (result == 1 and color == 1)

        if result == 2:
            mat_adv = max(-1.0, min(1.0, (my_mat - opp_mat) / 10.0))
            totals["outcome"] += weights["draw_bonus"] * (0.5 + 0.5 * mat_adv)
        elif is_win:
            totals["outcome"] += weights["win_bonus"] + weights["checkmate_bonus"]
        elif is_loss:
            totals["outcome"] += weights["loss_penalty"]

        totals["material"] += (my_mat - opp_mat) * weights["material_weight"]
        totals["mobility"] += (my_mob - opp_mob) * weights["mobility_weight"]
        totals["king_safety"] += my_ks * weights["king_safety_weight"]
        totals["opp_king_safety"] -= opp_ks * weights["opp_king_safety_weight"]
        kd_key = "white_king_danger" if color == 0 else "black_king_danger"
        totals["king_danger"] += game.get(kd_key, 0.0) * weights["king_danger_weight"]
        cap_key = "white_captures_value" if color == 0 else "black_captures_value"
        totals["captures"] += game.get(cap_key, 0.0) * weights["capture_weight"]
        totals["move_penalty"] += game["move_count"] * weights["move_count_penalty"]

    return {k: v / n for k, v in totals.items()}


def compute_outcome_rates(
    results: list[dict], color: int,
) -> tuple[float, float, float]:
    """Compute win/draw/loss rates for a color."""
    wins = draws = losses = 0
    for game in results:
        r = game["result"]
        is_win = (r == 1 and color == 0) or (r == -1 and color == 1)
        is_loss = (r == -1 and color == 0) or (r == 1 and color == 1)
        if is_win:
            wins += 1
        elif r == 2:
            draws += 1
        elif is_loss:
            losses += 1
    total = max(1, wins + draws + losses)
    return wins / total, draws / total, losses / total


def compute_tournament_scores(
    results: list[dict],
    pop_size: int,
    color: int,
) -> list[float]:
    """Compute tournament scores: 1.0 for win, 0.5+material_bonus for draw, 0.0 for loss."""
    scores = [0.0] * pop_size
    counts = [0] * pop_size

    for game in results:
        if color == 0:
            idx = game["white_idx"]
            my_mat = game["white_material"]
            opp_mat = game["black_material"]
        else:
            idx = game["black_idx"]
            my_mat = game["black_material"]
            opp_mat = game["white_material"]

        result = game["result"]
        is_win = (result == 1 and color == 0) or (result == -1 and color == 1)

        if is_win:
            scores[idx] += 1.0
        elif result == 2:
            mat_bonus = max(-0.25, min(0.25, (my_mat - opp_mat) / 40.0))
            scores[idx] += 0.5 + mat_bonus

        counts[idx] += 1

    for i in range(pop_size):
        if counts[i] > 0:
            scores[i] /= counts[i]

    return scores


def aggregate_game_stats(
    results: list[dict],
) -> tuple[int, float, float]:
    """Compute total moves, white material avg, black material avg in a single pass.

    Returns:
        (total_moves, white_material_avg, black_material_avg)
    """
    total_moves = 0
    w_mat_sum = 0.0
    b_mat_sum = 0.0
    for g in results:
        total_moves += g["move_count"]
        w_mat_sum += g["white_material"]
        b_mat_sum += g["black_material"]
    n = max(1, len(results))
    return total_moves, w_mat_sum / n, b_mat_sum / n
