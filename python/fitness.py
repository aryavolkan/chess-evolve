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


def merge_fitness_weights(config: dict) -> dict:
    """Build fitness weights dict from FITNESS_DEFAULTS, overridden by config."""
    weights = dict(FITNESS_DEFAULTS)
    for key in FITNESS_DEFAULTS:
        if key in config:
            weights[key] = config[key]
    return weights


def blend_fitness(
    base: list[float], other: list[float], weight: float,
) -> list[float]:
    """Blend two fitness lists: (1-weight)*base + weight*other."""
    w1 = 1.0 - weight
    return [w1 * b + weight * o for b, o in zip(base, other, strict=True)]


def _extract_game_data(game: dict, color: int) -> tuple:
    """Extract per-color game data. Returns (idx, my_mat, opp_mat, my_ks, opp_ks,
    my_mob, opp_mob, my_kd, my_cap, result, move_count, is_win, is_loss)."""
    if color == 0:
        idx = game["white_idx"]
        my_mat = game["white_material"]
        opp_mat = game["black_material"]
        my_ks = game["white_king_safety"]
        opp_ks = game["black_king_safety"]
        my_mob = game["white_mobility"]
        opp_mob = game["black_mobility"]
        my_kd = game.get("white_king_danger", 0.0)
        my_cap = game.get("white_captures_value", 0.0)
    else:
        idx = game["black_idx"]
        my_mat = game["black_material"]
        opp_mat = game["white_material"]
        my_ks = game["black_king_safety"]
        opp_ks = game["white_king_safety"]
        my_mob = game["black_mobility"]
        opp_mob = game["white_mobility"]
        my_kd = game.get("black_king_danger", 0.0)
        my_cap = game.get("black_captures_value", 0.0)

    result = game["result"]
    move_count = game["move_count"]
    is_win = (result == 1 and color == 0) or (result == -1 and color == 1)
    is_loss = (result == -1 and color == 0) or (result == 1 and color == 1)

    return (idx, my_mat, opp_mat, my_ks, opp_ks, my_mob, opp_mob,
            my_kd, my_cap, result, move_count, is_win, is_loss)


def _score_game(
    my_mat: float, opp_mat: float, my_ks: float, opp_ks: float,
    my_mob: float, opp_mob: float, my_kd: float, my_cap: float,
    result: int, move_count: int, is_win: bool, is_loss: bool,
    w: dict,
) -> tuple[float, dict[str, float]]:
    """Score a single game. Returns (total_fitness, component_breakdown)."""
    # Outcome
    if result == 2:
        mat_adv = max(-1.0, min(1.0, (my_mat - opp_mat) / 10.0))
        outcome = w["draw_bonus"] * (0.5 + 0.5 * mat_adv)
    elif is_win:
        outcome = w["win_bonus"] + w["checkmate_bonus"]
    elif is_loss:
        outcome = w["loss_penalty"]
    else:
        outcome = 0.0

    material = (my_mat - opp_mat) * w["material_weight"]
    mobility = (my_mob - opp_mob) * w["mobility_weight"]
    king_safety = my_ks * w["king_safety_weight"]
    opp_king_safety = -opp_ks * w["opp_king_safety_weight"]
    king_danger = my_kd * w["king_danger_weight"]
    captures = my_cap * w["capture_weight"]
    move_penalty = move_count * w["move_count_penalty"]

    total = outcome + material + mobility + king_safety + opp_king_safety + king_danger + captures + move_penalty

    breakdown = {
        "outcome": outcome, "material": material, "mobility": mobility,
        "king_safety": king_safety, "opp_king_safety": opp_king_safety,
        "king_danger": king_danger, "captures": captures, "move_penalty": move_penalty,
    }
    return total, breakdown


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

    for game in results:
        (idx, my_mat, opp_mat, my_ks, opp_ks, my_mob, opp_mob,
         my_kd, my_cap, result, move_count, is_win, is_loss) = _extract_game_data(game, color)

        f, _ = _score_game(
            my_mat, opp_mat, my_ks, opp_ks, my_mob, opp_mob,
            my_kd, my_cap, result, move_count, is_win, is_loss, weights,
        )

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
        (_, my_mat, opp_mat, my_ks, opp_ks, my_mob, opp_mob,
         my_kd, my_cap, result, move_count, is_win, is_loss) = _extract_game_data(game, color)

        _, breakdown = _score_game(
            my_mat, opp_mat, my_ks, opp_ks, my_mob, opp_mob,
            my_kd, my_cap, result, move_count, is_win, is_loss, weights,
        )

        for k, v in breakdown.items():
            totals[k] += v

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


def compute_all_stats(
    results: list[dict],
    pop_size: int,
    color: int,
    weights: dict,
) -> dict:
    """Single-pass computation of fitness, tournament scores, outcome rates, and game stats.

    Replaces 4-6 separate passes over the results list with one combined pass.

    Returns dict with keys:
        fitness: list[float]
        tournament_scores: list[float]
        outcome_rates: (win_rate, draw_rate, loss_rate)
        game_stats: (total_moves, white_material_avg, black_material_avg)
        fitness_breakdown: dict[str, float]
    """
    fitness = [0.0] * pop_size
    game_counts = [0] * pop_size
    scores = [0.0] * pop_size
    score_counts = [0] * pop_size

    wins = draws = losses = 0
    total_moves = 0
    w_mat_sum = 0.0
    b_mat_sum = 0.0

    breakdown_totals = {
        "outcome": 0.0, "material": 0.0, "mobility": 0.0,
        "king_safety": 0.0, "opp_king_safety": 0.0,
        "king_danger": 0.0, "captures": 0.0, "move_penalty": 0.0,
    }

    for game in results:
        (idx, my_mat, opp_mat, my_ks, opp_ks, my_mob, opp_mob,
         my_kd, my_cap, result, move_count, is_win, is_loss) = _extract_game_data(game, color)

        f, breakdown = _score_game(
            my_mat, opp_mat, my_ks, opp_ks, my_mob, opp_mob,
            my_kd, my_cap, result, move_count, is_win, is_loss, weights,
        )

        # Fitness accumulation
        fitness[idx] += f
        game_counts[idx] += 1

        # Tournament scores
        if is_win:
            scores[idx] += 1.0
        elif result == 2:
            mat_bonus = max(-0.25, min(0.25, (my_mat - opp_mat) / 40.0))
            scores[idx] += 0.5 + mat_bonus
        score_counts[idx] += 1

        # Outcome rates
        if is_win:
            wins += 1
        elif result == 2:
            draws += 1
        elif is_loss:
            losses += 1

        # Game stats
        total_moves += move_count
        w_mat_sum += game["white_material"]
        b_mat_sum += game["black_material"]

        # Breakdown
        for k, v in breakdown.items():
            breakdown_totals[k] += v

    # Normalize
    for i in range(pop_size):
        if game_counts[i] > 0:
            fitness[i] /= game_counts[i]
        if score_counts[i] > 0:
            scores[i] /= score_counts[i]

    n = max(1, len(results))
    total_outcomes = max(1, wins + draws + losses)

    return {
        "fitness": fitness,
        "tournament_scores": scores,
        "outcome_rates": (wins / total_outcomes, draws / total_outcomes, losses / total_outcomes),
        "game_stats": (total_moves, w_mat_sum / n, b_mat_sum / n),
        "fitness_breakdown": {k: v / n for k, v in breakdown_totals.items()},
    }
