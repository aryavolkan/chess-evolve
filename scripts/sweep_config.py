"""Sweep configuration for Chess-Evolve (Python dict form)."""

sweep_config = {
    "method": "bayes",
    "metric": {"name": "best_fitness", "goal": "maximize"},
    "parameters": {
        "population_size": {"values": [20, 50, 100]},
        "hidden_size": {"values": [32, 64, 128]},
        "elite_count": {"values": [2, 3, 5]},
        "games_per_individual": {"values": [3, 5, 10]},
        "max_moves_per_game": {"values": [80, 100, 150]},
        "mutation_rate": {"distribution": "uniform", "min": 0.05, "max": 0.30},
        "mutation_strength": {"distribution": "uniform", "min": 0.10, "max": 0.50},
        "crossover_rate": {"distribution": "uniform", "min": 0.50, "max": 0.90},
        "max_generations": {"value": 50},
        "use_tournament": {"value": True},
        "tournament_opponents": {"values": [3, 5, 7]},
        "tournament_mode": {"values": ["round_robin", "swiss"]},
    },
}
