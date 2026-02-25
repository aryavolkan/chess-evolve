# Chess-Evolve W&B Sweep Configuration
# Optimizes combined_best = min(white_best, black_best) to ensure balanced play.
# Search space narrowed based on sweep uq8yh9ck analysis:
#   - elite_count=2 strongly favors breakout runs
#   - hidden_size 128 clustered in mediocre zone
#   - tournament_opponents=5 slight edge
#   - max_generations reduced from 100→25 (convergence by gen 9)

sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'combined_best', 'goal': 'maximize'},
    'parameters': {
        # Population size
        'population_size': {'values': [30, 50]},

        # Elite count - fixed at 2 (strongest signal from data)
        'elite_count': {'value': 2},

        # Mutation parameters
        'mutation_rate': {'distribution': 'uniform', 'min': 0.15, 'max': 0.40},
        'mutation_strength': {'distribution': 'uniform', 'min': 0.05, 'max': 0.20},

        # Crossover rate
        'crossover_rate': {'distribution': 'uniform', 'min': 0.60, 'max': 0.85},

        # Network architecture (input=389, output=4096 fixed)
        'hidden_size': {'values': [32, 64]},

        # Training parameters
        'max_generations': {'value': 25},
        'games_per_individual': {'values': [2, 3]},
        'tournament_opponents': {'value': 5},
        'use_tournament': {'value': True},
        'tournament_mode': {'value': 'round_robin'},
        'max_moves_per_game': {'value': 100},

        # Diversity: immigration, fitness sharing, reduced selection pressure
        'immigration_rate': {'distribution': 'uniform', 'min': 0.05, 'max': 0.20},
        'fitness_sharing_sigma': {'distribution': 'uniform', 'min': 0.03, 'max': 0.15},
        'tournament_k': {'value': 2},  # k=2 reduces selection pressure vs default k=3
    }
}

# Alternative: Quick validation sweep (fewer params, faster iteration)
sweep_config_quick = {
    'method': 'grid',
    'metric': {'name': 'combined_best', 'goal': 'maximize'},
    'parameters': {
        'population_size': {'values': [30, 50]},
        'elite_count': {'value': 2},
        'mutation_rate': {'values': [0.20, 0.30]},
        'mutation_strength': {'values': [0.10, 0.15]},
        'crossover_rate': {'values': [0.70]},
        'hidden_size': {'values': [32, 64]},
        'max_generations': {'value': 15},
        'games_per_individual': {'value': 2},
        'tournament_opponents': {'value': 5},
        'use_tournament': {'value': True},
        'tournament_mode': {'value': 'round_robin'},
        'max_moves_per_game': {'value': 100},
        'immigration_rate': {'values': [0.10, 0.15]},
        'fitness_sharing_sigma': {'values': [0.05, 0.10]},
        'tournament_k': {'value': 2},
    }
}

# Alternative: Deep exploration (more compute, find optimal architecture)
sweep_config_deep = {
    'method': 'bayes',
    'metric': {'name': 'combined_best', 'goal': 'maximize'},
    'parameters': {
        'population_size': {'values': [30, 50, 100]},
        'elite_count': {'values': [2, 3]},
        'mutation_rate': {'distribution': 'uniform', 'min': 0.10, 'max': 0.50},
        'mutation_strength': {'distribution': 'uniform', 'min': 0.03, 'max': 0.25},
        'crossover_rate': {'distribution': 'uniform', 'min': 0.50, 'max': 0.90},
        'hidden_size': {'values': [32, 64, 96]},
        'max_generations': {'value': 50},
        'games_per_individual': {'values': [2, 3, 4]},
        'tournament_opponents': {'values': [4, 5, 6]},
        'use_tournament': {'value': True},
        'tournament_mode': {'values': ['round_robin', 'swiss']},
        'max_moves_per_game': {'value': 150},
        'immigration_rate': {'distribution': 'uniform', 'min': 0.05, 'max': 0.25},
        'fitness_sharing_sigma': {'distribution': 'uniform', 'min': 0.02, 'max': 0.20},
        'tournament_k': {'values': [2, 3]},
    }
}
