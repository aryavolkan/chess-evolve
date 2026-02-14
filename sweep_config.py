# Chess-Evolve W&B Sweep Configuration
# First exploratory sweep to find good hyperparameters for chess neural network evolution

sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'white_best', 'goal': 'maximize'},
    'parameters': {
        # Population size - chess is complex, start moderate
        'population_size': {'values': [20, 30, 50]},

        # Elite count - how many best networks to keep
        'elite_count': {'values': [2, 3, 5]},

        # Mutation parameters
        'mutation_rate': {'distribution': 'uniform', 'min': 0.15, 'max': 0.40},
        'mutation_strength': {'distribution': 'uniform', 'min': 0.05, 'max': 0.20},

        # Crossover rate
        'crossover_rate': {'distribution': 'uniform', 'min': 0.60, 'max': 0.85},

        # Network architecture (input=389, output=128 fixed)
        'hidden_size': {'values': [32, 64, 96, 128]},

        # Training parameters
        'max_generations': {'value': 100},
        'games_per_individual': {'values': [2, 3]},  # Games against random opponents
        'max_moves_per_game': {'value': 100},  # Move limit per game
    }
}

# Alternative: Quick validation sweep (fewer params, faster iteration)
sweep_config_quick = {
    'method': 'grid',
    'metric': {'name': 'white_best', 'goal': 'maximize'},
    'parameters': {
        'population_size': {'values': [30, 50]},
        'elite_count': {'values': [3, 5]},
        'mutation_rate': {'values': [0.20, 0.30]},
        'mutation_strength': {'values': [0.10, 0.15]},
        'crossover_rate': {'values': [0.70]},
        'hidden_size': {'values': [64, 96]},
        'max_generations': {'value': 50},
        'games_per_individual': {'value': 2},
        'max_moves_per_game': {'value': 100},
    }
}

# Alternative: Deep exploration (more compute, find optimal architecture)
sweep_config_deep = {
    'method': 'bayes',
    'metric': {'name': 'white_best', 'goal': 'maximize'},
    'parameters': {
        'population_size': {'values': [30, 60, 100]},
        'elite_count': {'values': [3, 5, 10]},
        'mutation_rate': {'distribution': 'uniform', 'min': 0.10, 'max': 0.50},
        'mutation_strength': {'distribution': 'uniform', 'min': 0.03, 'max': 0.25},
        'crossover_rate': {'distribution': 'uniform', 'min': 0.50, 'max': 0.90},
        'hidden_size': {'values': [64, 96, 128, 192]},
        'max_generations': {'value': 200},
        'games_per_individual': {'values': [2, 3, 4]},
        'max_moves_per_game': {'value': 150},
    }
}
