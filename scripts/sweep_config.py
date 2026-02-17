#!/usr/bin/env python3
"""W&B sweep configuration for chess-evolve hyperparameter optimization.

Initial exploration sweep to find good baseline parameters for:
- Population sizes
- Network architecture (hidden layers)
- Mutation parameters
- Fitness evaluation settings
"""

sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'white_best', 'goal': 'maximize'},
    'parameters': {
        # Population - start with moderate sizes
        'population_size': {'values': [20, 30, 40]},
        
        # Network architecture (current default: 389 inputs, 32/128/3 hidden)
        # Vary middle hidden layer sizes
        'hidden_layer_1': {'values': [24, 32, 48]},
        'hidden_layer_2': {'values': [96, 128, 160]},
        
        # Mutation parameters
        'mutation_rate': {'distribution': 'uniform', 'min': 0.1, 'max': 0.4},
        'mutation_strength': {'distribution': 'uniform', 'min': 0.05, 'max': 0.2},
        
        # Fitness evaluation
        'games_per_eval': {'values': [2, 3, 5]},  # Games to evaluate each network
        
        # Training duration
        'max_generations': {'value': 30},  # Keep it short for initial experiments
        
        # Elite count (proportion of population)
        'elite_count': {'values': [3, 5, 8]},
    }
}

# Alternative: Quick grid search for proof-of-concept
sweep_config_quick = {
    'method': 'grid',
    'metric': {'name': 'white_best', 'goal': 'maximize'},
    'parameters': {
        'population_size': {'values': [30]},
        'hidden_layer_1': {'values': [32]},
        'hidden_layer_2': {'values': [128]},
        'mutation_rate': {'values': [0.2, 0.3]},
        'mutation_strength': {'values': [0.1, 0.15]},
        'games_per_eval': {'values': [3]},
        'max_generations': {'value': 20},
        'elite_count': {'values': [5]},
    }
}

if __name__ == "__main__":
    import wandb
    
    # Use the quick config for initial validation
    sweep_id = wandb.sweep(sweep_config_quick, project='chess-evolve')
    print(f'Created sweep: {sweep_id}')
    print(f'View at: https://wandb.ai/{wandb.api.default_entity}/chess-evolve/sweeps/{sweep_id}')
    print('\nStart workers with:')
    print(f'  wandb agent {wandb.api.default_entity}/chess-evolve/{sweep_id}')
