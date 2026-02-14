#!/bin/bash
# Quick test: 10-generation training run to verify W&B integration works

echo "🧪 Testing Chess-Evolve W&B training (10 generations)"
echo ""

# Create minimal test config
cat > test_config.json <<EOF
{
  "population_size": 20,
  "hidden_size": 32,
  "elite_count": 2,
  "mutation_rate": 0.25,
  "mutation_strength": 0.12,
  "crossover_rate": 0.70,
  "games_per_individual": 2,
  "max_generations": 10,
  "max_moves_per_game": 100,
  "input_size": 389,
  "output_size": 128
}
EOF

echo "✓ Created test config (10 gens, pop=20, small network)"
echo ""

# Run test
python3 train_wandb.py --config test_config.json

echo ""
echo "✅ Test complete! Check W&B: https://wandb.ai/aryavolkan-personal/chess-evolve"
