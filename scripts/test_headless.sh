#!/usr/bin/env bash
# Quick test of headless training mode (no W&B)

set -e

PROJECT_PATH="$HOME/Projects/chess-evolve"
GODOT_PATH="/opt/homebrew/bin/godot"
GODOT_USER_DIR="$HOME/Library/Application Support/Godot/app_userdata/Chess Evolve"
METRICS_PATH="$GODOT_USER_DIR/metrics.json"
CONFIG_PATH="$GODOT_USER_DIR/sweep_config.json"

# Create test config (small run)
mkdir -p "$GODOT_USER_DIR"
cat > "$CONFIG_PATH" << EOF
{
  "population_size": 10,
  "input_size": 389,
  "hidden_size": 32,
  "output_size": 128,
  "elite_count": 2,
  "games_per_individual": 2,
  "max_moves_per_game": 50,
  "max_generations": 3,
  "mutation_rate": 0.15,
  "mutation_strength": 0.3
}
EOF

echo "✓ Test config written"

# Clear old metrics
rm -f "$METRICS_PATH"

# Run headless
echo "🚀 Launching headless test (3 generations)..."
"$GODOT_PATH" --path "$PROJECT_PATH" --headless --rendering-driver dummy -- --auto-train

echo ""
echo "📊 Final metrics:"
if [ -f "$METRICS_PATH" ]; then
    cat "$METRICS_PATH" | python3 -m json.tool
else
    echo "❌ No metrics file generated"
    exit 1
fi

echo ""
echo "✅ Headless mode working!"
