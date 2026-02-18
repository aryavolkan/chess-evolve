# NEAT Chess Evolution Improvements

This document summarizes the major improvements made to the chess-evolve project to enhance NEAT-based self-play training.

## Change 1: Minimax Wrapper (Value Network + Alpha-Beta Search)

### What was implemented:
- Added `ai/minimax_player.gd`: A new player class that uses minimax search with alpha-beta pruning
- The neural network now serves as a position evaluation function instead of directly choosing moves
- Configurable search depth (default: 2, can be increased to 3 for stronger play)
- The existing policy network architecture (389→hidden→128) is adapted to output position scores by summing the outputs

### Design decisions:
- Instead of modifying the network architecture to output a single value, we interpret the sum of the 128 outputs as a position score
- This maintains backward compatibility with existing networks while transitioning to search-based play
- The minimax player properly handles terminal states (checkmate, stalemate) with appropriate scores

### Configuration in `training_manager.gd`:
```gdscript
var use_minimax: bool = true  # Enable/disable minimax search
var minimax_depth: int = 2    # Search depth (2-3 recommended)
```

## Change 2: Hall of Fame Coevolution

### What was implemented:
- Added Hall of Fame arrays in `ai/evolution.gd` to maintain the best individuals from past generations
- Maximum 20 individuals per Hall of Fame (configurable via `HALL_OF_FAME_SIZE`)
- After each generation, the best white and black individuals are added to their respective Hall of Fame
- During fitness evaluation, individuals play against both current population members and Hall of Fame members
- Hall of Fame opponents are frozen (not mutated) to provide stable benchmarks

### Key features:
- Prevents cycling in coevolution by maintaining diverse historical opponents
- Both white and black populations play as primary players to ensure symmetric evaluation
- Configurable ratio of games against Hall of Fame vs current population

### Configuration in `training_manager.gd`:
```gdscript
var hall_of_fame_ratio: float = 0.5  # 50% of games against Hall of Fame
```

## Testing Instructions

1. **Test Minimax Player**:
   ```bash
   # Run with minimax enabled (default)
   godot --headless --script res://test/test_integration.gd
   
   # Compare with direct network output
   # Set use_minimax = false in training_manager.gd, then run again
   ```

2. **Test Hall of Fame**:
   ```bash
   # Run training for several generations
   python3 train_local.py --generations 10 --population 20
   
   # Check that Hall of Fame is being populated
   # Look for improved stability in fitness scores over generations
   ```

3. **Test Combined Features**:
   ```bash
   # Run a full training session with both features enabled
   python3 train_wandb.py --config configs/default_config.yaml
   
   # Monitor:
   # - Search depth impact on game quality
   # - Hall of Fame size growth
   # - Fitness progression stability
   ```

4. **Verify Backward Compatibility**:
   - Existing training scripts (train_wandb.py) should work without modification
   - Set `use_minimax = false` and `hall_of_fame_ratio = 0.0` to get original behavior

## Expected Improvements

1. **With Minimax Search**:
   - More strategic play (looking ahead 2-3 moves)
   - Better tactical awareness (avoiding simple blunders)
   - Slightly longer games due to better defense

2. **With Hall of Fame**:
   - More stable fitness progression (less cycling)
   - Better generalization (playing against diverse opponents)
   - Preservation of good strategies discovered in earlier generations

## Performance Considerations

- Minimax search increases computation time per move (roughly 20-50x for depth 2)
- Consider reducing population size or games per individual if training is too slow
- Hall of Fame has minimal performance impact (just stores best networks)

## Change 3: Round-Robin Tournament Selection

### What was implemented:
- Added tournament-based fitness evaluation system in `ai/training_manager.gd`
- Individuals now play against a structured set of opponents instead of random selection
- Tournament results (wins/draws/losses) determine fitness scores
- Support for both round-robin and Swiss-system tournaments

### Key features:

1. **Round-Robin Mode**:
   - Each individual plays against K opponents selected from different fitness quintiles
   - Ensures diverse matchups between strong and weak agents
   - Configurable number of opponents per individual (default: 4-6)
   - Final fitness = tournament score (1pt win, 0.5pt draw, 0pt loss) + 10% material/position bonus

2. **Swiss-System Mode**:
   - After round 1, pairs individuals with similar scores
   - More efficient for large populations than full round-robin
   - Reduces total game count while maintaining accuracy

3. **Backward Compatibility**:
   - `use_tournament = false` flag to revert to original random opponent selection
   - Existing training scripts work without modification

### Configuration:
```gdscript
# In training_manager.gd:
var use_tournament: bool = true              # Enable tournament mode
var tournament_mode: String = "round_robin"  # "round_robin" or "swiss"
var tournament_opponents: int = 4            # Opponents per individual

# In sweep_config.py:
'tournament_opponents': {'values': [4, 5, 6]},
'use_tournament': {'value': True},
'tournament_mode': {'value': 'round_robin'},
```

## Testing Instructions for Tournament Mode

1. **Test Round-Robin Tournament**:
   ```bash
   # Create a test config with tournament enabled
   cat > test_tournament.json << EOF
   {
     "population_size": 10,
     "hidden_size": 64,
     "use_tournament": true,
     "tournament_mode": "round_robin",
     "tournament_opponents": 4,
     "max_generations": 5
   }
   EOF
   
   # Run training with tournament mode
   python3 train_wandb.py --config test_tournament.json
   ```

2. **Test Swiss-System Tournament**:
   ```bash
   # Modify config for Swiss mode
   cat > test_swiss.json << EOF
   {
     "population_size": 20,
     "hidden_size": 64,
     "use_tournament": true,
     "tournament_mode": "swiss",
     "tournament_opponents": 4,
     "max_generations": 5
   }
   EOF
   
   python3 train_wandb.py --config test_swiss.json
   ```

3. **Compare with Random Selection**:
   ```bash
   # Run with tournaments disabled
   cat > test_random.json << EOF
   {
     "population_size": 10,
     "hidden_size": 64,
     "use_tournament": false,
     "games_per_individual": 4,
     "max_generations": 5
   }
   EOF
   
   python3 train_wandb.py --config test_random.json
   ```

## Expected Improvements with Tournaments

1. **More Reliable Fitness Rankings**:
   - Each individual's fitness is based on performance against multiple diverse opponents
   - Reduces variance from lucky/unlucky random pairings
   - Better identification of truly strong individuals

2. **Balanced Competition**:
   - Quintile-based pairing ensures weak and strong agents both get appropriate challenges
   - Prevents situations where weak agents only play other weak agents

3. **Tournament Metrics**:
   - Clear win/draw/loss records provide interpretable progress tracking
   - Swiss system adapts difficulty as the tournament progresses