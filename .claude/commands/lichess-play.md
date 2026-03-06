---
description: Play evolved genome on Lichess as a bot
argument-hint: [color] [num_games] [opponent]
allowed-tools: [Bash, Read]
---

Play the best evolved NEAT genome on Lichess as a bot account.

**Requirements**: `LICHESS_TOKEN` env var must be set with a Bot API token from https://lichess.org/account/oauth/token

**Arguments** (all optional, space-separated):
- `color`: `white` or `black` genome to load (default: white)
- `num_games`: number of games to play before stopping (default: unlimited)
- `opponent`: challenge this user/bot instead of waiting for challenges

**Steps:**

1. Verify dependencies are available:
   ```
   /home/aryasen/evolve/.venv/bin/python3 -c "import berserk, chess; print('ok')"
   ```

2. Verify genome loads successfully:
   ```
   /home/aryasen/evolve/.venv/bin/python3 lichess_bot.py --test --color <color>
   ```

3. Parse the user's arguments to determine color, num_games, and opponent. Build the command:
   ```
   /home/aryasen/evolve/.venv/bin/python3 lichess_bot.py --color <color> [--games <N>] [--challenge <opponent>]
   ```

4. Run the bot and report results: win/loss/draw record, rating changes, and game links (https://lichess.org/<game_id>).
