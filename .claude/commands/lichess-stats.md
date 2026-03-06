---
description: Show Lichess bot account rating and game history
allowed-tools: [Bash, Read]
---

Show the Lichess bot account's ratings and recent game history.

**Requirements**: `LICHESS_TOKEN` env var must be set.

**Steps:**

1. Fetch bot profile and ratings:
   ```
   /home/aryasen/evolve/.venv/bin/python3 -c "
   import os, berserk
   session = berserk.TokenSession(os.environ['LICHESS_TOKEN'])
   client = berserk.Client(session)
   profile = client.account.get()
   print(f\"Account: {profile['id']}\")
   perfs = profile.get('perfs', {})
   for tc in ('bullet', 'blitz', 'rapid', 'classical'):
       if tc in perfs:
           p = perfs[tc]
           print(f\"  {tc}: {p.get('rating', '?')} (games: {p.get('games', 0)})\" )
   "
   ```

2. Fetch recent games (last 10):
   ```
   /home/aryasen/evolve/.venv/bin/python3 -c "
   import os, berserk
   session = berserk.TokenSession(os.environ['LICHESS_TOKEN'])
   client = berserk.Client(session)
   bot_id = client.account.get()['id']
   games = list(client.games.export_by_player(bot_id, max=10, as_pgn=False))
   for g in games:
       players = g.get('players', {})
       w = players.get('white', {}).get('user', {}).get('name', '?')
       b = players.get('black', {}).get('user', {}).get('name', '?')
       winner = g.get('winner', 'draw')
       status = g.get('status', '?')
       print(f\"  {w} vs {b}: {winner} ({status}) https://lichess.org/{g['id']}\")
   "
   ```

3. Present a summary table with ratings per time control, win/draw/loss totals, and links to recent games.
