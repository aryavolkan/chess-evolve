---
description: Analyze games or positions using Lichess cloud eval and opening explorer
argument-hint: [game_id_or_fen]
allowed-tools: [Bash, Read]
---

Analyze a Lichess game or chess position using cloud evaluation and the opening explorer.

**Arguments** (optional):
- A Lichess game ID (e.g., `abcd1234`)
- A FEN string (e.g., `rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1`)
- `last` or omitted: analyze the most recent bot game

**Steps:**

1. Determine input type (game ID, FEN, or "last"):
   - If `last` or no argument, fetch the most recent game:
     ```
     /home/aryasen/evolve/.venv/bin/python3 -c "
     import os, berserk
     session = berserk.TokenSession(os.environ['LICHESS_TOKEN'])
     client = berserk.Client(session)
     bot_id = client.account.get()['id']
     games = list(client.games.export_by_player(bot_id, max=1, as_pgn=False))
     if games:
         print(games[0]['id'])
     "
     ```

2. For a **game ID**, export the PGN and get cloud evals:
   ```
   /home/aryasen/evolve/.venv/bin/python3 -c "
   import os, berserk, chess.pgn, io
   session = berserk.TokenSession(os.environ['LICHESS_TOKEN'])
   client = berserk.Client(session)
   game = client.games.export('<GAME_ID>', as_pgn=True, evals=True, clocks=True)
   print(game)
   "
   ```

3. For a **FEN string**, query the Lichess cloud eval and opening explorer:
   ```
   /home/aryasen/evolve/.venv/bin/python3 -c "
   import urllib.request, json
   fen = '<FEN>'
   # Cloud eval
   url = f'https://lichess.org/api/cloud-eval?fen={fen.replace(\" \", \"%20\")}&multiPv=3'
   resp = json.loads(urllib.request.urlopen(url).read())
   for pv in resp.get('pvs', []):
       cp = pv.get('cp', pv.get('mate', '?'))
       print(f\"  eval: {cp}  line: {pv.get(\"moves\", \"\")[:40]}\")
   # Opening explorer
   url2 = f'https://lichess.org/api/opening-explorer/lichess?fen={fen.replace(\" \", \"%20\")}'
   resp2 = json.loads(urllib.request.urlopen(url2).read())
   for m in resp2.get('moves', [])[:5]:
       total = m['white'] + m['draws'] + m['black']
       print(f\"  {m['uci']}: {total} games, white {m['white']/total*100:.0f}%\")
   "
   ```

4. Summarize findings: key positions, eval swings (blunders), opening name, and overall assessment of the genome's play quality.
