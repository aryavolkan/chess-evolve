#!/usr/bin/env python3
"""Lichess bot engine for evolved NEAT chess networks.

Loads a NEAT genome, connects to Lichess via the Bot API, and plays games.
Uses python-chess for board state and berserk for Lichess streaming.

Usage:
    python lichess_bot.py --test                                # dry-run test
    python lichess_bot.py --color white --games 5               # accept 5 challenges
    python lichess_bot.py --challenge stockfish_level_1 --games 3  # challenge a bot
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict

import berserk
import chess

# Try to import chess_cpu for board encoding; fall back to pure Python
try:
    import chess_cpu

    HAS_CHESS_CPU = True
except ImportError:
    HAS_CHESS_CPU = False

DEFAULT_GENOME_PATH = "neat_best_genomes.json"
INPUT_SIZE = 389
DEFAULT_OUTPUT_SIZE = 128


# ---------------------------------------------------------------------------
# Pure-Python NEAT sparse network (mirrors rust/chess-cpu/src/sparse_nn.rs)
# ---------------------------------------------------------------------------

class SparseNetwork:
    """Compiled NEAT network with topological-order forward pass."""

    def __init__(self, genome: dict):
        nodes = genome["nodes"]
        connections = genome["connections"]

        # Map node IDs to positions, sorted by (node_type, id)
        sorted_nodes = sorted(nodes, key=lambda n: (n["node_type"], n["id"]))
        self.node_count = len(sorted_nodes)

        max_id = max(n["id"] for n in sorted_nodes) + 1
        self.id_to_pos = [None] * max_id
        self.biases = [0.0] * self.node_count
        self.input_ids = []
        self.output_ids = []

        for pos, node in enumerate(sorted_nodes):
            self.id_to_pos[node["id"]] = pos
            self.biases[pos] = node["bias"]
            if node["node_type"] == 0:
                self.input_ids.append(node["id"])
            elif node["node_type"] == 2:
                self.output_ids.append(node["id"])

        # Build adjacency and incoming edges (enabled connections only)
        in_degree = [0] * self.node_count
        adjacency = defaultdict(list)
        self.incoming_edges = defaultdict(list)  # pos -> [(src_pos, weight)]

        for conn in connections:
            if not conn["enabled"]:
                continue
            src_id, dst_id = conn["in_id"], conn["out_id"]
            if src_id >= max_id or dst_id >= max_id:
                continue
            src_pos = self.id_to_pos[src_id]
            dst_pos = self.id_to_pos[dst_id]
            if src_pos is None or dst_pos is None:
                continue
            in_degree[dst_pos] += 1
            adjacency[src_pos].append(dst_pos)
            self.incoming_edges[dst_pos].append((src_pos, conn["weight"]))

        # Kahn's topological sort
        queue = [p for p in range(self.node_count) if in_degree[p] == 0]
        topo_order = []
        head = 0
        while head < len(queue):
            pos = queue[head]
            head += 1
            topo_order.append(pos)
            for downstream in adjacency[pos]:
                in_degree[downstream] -= 1
                if in_degree[downstream] == 0:
                    queue.append(downstream)

        # Fallback if cycles
        if len(topo_order) < self.node_count:
            topo_order = list(range(self.node_count))

        # eval_order = non-input nodes in topological order
        input_positions = set()
        for pos, node in enumerate(sorted_nodes):
            if node["node_type"] == 0:
                input_positions.add(pos)
        self.eval_order = [p for p in topo_order if p not in input_positions]

    def forward(self, inputs: list[float]) -> list[float]:
        """Run forward pass, return output activations."""
        activations = [0.0] * self.node_count

        # Set input activations
        for i, input_id in enumerate(self.input_ids):
            pos = self.id_to_pos[input_id]
            if pos is not None and i < len(inputs):
                activations[pos] = inputs[i]

        # Evaluate in topological order
        for pos in self.eval_order:
            s = self.biases[pos]
            for src_pos, weight in self.incoming_edges[pos]:
                s += activations[src_pos] * weight
            activations[pos] = math.tanh(s)

        # Collect outputs
        outputs = []
        for out_id in self.output_ids:
            pos = self.id_to_pos[out_id]
            outputs.append(activations[pos] if pos is not None else 0.0)
        return outputs


# ---------------------------------------------------------------------------
# Board encoding (pure Python fallback)
# ---------------------------------------------------------------------------

# Piece type to plane index (0-5): pawn, knight, bishop, rook, queen, king
_PIECE_PLANE = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def encode_board_python(board: chess.Board) -> list[float]:
    """Encode board to 389-float vector matching chess_cpu.encode_board_fen.

    6 planes x 64 squares (signed: +1 white, -1 black, absolute colors),
    plus 1 side-to-move flag (0=white, 1=black),
    plus 4 raw castling bits (WK, WQ, BK, BQ) = 389.
    """
    vec = [0.0] * INPUT_SIZE

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        plane = _PIECE_PLANE[piece.piece_type]
        # Absolute: +1 white, -1 black
        sign = 1.0 if piece.color == chess.WHITE else -1.0
        vec[plane * 64 + sq] = sign

    base = 384
    # side_to_move: 0.0 = white, 1.0 = black
    vec[base] = 0.0 if board.turn == chess.WHITE else 1.0

    # Raw castling bits: WK, WQ, BK, BQ
    vec[base + 1] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    vec[base + 2] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    vec[base + 3] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    vec[base + 4] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    return vec


def encode_board(board: chess.Board) -> list[float]:
    """Encode board position. Uses Rust when available, else pure Python."""
    if HAS_CHESS_CPU:
        return chess_cpu.encode_board_fen(board.fen())
    return encode_board_python(board)


# ---------------------------------------------------------------------------
# Move selection
# ---------------------------------------------------------------------------

def pick_move(network: SparseNetwork, board: chess.Board,
              output_size: int = DEFAULT_OUTPUT_SIZE) -> chess.Move:
    """Select the best legal move using the NEAT network (deterministic argmax)."""
    inputs = encode_board(board)
    outputs = network.forward(inputs)

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        raise ValueError("No legal moves available")

    best_move = None
    best_score = float("-inf")

    for move in legal_moves:
        from_sq = move.from_square
        to_sq = move.to_square

        if output_size <= 128:
            # Factored: score = outputs[from] + outputs[64 + to]
            score = outputs[from_sq] + outputs[64 + to_sq]
        else:
            # Full 4096: score = outputs[from * 64 + to]
            idx = from_sq * 64 + to_sq
            score = outputs[idx] if idx < len(outputs) else 0.0

        if score > best_score:
            best_score = score
            best_move = move

    return best_move


# ---------------------------------------------------------------------------
# NeatLichessBot
# ---------------------------------------------------------------------------

class NeatLichessBot:
    """Lichess bot powered by an evolved NEAT genome."""

    def __init__(self, token: str, genome_json: str,
                 output_size: int = DEFAULT_OUTPUT_SIZE):
        self.output_size = output_size
        self.network = SparseNetwork(json.loads(genome_json))
        self.session = berserk.TokenSession(token)
        self.client = berserk.Client(self.session)
        self.bot_id = self.client.account.get()["id"]

        # Session stats
        self.stats = {"wins": 0, "losses": 0, "draws": 0, "games": []}

    def pick_move(self, board: chess.Board) -> chess.Move:
        return pick_move(self.network, board, self.output_size)

    def play_game(self, game_id: str):
        """Stream a game and respond with moves."""
        print(f"Playing game {game_id}...")
        stream = self.client.bots.stream_game_state(game_id)

        # First event contains full game info
        game_info = next(stream)
        board = chess.Board()

        white_info = game_info.get("white", {})
        white_id = white_info.get("id", white_info.get("name", "")).lower()
        am_white = white_id == self.bot_id

        # Apply initial moves if any
        initial_state = game_info
        if game_info["type"] == "gameFull":
            initial_state = game_info.get("state", game_info)

        moves_str = initial_state.get("moves", "")
        if moves_str:
            for uci in moves_str.split():
                board.push_uci(uci)

        # Make a move if it's our turn
        if self._is_our_turn(board, am_white) and not board.is_game_over():
            move = self.pick_move(board)
            try:
                self.client.bots.make_move(game_id, move.uci())
            except berserk.exceptions.ResponseError:
                pass  # game may have ended or not our turn

        # Stream remaining game state
        for event in stream:
            if event["type"] == "gameState":
                # Rebuild board from full move list
                board = chess.Board()
                moves_str = event.get("moves", "")
                if moves_str:
                    for uci in moves_str.split():
                        board.push_uci(uci)

                status = event.get("status", "started")
                if status not in ("started", "created"):
                    self._record_result(game_id, event, am_white)
                    break

                if self._is_our_turn(board, am_white) and not board.is_game_over():
                    move = self.pick_move(board)
                    try:
                        self.client.bots.make_move(game_id, move.uci())
                    except berserk.exceptions.ResponseError:
                        pass  # game may have ended or not our turn

            elif event["type"] == "gameFull":
                # Shouldn't happen mid-stream, but handle gracefully
                pass

    def _is_our_turn(self, board: chess.Board, am_white: bool) -> bool:
        return (board.turn == chess.WHITE) == am_white

    def _record_result(self, game_id: str, state: dict, am_white: bool):
        status = state.get("status", "unknown")
        winner = state.get("winner")

        if winner is None:
            result = "draw"
            self.stats["draws"] += 1
        elif (winner == "white") == am_white:
            result = "win"
            self.stats["wins"] += 1
        else:
            result = "loss"
            self.stats["losses"] += 1

        self.stats["games"].append({
            "id": game_id, "result": result, "status": status
        })
        total = self.stats["wins"] + self.stats["losses"] + self.stats["draws"]
        print(f"  Game {game_id}: {result} ({status}) "
              f"[{self.stats['wins']}W/{self.stats['draws']}D/{self.stats['losses']}L "
              f"= {total} games]")

    def run(self, accept_challenges: bool = True, max_games: int | None = None,
            color: str | None = None, challenge_opponent: str | None = None):
        """Main event loop. Accept challenges or issue one, then play."""
        games_played = 0

        if challenge_opponent:
            print(f"Challenging {challenge_opponent}...")
            self.challenge_bot(challenge_opponent, color=color)

        print(f"Listening for events as {self.bot_id}...")
        for event in self.client.bots.stream_incoming_events():
            if event["type"] == "challenge":
                challenge = event["challenge"]
                challenger = challenge["challenger"]["id"]
                if challenger == self.bot_id:
                    continue  # skip our own outgoing challenges
                if accept_challenges:
                    print(f"Accepting challenge from {challenger}")
                    try:
                        self.client.bots.accept_challenge(challenge["id"])
                    except berserk.exceptions.ResponseError as e:
                        print(f"  Could not accept: {e}")

            elif event["type"] == "gameStart":
                game_id = event["game"]["gameId"]
                try:
                    self.play_game(game_id)
                except Exception as e:
                    print(f"  Error in game {game_id}: {e}")

                games_played += 1
                if max_games and games_played >= max_games:
                    print(f"Reached {max_games} games, stopping.")
                    break

    def challenge_bot(self, opponent: str, color: str | None = None,
                      clock_limit: int = 300, clock_increment: int = 3,
                      rated: bool = False):
        """Send a challenge to another Lichess bot/user."""
        kwargs = {
            "clock_limit": clock_limit,
            "clock_increment": clock_increment,
        }
        if color:
            kwargs["color"] = color
        self.client.challenges.create(opponent, rated=rated, **kwargs)

    def get_stats(self) -> dict:
        """Return session stats plus current Lichess ratings."""
        try:
            profile = self.client.account.get()
            perfs = profile.get("perfs", {})
            ratings = {}
            for tc in ("bullet", "blitz", "rapid", "classical", "correspondence"):
                if tc in perfs:
                    ratings[tc] = perfs[tc].get("rating", "?")
        except Exception:
            ratings = {}

        return {**self.stats, "ratings": ratings, "bot_id": self.bot_id}


# ---------------------------------------------------------------------------
# Genome loading
# ---------------------------------------------------------------------------

def load_genome(path: str, color: str = "white") -> tuple[str, int]:
    """Load a genome JSON string from the best-genomes file.

    Returns (genome_json_string, output_size).
    """
    with open(path) as f:
        data = json.load(f)

    genome_json = data[color]
    # Detect output_size from genome
    genome = json.loads(genome_json)
    output_count = sum(1 for n in genome["nodes"] if n["node_type"] == 2)
    return genome_json, output_count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Lichess bot powered by evolved NEAT genomes"
    )
    parser.add_argument("--genome", type=str, default=DEFAULT_GENOME_PATH,
                        help="Path to neat_best_genomes.json")
    parser.add_argument("--color", type=str, default="white",
                        choices=["white", "black"],
                        help="Which genome to load (default: white)")
    parser.add_argument("--games", type=int, default=None,
                        help="Max games to play before stopping")
    parser.add_argument("--challenge", type=str, default=None,
                        help="Challenge this user/bot instead of waiting")
    parser.add_argument("--test", action="store_true",
                        help="Dry run: load genome, pick first move, exit")
    args = parser.parse_args()

    # Load genome
    genome_json, output_size = load_genome(args.genome, args.color)
    genome = json.loads(genome_json)
    n_hidden = sum(1 for n in genome["nodes"] if n["node_type"] == 1)
    n_conns = sum(1 for c in genome["connections"] if c["enabled"])
    print(f"Loaded {args.color} genome: {output_size} outputs, "
          f"{n_hidden} hidden nodes, {n_conns} enabled connections")

    if args.test:
        # Dry run: build network, encode start position, pick a move
        network = SparseNetwork(genome)
        board = chess.Board()
        inputs = encode_board(board)
        print(f"Board encoding: {len(inputs)} floats, "
              f"sum={sum(inputs):.2f}")

        move = pick_move(network, board, output_size)
        print(f"First move (start position): {move.uci()}")
        assert move in board.legal_moves, f"{move.uci()} is not legal!"
        print("Test passed: move is legal.")
        return

    # Real game mode — requires LICHESS_TOKEN
    token = os.environ.get("LICHESS_TOKEN")
    if not token:
        print("Error: Set LICHESS_TOKEN environment variable.", file=sys.stderr)
        print("  Get a Bot API token at https://lichess.org/account/oauth/token",
              file=sys.stderr)
        sys.exit(1)

    bot = NeatLichessBot(token, genome_json, output_size)
    print(f"Bot account: {bot.bot_id}")

    try:
        bot.run(
            max_games=args.games,
            color=args.color if args.challenge else None,
            challenge_opponent=args.challenge,
        )
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        stats = bot.get_stats()
        print(f"\nSession: {stats['wins']}W / {stats['draws']}D / {stats['losses']}L")
        if stats["ratings"]:
            for tc, rating in stats["ratings"].items():
                print(f"  {tc}: {rating}")


if __name__ == "__main__":
    main()
