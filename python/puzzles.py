"""Lichess puzzle loader for Stage 0 curriculum training.

Downloads the Lichess puzzle database (CSV), caches locally, and provides
filtered puzzle sets for NEAT fitness evaluation.

Puzzle CSV columns: PuzzleId, FEN, Moves, Rating, RatingDeviation,
Popularity, NbPlays, Themes, GameUrl, OpeningTags.

Moves are space-separated UCI strings. The first move is the opponent's
(sets up the puzzle), and the second is the expected solution.
"""

from __future__ import annotations

import csv
import fcntl
import subprocess
from pathlib import Path

# Where to cache the puzzle CSV locally
_CACHE_DIR = Path(__file__).parent.parent / ".puzzle_cache"
_CSV_PATH = _CACHE_DIR / "lichess_db_puzzle.csv"
_ZST_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"


def _ensure_downloaded() -> Path:
    """Download and decompress the Lichess puzzle CSV if not cached.

    Uses a file lock to prevent multiple workers from racing on download.
    """
    if _CSV_PATH.exists():
        return _CSV_PATH

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = _CACHE_DIR / "download.lock"

    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        # Re-check after acquiring lock — another worker may have finished
        if _CSV_PATH.exists():
            return _CSV_PATH

        zst_path = _CACHE_DIR / "lichess_db_puzzle.csv.zst"

        print(f"  Downloading Lichess puzzles from {_ZST_URL} ...")
        subprocess.run(
            ["curl", "-L", "-o", str(zst_path), _ZST_URL],
            check=True,
        )

        print("  Decompressing ...")
        subprocess.run(
            ["zstd", "-d", str(zst_path), "-o", str(_CSV_PATH)],
            check=True,
        )
        zst_path.unlink(missing_ok=True)
        print(f"  Cached {_CSV_PATH.stat().st_size / 1e6:.0f} MB of puzzles")

    return _CSV_PATH


def load_puzzles(
    max_rating: int = 800,
    min_popularity: int = 80,
    max_count: int = 500,
    themes: set[str] | None = None,
) -> list[dict]:
    """Load puzzles from the Lichess database, filtered by difficulty.

    Each puzzle dict has: fen, best_move (UCI), rating, themes.
    The FEN already has the opponent's move applied (puzzle position).

    Args:
        max_rating: Maximum puzzle rating (800 = very easy).
        min_popularity: Minimum popularity score (0-100).
        max_count: Maximum number of puzzles to return.
        themes: If set, only include puzzles matching at least one theme.
            Example themes: mateIn1, hangingPiece, fork, pin, skewer.

    Returns:
        List of puzzle dicts with keys: fen, best_move, rating, themes.
    """
    csv_path = _ensure_downloaded()

    puzzles = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 8:
                continue

            puzzle_id, fen, moves, rating_str, _, popularity_str, _, puzzle_themes = row[:8]

            # Skip header
            if puzzle_id == "PuzzleId":
                continue

            try:
                rating = int(rating_str)
                popularity = int(popularity_str)
            except ValueError:
                continue

            if rating > max_rating or popularity < min_popularity:
                continue

            # Theme filter
            puzzle_theme_set = set(puzzle_themes.split())
            if themes and not puzzle_theme_set & themes:
                continue

            # Parse moves: first move is opponent's (apply to FEN), second is solution
            move_list = moves.split()
            if len(move_list) < 2:
                continue

            # Apply the opponent's setup move to get the puzzle position
            try:
                import chess
                board = chess.Board(fen)
                board.push_uci(move_list[0])
                puzzle_fen = board.fen()
            except Exception:
                continue

            puzzles.append({
                "fen": puzzle_fen,
                "best_move": move_list[1],
                "rating": rating,
                "themes": puzzle_themes,
            })

            if len(puzzles) >= max_count:
                break

    return puzzles


def get_puzzle_fens_and_moves(
    puzzles: list[dict],
) -> tuple[list[str], list[str]]:
    """Extract parallel FEN and best_move lists for Rust evaluation."""
    fens = [p["fen"] for p in puzzles]
    moves = [p["best_move"] for p in puzzles]
    return fens, moves
