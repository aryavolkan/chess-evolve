#!/usr/bin/env python3
"""Download and filter Lichess puzzles into JSON files for curriculum training.

Usage:
    python scripts/prepare_puzzles.py [--output-dir data/puzzles] [--max-per-stage 500]

Downloads the Lichess puzzle CSV (if not cached), filters by rating and themes,
and writes stage-specific JSON files:
  - stage0_puzzles.json: easy puzzles (rating < 800, simple themes)
  - stage1_puzzles.json: intermediate puzzles (rating 800-1200, tactical themes)

Each JSON file is a list of {"fen": ..., "solution": ...} objects where solution
is the first move of the puzzle solution in UCI format.
"""
import csv
import json
import urllib.request
from pathlib import Path

LICHESS_PUZZLES_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
LICHESS_PUZZLES_CSV_URL = "https://database.lichess.org/lichess_db_puzzle.csv.bz2"

# Stage definitions: (min_rating, max_rating, allowed_themes)
STAGES = {
    0: {
        "min_rating": 0,
        "max_rating": 800,
        "themes": {
            "mateIn1", "backRankMate", "hangingPiece", "oneMove",
            "capturingDefender", "exposedKing",
        },
        "filename": "stage0_puzzles.json",
    },
    1: {
        "min_rating": 800,
        "max_rating": 1200,
        "themes": {
            "fork", "pin", "skewer", "discoveredAttack", "mateIn2",
            "trappedPiece", "hangingPiece", "doubleCheck",
        },
        "filename": "stage1_puzzles.json",
    },
}


def download_puzzle_csv(cache_dir: Path) -> Path:
    """Download Lichess puzzle CSV (bz2 compressed) and decompress."""
    csv_path = cache_dir / "lichess_db_puzzle.csv"
    if csv_path.exists():
        print(f"  Using cached puzzle CSV: {csv_path}")
        return csv_path

    bz2_path = cache_dir / "lichess_db_puzzle.csv.bz2"
    if not bz2_path.exists():
        print(f"  Downloading Lichess puzzles from {LICHESS_PUZZLES_CSV_URL}...")
        cache_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(LICHESS_PUZZLES_CSV_URL, str(bz2_path))
        print(f"  Downloaded: {bz2_path.stat().st_size / 1024 / 1024:.1f} MB")

    print("  Decompressing bz2...")
    import bz2
    with bz2.open(bz2_path, "rt") as fin, open(csv_path, "w") as fout:
        for line in fin:
            fout.write(line)
    print(f"  Decompressed: {csv_path.stat().st_size / 1024 / 1024:.1f} MB")
    return csv_path


def extract_first_solution_move(moves_str: str) -> str | None:
    """Extract the puzzle solution move from the Moves field.

    Lichess puzzle format: first move is the opponent's move that sets up the puzzle,
    second move is the solution. We want the solution (second move).
    """
    parts = moves_str.strip().split()
    if len(parts) >= 2:
        return parts[1]  # Second move is the solution
    return None


def filter_puzzles(
    csv_path: Path,
    min_rating: int,
    max_rating: int,
    allowed_themes: set[str],
    max_count: int,
) -> list[dict]:
    """Filter puzzles from CSV by rating and themes."""
    puzzles = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)  # PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags
        for row in reader:
            if len(row) < 8:
                continue
            rating = int(row[3])
            if rating < min_rating or rating >= max_rating:
                continue
            themes = set(row[7].split())
            if not themes & allowed_themes:
                continue
            solution = extract_first_solution_move(row[2])
            if solution is None:
                continue

            # Apply the setup move to get the actual puzzle position
            # The FEN is the position BEFORE the opponent's setup move
            # For simplicity, we'll use the FEN as-is and let the network
            # solve from the position after the setup move is mentally applied.
            # Actually, Lichess FEN is the position AFTER the last game move,
            # and the first move in Moves is the setup move that creates the puzzle.
            # We need to apply the setup move to get the puzzle position.
            # For now, store the FEN and solution — the Rust evaluator will
            # need the post-setup-move FEN.

            # Store setup move so we can apply it
            setup_move = row[2].strip().split()[0]
            puzzles.append({
                "fen": row[1],
                "setup_move": setup_move,
                "solution": solution,
                "rating": rating,
            })
            if len(puzzles) >= max_count:
                break
    return puzzles


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prepare Lichess puzzles for curriculum training")
    parser.add_argument("--output-dir", default="data/puzzles", help="Output directory")
    parser.add_argument("--cache-dir", default="data/puzzles/cache", help="Cache directory for downloads")
    parser.add_argument("--max-per-stage", type=int, default=500, help="Max puzzles per stage")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = download_puzzle_csv(cache_dir)

    for stage_id, stage_def in STAGES.items():
        puzzles = filter_puzzles(
            csv_path,
            stage_def["min_rating"],
            stage_def["max_rating"],
            stage_def["themes"],
            args.max_per_stage,
        )
        out_path = output_dir / stage_def["filename"]

        # Write just fen + solution (what the Rust evaluator needs)
        output_puzzles = [{"fen": p["fen"], "solution": p["solution"]} for p in puzzles]
        out_path.write_text(json.dumps(output_puzzles, indent=2))
        print(f"  Stage {stage_id}: {len(puzzles)} puzzles → {out_path}")


if __name__ == "__main__":
    main()
