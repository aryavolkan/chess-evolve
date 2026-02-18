#!/usr/bin/env bash
set -euo pipefail

SWEEP_ID=${1:-""}
WORKERS=${2:-2}

if [[ -z "$SWEEP_ID" ]]; then
  echo "Usage: $0 <sweep_id> <num_workers>"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

for i in $(seq 1 "$WORKERS"); do
  LOG_FILE="worker${i}.log"
  echo "Starting worker ${i} → ${LOG_FILE}"
  nohup python3 chess_sweep_worker.py --sweep-id "$SWEEP_ID" --count 1 > "$LOG_FILE" 2>&1 &
  sleep 2
done

echo ""
echo "Workers launched."
echo "Logs: tail -f worker*.log"
