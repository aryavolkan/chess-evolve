#!/usr/bin/env bash
# Wait for API to be ready, then start Vite frontend.
for i in $(seq 1 30); do
    curl -s -o /dev/null http://localhost:8420/api/system && break
    sleep 1
done
cd /Users/aryasen/projects/chess-evolve/monitor/frontend
exec /opt/homebrew/bin/npx vite --port 5173
