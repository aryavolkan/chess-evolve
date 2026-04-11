#!/usr/bin/env bash
# Start the worker monitor dashboard (API + frontend).
# Usage: ./monitor/start.sh

set -e

DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR"

# Kill any existing instances
pkill -f "uvicorn monitor.api:app" 2>/dev/null && echo "Stopped old API" || true
pkill -f "vite.*monitor/frontend" 2>/dev/null && echo "Stopped old frontend" || true
sleep 1

# Start API
.venv/bin/python -m uvicorn monitor.api:app --host 0.0.0.0 --port 8420 > /tmp/monitor-api.log 2>&1 &
API_PID=$!
echo "API started: PID $API_PID (port 8420)"

# Install frontend deps if needed
if [ ! -d monitor/frontend/node_modules ]; then
    echo "Installing frontend dependencies..."
    cd monitor/frontend && npm install && cd "$DIR"
fi

# Start frontend
cd monitor/frontend && npm run dev > /tmp/monitor-frontend.log 2>&1 &
FE_PID=$!
cd "$DIR"
echo "Frontend started: PID $FE_PID (port 5173)"

TS_IP=$(tailscale ip -4 2>/dev/null || true)
echo ""
echo "Dashboard: http://localhost:5173"
[ -n "$TS_IP" ] && echo "Tailscale: http://$TS_IP:5173"
echo "API:       http://localhost:8420/api/workers"
echo ""
echo "Stop: ./monitor/stop.sh"
