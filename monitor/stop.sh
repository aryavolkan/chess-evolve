#!/usr/bin/env bash
# Stop the worker monitor dashboard.
pkill -f "uvicorn monitor.api:app" 2>/dev/null && echo "API stopped" || echo "API not running"
pkill -f "vite.*monitor/frontend" 2>/dev/null && echo "Frontend stopped" || echo "Frontend not running"
