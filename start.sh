#!/bin/bash
set -e

echo "research-mlx-ui starting..."

cleanup() {
    [ -n "$CLIENT_PID" ] && kill "$CLIENT_PID" 2>/dev/null
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
}
trap cleanup EXIT

# Check prerequisites
command -v uv >/dev/null 2>&1 || { echo "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
command -v node >/dev/null 2>&1 || { echo "Node.js not found. Install from nodejs.org"; exit 1; }

# Install Python deps
uv sync

# Install frontend deps
cd client && npm install --legacy-peer-deps && cd ..

# Run prepare.py if data doesn't exist
if [ ! -d "$HOME/.cache/autoresearch" ]; then
    echo "First run: downloading data (this takes ~2 minutes)..."
    uv run prepare.py
fi

# Build frontend (or run dev server)
if [ "$DEV" = "1" ]; then
    cd client && npx vite --port 5173 &
    CLIENT_PID=$!
    cd ..
    echo "Frontend dev server: http://localhost:5173"
else
    cd client && npm run build && cd ..
fi

# Start FastAPI server
echo "Starting server at http://localhost:8000"
uv run uvicorn server.main:app --host 0.0.0.0 --port 8000 &
SERVER_PID=$!

wait $SERVER_PID
