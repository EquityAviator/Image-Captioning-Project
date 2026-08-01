#!/usr/bin/env bash
# Auto-restart wrapper for the CaptionAI backend.
# Restarts the uvicorn process if it dies, with a small backoff.
set -u

cd "$(dirname "$0")/.."
export PORT="${PORT:-8000}"
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES=-1
export TF_ENABLE_ONEDNN_OPTS=0
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/backend"

LOG=/tmp/backend.log
PIDFILE=/tmp/backend.pid

# Kill any existing instance
if [ -f "$PIDFILE" ]; then
    OLDPID=$(cat "$PIDFILE" 2>/dev/null || true)
    if [ -n "${OLDPID:-}" ] && kill -0 "$OLDPID" 2>/dev/null; then
        kill "$OLDPID" 2>/dev/null || true
        sleep 2
    fi
fi
pkill -f "uvicorn main:app" 2>/dev/null || true
sleep 1

attempt=0
while true; do
    attempt=$((attempt + 1))
    echo "[$(date -u +%H:%M:%S)] starting backend (attempt $attempt) …"
    python3 -m uvicorn main:app \
        --host 0.0.0.0 \
        --port "$PORT" \
        --log-level info \
        --app-dir backend \
        --timeout-keep-alive 30 \
        >>"$LOG" 2>&1 &
    PID=$!
    echo "$PID" > "$PIDFILE"
    echo "[$(date -u +%H:%M:%S)] backend PID=$PID"
    wait "$PID"
    EXIT=$?
    echo "[$(date -u +%H:%M:%S)] backend exited with $EXIT — restarting in 3s"
    sleep 3
    # Truncate log if it grew too large
    if [ -f "$LOG" ] && [ "$(wc -c <"$LOG")" -gt 5242880 ]; then
        tail -c 2097152 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
    fi
done
