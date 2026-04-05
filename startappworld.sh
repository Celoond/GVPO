#!/bin/bash
PORTS=({9000..9031})
APPWORLDROOT="${APPWORLDROOT:-/path/to/appworld}"
REDIS_PORT="${REDIS_PORT:-6390}"
REDIS_SERVER_BIN="${REDIS_SERVER_BIN:-redis-server}"
REDIS_CLI_BIN="${REDIS_CLI_BIN:-redis-cli}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="${LOG_DIR:-./servelogs}"

mkdir -p "$LOG_DIR"

nohup "$REDIS_SERVER_BIN" --port "$REDIS_PORT" > "$LOG_DIR/redis.log" 2>&1 &
# Wait for Redis to start.
sleep 1

cd "$APPWORLDROOT"
for i in "${!PORTS[@]}"; do
    PORT=${PORTS[i]}
    echo "Starting AppWorld on port $PORT with root $APPWORLDROOT..."
    nohup appworld serve environment --port "$PORT" --root "$APPWORLDROOT" > "$LOG_DIR/log_${PORT}.out" 2>&1 &
    echo "Starting worker for port $PORT..."
    "$PYTHON_BIN" -u "app_worker.py" "$PORT" > "$LOG_DIR/worker_${PORT}.log" 2>&1 &
done

echo "All services started!"
