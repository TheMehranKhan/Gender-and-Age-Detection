#!/bin/bash

# Function to kill background processes on exit
cleanup() {
    echo "Stopping servers..."
    if [ -n "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null
    fi
    if [ -n "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null
    fi
    exit
}

# Trap signals for cleanup on exit or terminal close
trap cleanup SIGINT SIGHUP SIGTERM

# Start Backend
echo "Starting Backend..."
python3 backend/main.py &
BACKEND_PID=$!

# Wait a bit for backend to initialize
sleep 2

# Start Frontend
echo "Starting Frontend..."
python3 frontend/server.py &
FRONTEND_PID=$!

echo "Backend running on PID $BACKEND_PID"
echo "Frontend running on PID $FRONTEND_PID"
echo "Access Frontend at http://localhost:3002"
echo "Press Ctrl+C to stop"

# Wait for processes
wait
