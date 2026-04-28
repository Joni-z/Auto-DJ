#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
CONDA_ENV="${CONDA_ENV:-ml}"

find_lan_ip() {
  local ip
  ip="$(ipconfig getifaddr en0 2>/dev/null || true)"
  if [[ -z "$ip" ]]; then
    ip="$(ipconfig getifaddr en1 2>/dev/null || true)"
  fi
  if [[ -z "$ip" ]]; then
    ip="$(ifconfig | awk '/inet / && $2 != "127.0.0.1" { print $2; exit }')"
  fi
  printf "%s" "${ip:-127.0.0.1}"
}

LAN_IP="$(find_lan_ip)"
BACKEND_LOCAL="http://127.0.0.1:${BACKEND_PORT}"
FRONTEND_LOCAL="http://127.0.0.1:${FRONTEND_PORT}"
BACKEND_LAN="http://${LAN_IP}:${BACKEND_PORT}"
FRONTEND_LAN="http://${LAN_IP}:${FRONTEND_PORT}"

cleanup() {
  echo
  echo "Stopping AutoDJ dev servers..."
  if [[ -n "${BACKEND_PID:-}" ]]; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
  if [[ -n "${FRONTEND_PID:-}" ]]; then
    kill "$FRONTEND_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "Starting AutoDJ"
echo
echo "Local:"
echo "  Frontend: ${FRONTEND_LOCAL}"
echo "  Backend:  ${BACKEND_LOCAL}"
echo
echo "LAN:"
echo "  Frontend: ${FRONTEND_LAN}"
echo "  Backend:  ${BACKEND_LAN}"
echo
echo "Press Ctrl+C to stop both servers."
echo

(
  cd "${ROOT_DIR}/backend"
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
  else
    echo "conda was not found; using current Python environment for backend."
  fi
  uvicorn app.main:app --reload --host 0.0.0.0 --port "${BACKEND_PORT}"
) &
BACKEND_PID=$!

(
  cd "${ROOT_DIR}/frontend"
  npm exec vite -- --host 0.0.0.0 --port "${FRONTEND_PORT}"
) &
FRONTEND_PID=$!

wait "$BACKEND_PID" "$FRONTEND_PID"
