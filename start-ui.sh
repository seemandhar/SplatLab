#!/usr/bin/env bash
# Start the SplatLab control panel.
#
#   bash start-ui.sh                      # localhost:8420, no auth
#   PORT=9000 bash start-ui.sh            # custom port
#   HOST=0.0.0.0 GSS_TOKEN=secret bash start-ui.sh   # LAN access, token-gated
#
# The dashboard is a thin wrapper over the existing tmux + ledger machinery —
# it does not need a build step or any npm install. Requires: node, tmux,
# python3 (+ pyyaml), and the `claude` CLI on PATH for launching runs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PORT="${PORT:-8420}"
export HOST="${HOST:-127.0.0.1}"

if ! command -v node >/dev/null 2>&1; then
  echo "ERROR: node is required (v18+). Found none on PATH." >&2; exit 1
fi

echo "Starting SplatLab on http://${HOST/0.0.0.0/localhost}:${PORT}"
if [[ "${HOST}" == "0.0.0.0" && -z "${GSS_TOKEN:-}" ]]; then
  echo "WARNING: binding 0.0.0.0 with no GSS_TOKEN — anyone on your network can control runs." >&2
fi

exec node "$ROOT/ui/server.js"
