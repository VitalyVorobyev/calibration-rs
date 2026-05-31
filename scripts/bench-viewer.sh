#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <dataset-id>" >&2
  exit 2
fi

dataset="$1"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
public_registry="$root/crates/vision-calibration-bench/registry/public.json"
private_registry="$root/crates/vision-calibration-bench/registry/private.json"
registry="$public_registry"
features="tier-b"

if ! grep -q "\"id\"[[:space:]]*:[[:space:]]*\"$dataset\"" "$public_registry"; then
  if [[ -f "$private_registry" ]] && grep -q "\"id\"[[:space:]]*:[[:space:]]*\"$dataset\"" "$private_registry"; then
    registry="$private_registry"
    features="tier-b laser"
  fi
fi

out="${TMPDIR:-/tmp}/calib-bench-${dataset}.json"
port="${CALIB_VIEWER_PORT:-5173}"
viewer="$root/tools/calibration-viewer"

echo "Running benchmark: $dataset" >&2
echo "Registry: $registry" >&2
cargo run -p vision-calibration-bench --features "$features" --bin calib-bench -- \
  run --dataset "$dataset" --registry "$registry" > "$out"

echo "Wrote $out" >&2

if [[ ! -d "$viewer/node_modules" ]]; then
  (cd "$viewer" && npm install)
fi

if ! curl -fsS "http://127.0.0.1:$port/" >/dev/null 2>&1; then
  echo "Starting calibration-viewer on port $port" >&2
  (cd "$viewer" && npm run dev -- --port "$port" --strictPort > /tmp/calibration-viewer.log 2>&1 &)
  for _ in {1..60}; do
    if curl -fsS "http://127.0.0.1:$port/" >/dev/null 2>&1; then
      break
    fi
    sleep 0.25
  done
fi

url="http://127.0.0.1:$port/?bench=/@fs$out"
echo "Opening $url" >&2
if command -v open >/dev/null 2>&1; then
  open "$url"
else
  echo "$url"
fi
