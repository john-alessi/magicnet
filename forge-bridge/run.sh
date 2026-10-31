#!/usr/bin/env bash
# Run the instrumented harness.
#
# Two Forge quirks are handled here:
#   - it must run from a directory containing res/ (we symlink it)
#   - -Djava.awt.headless=true makes it exit(1) silently; it needs a real or
#     virtual X display, so use xvfb-run when DISPLAY is unset
set -euo pipefail

cd "$(dirname "$0")"

FORGE_JAR="${FORGE_JAR:-/usr/share/java/forge-gui-desktop.jar}"
FORGE_RES="${FORGE_RES:-/usr/share/forge-gui-desktop/res}"
HEAP="${HEAP:-2048m}"

if [[ ! -e res ]]; then
  ln -s "$FORGE_RES" res
  echo "linked res -> $FORGE_RES"
fi

JAVA_ARGS=(-Xmx"$HEAP" -cp "build:$FORGE_JAR" magicnet.bridge.Harness "$@")

if [[ -z "${DISPLAY:-}" ]]; then
  if command -v xvfb-run >/dev/null 2>&1; then
    echo "no DISPLAY; using xvfb-run"
    exec xvfb-run -a java "${JAVA_ARGS[@]}"
  fi
  echo "warning: no DISPLAY and no xvfb-run; Forge will likely exit silently" >&2
fi

exec java "${JAVA_ARGS[@]}"
