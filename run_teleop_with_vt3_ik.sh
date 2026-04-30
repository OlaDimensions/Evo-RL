#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/src/lerobot/gui/hil_recording/run_teleop_with_vt3_ik.sh" "$@"
