#!/usr/bin/env bash
# Host-side one-time setup: lock Jetson Orin to max performance so sustained
# 25-30 fps YOLO inference doesn't throttle mid-day.
#
# Usage: ./scripts/jetson-power.sh   (re-run after every reboot:
# jetson_clocks is not persistent)
set -euo pipefail

echo "before: $(nvpmodel -q)"
sudo nvpmodel -m 0
sudo jetson_clocks
echo "after:  $(nvpmodel -q)"
echo "power mode set (MAXN + clocks maxed)."
