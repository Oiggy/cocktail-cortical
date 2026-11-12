#!/bin/bash
# Opens one SSH connection to the cluster that the per-condition notebooks
# reuse to read results and submit jobs: one Duo login, then it stays open
# in the background (12 hours by default). Run it in a Mac terminal before
# using the notebooks, and again after the laptop sleeps or the time is up.
#
# From the repo root on the Mac:
#   bash cluster/connect.sh            # open (or check) the connection
#   bash cluster/connect.sh close      # close it
#
# Options (environment variables):
#   CLUSTER        cluster address   (default: narval.alliancecan.ca)
#   CLUSTER_USER   cluster username  (default: iggy)
#   CONNECT_HOURS  how long it stays open (default: 12)
#
# CLUSTER and CLUSTER_USER must match CLUSTER_HOST in analysis/parameters.py,
# and the control path must match CONTROL_PATH in analysis/experiment.py.
set -euo pipefail

CLUSTER="${CLUSTER:-narval.alliancecan.ca}"
CLUSTER_USER="${CLUSTER_USER:-iggy}"
CONNECT_HOURS="${CONNECT_HOURS:-12}"
CONTROL="$HOME/.ssh/cocktail-%r@%h:%p"
TARGET="$CLUSTER_USER@$CLUSTER"

if [ "${1:-}" = "close" ]; then
    ssh -o ControlPath="$CONTROL" -O exit "$TARGET" 2>/dev/null && echo "Closed." || echo "No open connection."
    exit 0
fi

if ssh -o ControlPath="$CONTROL" -O check "$TARGET" 2>/dev/null; then
    echo "Already connected to $CLUSTER."
    exit 0
fi

mkdir -p "$HOME/.ssh"
ssh -fN -o ControlMaster=yes -o ControlPath="$CONTROL" -o ControlPersist="${CONNECT_HOURS}h" \
    -o ServerAliveInterval=60 "$TARGET"
echo "Connected to $CLUSTER for up to $CONNECT_HOURS hours. The notebooks can now reach the cluster."
