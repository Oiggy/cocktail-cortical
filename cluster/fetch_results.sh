#!/bin/bash
# Downloads the saved TRF fits and model tests that cluster/run_analysis.sbatch
# wrote on the cluster (dataset/cocktail/bids/derivatives/results) into the
# Mac's dataset, so the per-condition notebooks can read them. One login
# (Duo MFA); only new or changed files are copied.
#
# From the repo root on the Mac:
#   bash cluster/fetch_results.sh
#
# Options (environment variables):
#   CLUSTER       cluster address   (default: narval.alliancecan.ca)
#   CLUSTER_USER  cluster username  (default: iggy)
#   MAC_DATASET   local dataset dir (default: sibling "dataset/" next to repo)
set -euo pipefail

CLUSTER="${CLUSTER:-narval.alliancecan.ca}"
CLUSTER_USER="${CLUSTER_USER:-iggy}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAC_DATASET="${MAC_DATASET:-$(dirname "$REPO")/dataset}"

SRC="${CLUSTER_USER}@${CLUSTER}:projects/def-ibruce/${CLUSTER_USER}/dataset/cocktail/bids/derivatives/results/"
DST="$MAC_DATASET/cocktail/bids/derivatives/results/"

mkdir -p "$DST"
rsync -avP --exclude='*.part' "$SRC" "$DST"
echo "Saved results on this Mac: $(ls "$DST" | grep -c '\.pickle$')"
