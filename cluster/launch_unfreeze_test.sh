#!/bin/bash
# One command, on an Alliance LOGIN node, from the repo root, after `git pull`:
#   bash cluster/launch_unfreeze_test.sh
# Sets up the cocktail-cortical-unfreeze environment if it is not there yet
# (the login node has the internet; compute nodes do not), then submits the
# test job. After that the laptop can be switched off: the cluster emails when
# the job ends. The summary is in
#   dataset/cocktail/bids_unfreeze/derivatives/results_unfreeze/summary.txt
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(dirname "$REPO")"
cd "$REPO"
if [ ! -x "$WORK/micromamba/envs/cocktail-cortical-unfreeze/bin/python" ] \
   || ! "$WORK/micromamba/envs/cocktail-cortical-unfreeze/bin/python" -c "import inspect, eelbrain; assert 'unfreeze' in inspect.signature(eelbrain.boosting).parameters" 2>/dev/null; then
    bash cluster/setup_unfreeze_env.sh
fi
# The test's dataset copy, and a check on this node that the pipeline opens in it (nothing is fitted),
# so a problem shows here instead of in the job
bash cluster/make_unfreeze_root.sh
(cd analysis && XDG_CACHE_HOME="$WORK/.cache" COCKTAIL_RESULTS=remote MPLBACKEND=Agg "$WORK/micromamba/envs/cocktail-cortical-unfreeze/bin/python" unfreeze_test.py --check)
mkdir -p cluster/runs
sbatch --account=def-ibruce ${UNFREEZE_EPOCHS:+--export=ALL,UNFREEZE_EPOCHS="$UNFREEZE_EPOCHS"} cluster/run_unfreeze_test.sbatch
echo "Submitted. Check with: squeue -u \$USER ; log: cluster/runs/unfreeze-test-<job>.out"
