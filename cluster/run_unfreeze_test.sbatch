#!/bin/bash
# The boosting `unfreeze` test (analysis/unfreeze_test.py) on a compute node:
# held-out prediction with and without `unfreeze`, cocktail conditions only,
# in its own dataset root and results folder. Nothing of the main analysis is
# written to. Needs cluster/setup_unfreeze_env.sh run once first (login node).
#
# Submit from the repo root (or use cluster/launch_unfreeze_test.sh):
#   sbatch --account=def-ibruce cluster/run_unfreeze_test.sbatch
# Only some conditions:
#   sbatch --account=def-ibruce --export=ALL,UNFREEZE_EPOCHS="diotic binaural" cluster/run_unfreeze_test.sbatch
#
# Each fit is saved as soon as it is done, so after a TIMEOUT just submit again.
#
#SBATCH --job-name=unfreeze-test
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=cluster/runs/%x-%j.out
#SBATCH --mail-type=END,FAIL
set -uo pipefail

REPO="$SLURM_SUBMIT_DIR"
WORK="$(dirname "$REPO")"
export XDG_CACHE_HOME="$WORK/.cache"
export MAMBA_ROOT_PREFIX="$WORK/micromamba"
eval "$("$WORK/bin/micromamba" shell hook -s bash)"
micromamba activate cocktail-cortical-unfreeze || { echo "No cocktail-cortical-unfreeze environment: run bash cluster/setup_unfreeze_env.sh on a login node"; exit 1; }

export COCKTAIL_ANSWER="${COCKTAIL_ANSWER:-s}"
export COCKTAIL_RESULTS=remote
bash "$REPO/cluster/make_unfreeze_root.sh" || exit 1

cd "$REPO/analysis"
python unfreeze_test.py
status=$?
echo "=== done $(date '+%F %T'), exit $status"
echo "=== summary: $WORK/dataset/cocktail/bids_unfreeze/derivatives/results_unfreeze/summary.txt"
exit $status
