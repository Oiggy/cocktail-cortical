#!/bin/bash
# The DTU forward-model analysis (dtu/dtu_pipeline.py), one stage per job:
#   STAGE=prepare  predictors and EEG (once)
#   STAGE=split    the overt/masked onsets of every two-talker mixture (once)
#   STAGE=fit      one subject's TRFs (as a job array: --array=1-18, task ID = subject)
#   STAGE=stats    group tests and TRFs, read by analysis/dtu_*.ipynb
# Submitted in that order, each waiting for the one before, by dtu_run() in analysis/dtu.ipynb, which also
# sets the job sizes. Everything is written under dataset/dtu/derivatives; finished fits are kept, so a
# rerun only does what is missing.
#SBATCH --output=cluster/runs/%x-%A_%a.out
set -uo pipefail

REPO="$SLURM_SUBMIT_DIR"
WORK="$(dirname "$REPO")"
export XDG_CACHE_HOME="$WORK/.cache"
export MAMBA_ROOT_PREFIX="$WORK/micromamba"
export TQDM_DISABLE=1
eval "$("$WORK/bin/micromamba" shell hook -s bash)"
micromamba activate cocktail-cortical

case "${STAGE:?set STAGE=prepare, rawcheck, split, fit or stats}" in
    fit) python "$REPO/dtu/dtu_pipeline.py" fit "${SLURM_ARRAY_TASK_ID:?fit runs as a job array}" ;;
    *)   python "$REPO/dtu/dtu_pipeline.py" "$STAGE" ;;
esac
