#!/bin/bash
# Runs analysis/cortical_analysis.ipynb top to bottom on a compute node,
# filling the eelbrain cache (every TRF fit and model comparison) under
# dataset/cocktail/bids/derivatives/. Copy that back to the MacBook and
# the notebook there loads the fits instead of computing them.
#
# Submit from the repo root:
#   sbatch --account=def-ibruce cluster/run_notebook.sbatch
#
# Preprocessing prompts ("redo or skip subjects already done?") are
# answered with COCKTAIL_ANSWER: 's' (skip, default) or 'r' (redo), e.g.
#   sbatch --account=def-ibruce --export=ALL,COCKTAIL_ANSWER=r cluster/run_notebook.sbatch
#
#SBATCH --job-name=cocktail-trfs
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=cluster/runs/%x-%j.out
#SBATCH --mail-type=END,FAIL
set -euo pipefail

REPO="$SLURM_SUBMIT_DIR"
WORK="$(dirname "$REPO")"
export XDG_CACHE_HOME="$WORK/.cache"
export MAMBA_ROOT_PREFIX="$WORK/micromamba"
eval "$("$WORK/bin/micromamba" shell hook -s bash)"
micromamba activate cocktail-cortical

export COCKTAIL_ANSWER="${COCKTAIL_ANSWER:-s}"

cd "$REPO/analysis"
jupyter nbconvert --to notebook --execute cortical_analysis.ipynb \
    --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 \
    --output-dir "$REPO/cluster/runs" --output "cortical_analysis_${SLURM_JOB_ID}.ipynb"
