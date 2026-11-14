#!/bin/bash
# Checks that the overt/masked onset predictors line up with the audio that
# was played (predictors/check_alignment.py). A short job: reads the
# stimulus .wav files and predictor pickles, writes nothing but its report
# (results/alignment_check.txt, and this job's log).
#
# From a notebook: check_alignment(). By hand, from the repo root:
#   sbatch --account=def-ibruce cluster/check_alignment.sbatch
#
#SBATCH --job-name=cocktail-alignment
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=cluster/runs/%x-%j.out
set -uo pipefail

REPO="$SLURM_SUBMIT_DIR"
WORK="$(dirname "$REPO")"
export XDG_CACHE_HOME="$WORK/.cache"
export MAMBA_ROOT_PREFIX="$WORK/micromamba"
eval "$("$WORK/bin/micromamba" shell hook -s bash)"
micromamba activate cocktail-cortical

python "$REPO/predictors/check_alignment.py"
