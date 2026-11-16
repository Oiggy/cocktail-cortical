#!/bin/bash
# Reports how the downloaded DTU dataset is laid out (dtu/inspect_dtu.py).
# From a notebook: dtu_inspect(). By hand, from the repo root:
#   sbatch --account=def-ibruce cluster/dtu_inspect.sbatch
#SBATCH --job-name=cocktail-dtu-inspect
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=cluster/runs/%x-%j.out
set -uo pipefail

REPO="$SLURM_SUBMIT_DIR"
WORK="$(dirname "$REPO")"
export XDG_CACHE_HOME="$WORK/.cache"
export MAMBA_ROOT_PREFIX="$WORK/micromamba"
eval "$("$WORK/bin/micromamba" shell hook -s bash)"
micromamba activate cocktail-cortical

python "$REPO/dtu/inspect_dtu.py"
