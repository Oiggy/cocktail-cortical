#!/bin/bash
# Runs the per-condition notebooks (analysis/clean.ipynb, diotic.ipynb, ...)
# on a compute node with COCKTAIL_RESULTS=compute: every TRF fit and model
# test they need is fitted (or read from this cluster's eelbrain cache) and
# saved as a plain pickle in dataset/cocktail/bids/derivatives/results.
# The notebooks on the Mac read them from here over SSH; nothing is copied
# to the Mac. A notebook can also submit this job itself:
# submit_cluster_run('diotic').
#
# Submit from the repo root:
#   sbatch --account=def-ibruce cluster/run_analysis.sbatch
# Only some notebooks:
#   sbatch --account=def-ibruce --export=ALL,NOTEBOOKS="clean diotic" cluster/run_analysis.sbatch
#
# Results are saved one by one as they finish, so after a TIMEOUT just
# submit again: what is already saved is read back in seconds.
#
#SBATCH --job-name=cocktail-results
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=cluster/runs/%x-%j.out
#SBATCH --mail-type=END,FAIL
set -uo pipefail

REPO="$SLURM_SUBMIT_DIR"
WORK="$(dirname "$REPO")"
export XDG_CACHE_HOME="$WORK/.cache"
export MAMBA_ROOT_PREFIX="$WORK/micromamba"
eval "$("$WORK/bin/micromamba" shell hook -s bash)"
micromamba activate cocktail-cortical

export COCKTAIL_ANSWER="${COCKTAIL_ANSWER:-s}"
export COCKTAIL_RESULTS=compute
NOTEBOOKS="${NOTEBOOKS:-clean diotic diotic_overt_masked_onsets dichotic_left dichotic_left_overt_masked_onsets dichotic_right dichotic_right_overt_masked_onsets binaural binaural_overt_masked_onsets}"
RESULTS="$WORK/dataset/cocktail/bids/derivatives/results"

cd "$REPO/analysis"
python -c "from parameters import preprocess; preprocess()" || exit 1

failed=()
for nb in $NOTEBOOKS; do
    echo "=== $nb  $(date '+%F %T')"
    jupyter nbconvert --to notebook --execute "$nb.ipynb" \
        --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 \
        --output-dir "$REPO/cluster/runs" --output "${nb}_${SLURM_JOB_ID}.ipynb" || failed+=("$nb")
done

echo "=== saved results: $(ls "$RESULTS" 2>/dev/null | grep -c '\.pickle$')"
if [ ${#failed[@]} -gt 0 ]; then
    echo "FAILED: ${failed[*]} (see cluster/runs/<notebook>_${SLURM_JOB_ID}.ipynb)"
    exit 1
fi
