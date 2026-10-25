#!/bin/bash
# Automates the full cluster pipeline from the Mac:
#   1. Upload dataset to Narval via rsync
#   2. Submit the SLURM job
#   3. Poll until the job finishes
#   4. Download the derivatives (TRF cache) back to the Mac
#
# Uses SSH ControlMaster so you only authenticate (Duo MFA) once.
#
# Usage (from the repo root on Mac):
#   bash cluster/run_on_cluster.sh
#
# Or from the notebook:
#   !bash cluster/run_on_cluster.sh
#
# Options (environment variables):
#   CLUSTER       cluster address   (default: narval.alliancecan.ca)
#   CLUSTER_USER  cluster username  (default: iggy)
#   ACCOUNT       SLURM account     (default: def-ibruce)
#   MAC_DATASET   local dataset dir (default: sibling "dataset/" next to repo)
#   COCKTAIL_ANSWER  s=skip done subjects, r=redo (default: s)
#   SKIP_UPLOAD   set to 1 to skip the dataset upload step
#   SKIP_DOWNLOAD set to 1 to skip downloading results
set -euo pipefail

CLUSTER="${CLUSTER:-narval.alliancecan.ca}"
CLUSTER_USER="${CLUSTER_USER:-iggy}"
ACCOUNT="${ACCOUNT:-def-ibruce}"
COCKTAIL_ANSWER="${COCKTAIL_ANSWER:-s}"
SKIP_UPLOAD="${SKIP_UPLOAD:-0}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAC_DATASET="${MAC_DATASET:-$(dirname "$REPO")/dataset}"

REMOTE_BASE="projects/def-ibruce/${CLUSTER_USER}"
REMOTE_REPO="${REMOTE_BASE}/cocktail-cortical"
REMOTE_DATASET="${REMOTE_BASE}/dataset"

SSH_SOCKET="/tmp/ssh-cluster-$$"
SSH_OPTS="-o ControlMaster=auto -o ControlPath=$SSH_SOCKET -o ControlPersist=3600"

cleanup() {
    ssh $SSH_OPTS -O exit "${CLUSTER_USER}@${CLUSTER}" 2>/dev/null || true
    rm -f "$SSH_SOCKET"
}
trap cleanup EXIT

echo "==> Connecting to ${CLUSTER} (MFA will be required once)..."
ssh $SSH_OPTS -fN "${CLUSTER_USER}@${CLUSTER}"
echo "    Connected. All subsequent operations reuse this session."

remote() {
    ssh $SSH_OPTS "${CLUSTER_USER}@${CLUSTER}" "$@"
}

# --- Step 1: Upload dataset ---
if [ "$SKIP_UPLOAD" != "1" ]; then
    if [ ! -d "$MAC_DATASET" ]; then
        echo "ERROR: Dataset directory not found: $MAC_DATASET"
        exit 1
    fi
    echo ""
    echo "==> Uploading dataset to ${CLUSTER}:${REMOTE_DATASET}/ ..."
    rsync -avP -e "ssh $SSH_OPTS" \
        "$MAC_DATASET/" "${CLUSTER_USER}@${CLUSTER}:${REMOTE_DATASET}/"
    echo "    Upload complete."
else
    echo ""
    echo "==> Skipping dataset upload (SKIP_UPLOAD=1)."
fi

# --- Step 2: Pull latest code on cluster ---
echo ""
echo "==> Pulling latest code on cluster..."
remote "cd ~/${REMOTE_REPO} && git pull --no-rebase --no-edit"

# rsync on macOS drops sub-second mtimes, which makes eelbrain treat every
# cached file on the cluster as stale; copy the exact times across.
echo ""
echo "==> Syncing exact file times to the cluster copy..."
python3 "$REPO/cluster/sync_mtimes.py" dump "$MAC_DATASET" \
    | remote "python3 ~/${REMOTE_REPO}/cluster/sync_mtimes.py apply ~/${REMOTE_DATASET}"

# --- Step 3: Submit the job ---
echo ""
echo "==> Submitting SLURM job..."
remote "mkdir -p ~/${REMOTE_REPO}/cluster/runs"

SBATCH_CMD="cd ~/${REMOTE_REPO} && export XDG_CACHE_HOME=~/${REMOTE_BASE}/.cache"
if [ "$COCKTAIL_ANSWER" = "r" ]; then
    SBATCH_CMD="$SBATCH_CMD && sbatch --account=${ACCOUNT} --export=ALL,COCKTAIL_ANSWER=r cluster/run_notebook.sbatch"
else
    SBATCH_CMD="$SBATCH_CMD && sbatch --account=${ACCOUNT} cluster/run_notebook.sbatch"
fi

SUBMIT_OUTPUT=$(remote "$SBATCH_CMD")
echo "    $SUBMIT_OUTPUT"

JOBID=$(echo "$SUBMIT_OUTPUT" | grep -oE '[0-9]+$')
if [ -z "$JOBID" ]; then
    echo "ERROR: Could not parse job ID from sbatch output."
    exit 1
fi
echo "    Job ID: $JOBID"

# --- Step 4: Poll until job finishes ---
echo ""
echo "==> Waiting for job $JOBID to finish (polling every 60s)..."
echo "    You can also check: ssh ${CLUSTER_USER}@${CLUSTER} squeue -u ${CLUSTER_USER}"
echo ""

while true; do
    STATUS=$(remote "squeue -j $JOBID -h -o '%T' 2>/dev/null" || echo "")
    if [ -z "$STATUS" ]; then
        echo "    Job $JOBID is no longer in the queue. Checking result..."
        break
    fi
    NOW=$(date '+%H:%M:%S')
    echo "    [$NOW] Status: $STATUS"
    sleep 60
done

JOB_RESULT=$(remote "sacct -j $JOBID --format=State --noheader -P | head -1" || echo "UNKNOWN")
JOB_RESULT=$(echo "$JOB_RESULT" | tr -d '[:space:]')
echo "    Final state: $JOB_RESULT"

if [ "$JOB_RESULT" != "COMPLETED" ]; then
    echo ""
    echo "WARNING: Job did not complete successfully (state: $JOB_RESULT)."
    echo "Check the log: ssh ${CLUSTER_USER}@${CLUSTER} cat ~/${REMOTE_REPO}/cluster/runs/cocktail-trfs-${JOBID}.out"
    if [ "$SKIP_DOWNLOAD" != "1" ]; then
        echo "Skipping download of results."
    fi
    exit 1
fi

# --- Step 5: Download results ---
if [ "$SKIP_DOWNLOAD" != "1" ]; then
    echo ""
    echo "==> Downloading derivatives (TRF cache) to Mac..."
    rsync -avP -e "ssh $SSH_OPTS" \
        "${CLUSTER_USER}@${CLUSTER}:${REMOTE_DATASET}/cocktail/bids/derivatives/" \
        "$MAC_DATASET/cocktail/bids/derivatives/"
    # Match the Mac copy's file times to the cluster's (which may be whole
    # seconds), so the downloaded cache fingerprints match the Mac's inputs.
    remote "python3 ~/${REMOTE_REPO}/cluster/sync_mtimes.py dump ~/${REMOTE_DATASET}" \
        | python3 "$REPO/cluster/sync_mtimes.py" apply "$MAC_DATASET"
    echo "    Download complete."
else
    echo ""
    echo "==> Skipping download (SKIP_DOWNLOAD=1)."
fi

echo ""
echo "==> Done! The TRF cache is now on your Mac."
echo "   Open cortical_analysis.ipynb and run it -- fits will load from cache."
