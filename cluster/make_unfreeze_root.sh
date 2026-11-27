#!/bin/bash
# Makes the dataset root the unfreeze test (analysis/unfreeze_test.py) runs in:
# dataset/cocktail/bids_unfreeze, next to the real bids/ folder. The test uses a
# different eelbrain build than the main analysis, and eelbrain writes to the
# dataset's derivatives while it works, so the real dataset must never be its
# working folder:
#   - the preprocessed EEG (derivatives/mne: bad channels, ICA, filtered raw) is COPIED,
#   - the raw data and the predictors are linked (they are only read),
#   - everything else (the saved results, eelbrain's own cache, figures) is left out:
#     the test builds its own.
# Safe to run again: what is there already is kept. Run on a login or compute node:
#   bash cluster/make_unfreeze_root.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(dirname "$REPO")"
SRC="$WORK/dataset/cocktail/bids"
DST="$WORK/dataset/cocktail/bids_unfreeze"
[ -d "$SRC/derivatives/mne" ] || { echo "No $SRC/derivatives/mne: nothing to copy" >&2; exit 1; }

mkdir -p "$DST/derivatives"
for entry in "$SRC"/* "$SRC"/.[!.]*; do
    [ -e "$entry" ] || continue
    name="$(basename "$entry")"
    [ "$name" = derivatives ] && continue
    [ -e "$DST/$name" ] || [ -L "$DST/$name" ] || ln -s "$entry" "$DST/$name"
done

if [ -d "$SRC/derivatives/predictors" ] && [ ! -e "$DST/derivatives/predictors" ]; then
    ln -s "$SRC/derivatives/predictors" "$DST/derivatives/predictors"
fi

if [ ! -d "$DST/derivatives/mne" ]; then
    echo "copying derivatives/mne ($(du -sh "$SRC/derivatives/mne" | cut -f1)) ..."
    rm -rf "$DST/derivatives/mne.partial"
    cp -a "$SRC/derivatives/mne" "$DST/derivatives/mne.partial"
    mv "$DST/derivatives/mne.partial" "$DST/derivatives/mne"
fi
echo "ready: $DST"
ls -la "$DST" "$DST/derivatives" | head -40
