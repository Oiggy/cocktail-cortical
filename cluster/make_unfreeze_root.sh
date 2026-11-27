#!/bin/bash
# Makes the dataset root the unfreeze test (analysis/unfreeze_test.py) runs in:
# dataset/cocktail/bids_unfreeze, next to the real bids/ folder. The test uses a
# different eelbrain build than the main analysis, and eelbrain writes to the
# dataset's derivatives while it works, so the real dataset must never be its
# working folder:
#   - the preprocessed EEG (derivatives/mne: bad channels, ICA, filtered raw) is COPIED,
#   - the raw data and the predictors are linked file by file (they are only read): real
#     folders holding links to the files, because the search for the recordings
#     (mne-bids, Path.rglob) does not look inside linked folders,
#   - lock files and everything else in derivatives (the saved results, eelbrain's own
#     cache, figures) are left out: the test builds its own.
# Safe to run again: what is there already is kept. Run on a login or compute node:
#   bash cluster/make_unfreeze_root.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(dirname "$REPO")"
SRC="$WORK/dataset/cocktail/bids"
DST="$WORK/dataset/cocktail/bids_unfreeze"
[ -d "$SRC/derivatives/mne" ] || { echo "No $SRC/derivatives/mne: nothing to copy" >&2; exit 1; }
mkdir -p "$DST/derivatives"

# link_tree SRC DST: DST mirrors SRC with real folders and links to the files (no lock files)
link_tree() {
    local src="$1" dst="$2"
    # an earlier version linked whole folders: remove that link (only the link, never its target)
    if [ -L "$dst" ]; then rm "$dst"; fi
    [ -d "$dst" ] && return 0
    mkdir -p "$dst.partial"
    (cd "$src" && find . -type d) | while read -r d; do mkdir -p "$dst.partial/$d"; done
    (cd "$src" && find . \( -type f -o -type l \) ! -name '*.lock' ! -name '.DS_Store') | while read -r f; do
        ln -s "$src/${f#./}" "$dst.partial/$f"
    done
    mv "$dst.partial" "$dst"
}

for entry in "$SRC"/* "$SRC"/.[!.]*; do
    [ -e "$entry" ] || continue
    name="$(basename "$entry")"
    case "$name" in derivatives|*.lock|.DS_Store) continue ;; esac
    if [ -d "$entry" ]; then
        link_tree "$entry" "$DST/$name"
    elif [ ! -e "$DST/$name" ] && [ ! -L "$DST/$name" ]; then
        ln -s "$entry" "$DST/$name"
    fi
done
# lock files linked by an earlier version
find "$DST" -maxdepth 1 -name '*.lock' -type l -delete
find "$DST" -maxdepth 1 -name '.DS_Store' -type l -delete

if [ -d "$SRC/derivatives/predictors" ]; then
    link_tree "$SRC/derivatives/predictors" "$DST/derivatives/predictors"
fi

if [ ! -d "$DST/derivatives/mne" ]; then
    echo "copying derivatives/mne ($(du -sh "$SRC/derivatives/mne" | cut -f1)) ..."
    rm -rf "$DST/derivatives/mne.partial"
    cp -a "$SRC/derivatives/mne" "$DST/derivatives/mne.partial"
    mv "$DST/derivatives/mne.partial" "$DST/derivatives/mne"
fi
echo "ready: $DST ($(find "$DST" -name '*task-*' | wc -l) recording files visible)"
