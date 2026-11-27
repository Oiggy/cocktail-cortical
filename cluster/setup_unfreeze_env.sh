#!/bin/bash
# One-time setup on an Alliance LOGIN node (needs the internet): a second
# environment, cocktail-cortical-unfreeze, with Christian Brodbeck's
# boosting/unfreeze branch of eelbrain. The main cocktail-cortical environment
# is not touched.
#
# Usage, from the repo root:
#   bash cluster/setup_unfreeze_env.sh                 # the branch's commit below
#   bash cluster/setup_unfreeze_env.sh <commit>        # another commit of the branch
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(dirname "$REPO")"
COMMIT="${1:-9a841e138d1dfa5dcc477a8f4d544822655e8a25}"  # boosting/unfreeze, 2026-11-20 "Boosting: unfreeze parameter"
ENV_NAME=cocktail-cortical-unfreeze
SRC="$WORK/eelbrain-unfreeze"
export MAMBA_ROOT_PREFIX="$WORK/micromamba"
export XDG_CACHE_HOME="$WORK/.cache"
[ -x "$WORK/bin/micromamba" ] || { echo "No micromamba in $WORK/bin: run bash cluster/setup_env.sh first" >&2; exit 1; }
command -v gcc >/dev/null || { echo "No C compiler (gcc) on this node; eelbrain's boosting has to be compiled" >&2; exit 1; }

# The main environment's recipe without eelbrain (that comes from the branch) and under another name
TMP_YML="$(mktemp --suffix=.yml)"
grep -v -E '^\s*- eelbrain' "$REPO/environment.yml" | sed "s/^name: cocktail-cortical$/name: $ENV_NAME/" > "$TMP_YML"
if [ ! -x "$MAMBA_ROOT_PREFIX/envs/$ENV_NAME/bin/python" ]; then
    "$WORK/bin/micromamba" create -y -n "$ENV_NAME" -f "$TMP_YML" -c conda-forge
fi
rm -f "$TMP_YML"

if [ ! -d "$SRC/.git" ]; then
    git clone https://github.com/christianbrodbeck/Eelbrain "$SRC"
fi
git -C "$SRC" fetch origin boosting/unfreeze
git -C "$SRC" checkout -q "$COMMIT"
echo "eelbrain source at $(git -C "$SRC" log -1 --format='%h %ad %s' --date=short)"

# The build needs a version number, and a plain checkout has no tags to take one from
SETUPTOOLS_SCM_PRETEND_VERSION=0.43.0.dev1 "$WORK/bin/micromamba" run -n "$ENV_NAME" pip install "$SRC"

"$WORK/bin/micromamba" run -n "$ENV_NAME" python - <<'EOF'
import inspect
import eelbrain
assert 'unfreeze' in inspect.signature(eelbrain.boosting).parameters, "this eelbrain has no unfreeze parameter"
from eelbrain.pipeline import Boosting, Pipeline
print('eelbrain', eelbrain.__version__, 'from', eelbrain.__file__)
print('unfreeze available: OK')
EOF
