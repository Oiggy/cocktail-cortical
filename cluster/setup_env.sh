#!/bin/bash
# One-time setup on an Alliance login node: installs micromamba and the
# cocktail-cortical environment (environment.yml) next to this repo.
#
# Usage, from the repo root:
#   bash cluster/setup_env.sh                # newest eelbrain dev build
#   bash cluster/setup_env.sh 0.42.0a7       # a specific eelbrain version
# Pass the same eelbrain version your MacBook has, so the cache written
# here is read back by the same version there.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(dirname "$REPO")"
EELBRAIN_VERSION="${1:-}"
export MAMBA_ROOT_PREFIX="$WORK/micromamba"

cd "$WORK"
if [ ! -x bin/micromamba ]; then
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba
fi

bin/micromamba create -y -n cocktail-cortical -f "$REPO/environment.yml"
if [ -n "$EELBRAIN_VERSION" ]; then
    bin/micromamba install -y -n cocktail-cortical \
        -c conda-forge -c conda-forge/label/eelbrain_dev "eelbrain=$EELBRAIN_VERSION"
fi

bin/micromamba run -n cocktail-cortical python -c "import eelbrain; print('eelbrain', eelbrain.__version__)"
