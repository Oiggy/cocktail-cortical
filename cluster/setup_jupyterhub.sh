#!/bin/bash
# One-time setup on the cluster: makes the cocktail-cortical environment
# (from cluster/setup_env.sh) a kernel in the cluster's JupyterHub, so the
# per-condition notebooks can run there from any browser, with the laptop
# closed. Run it in a cluster terminal, from the repo root:
#   bash cluster/setup_jupyterhub.sh
# In JupyterHub, pick the "Python (cocktail-cortical)" kernel for the notebooks.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(dirname "$REPO")"
ENV="$WORK/micromamba/envs/cocktail-cortical"
if [ ! -x "$ENV/bin/python" ]; then
    echo "No environment at $ENV: run bash cluster/setup_env.sh first." >&2
    exit 1
fi

"$ENV/bin/python" -m ipykernel install --user --name cocktail-cortical --display-name "Python (cocktail-cortical)"

# Same caches as the cluster jobs, and the environment's own programs first
"$ENV/bin/python" - "$ENV" "$WORK/.cache" << 'EOF'
import json, os, sys
from jupyter_client.kernelspec import KernelSpecManager
env, cache = sys.argv[1], sys.argv[2]
path = os.path.join(KernelSpecManager().get_kernel_spec('cocktail-cortical').resource_dir, 'kernel.json')
spec = json.load(open(path))
spec['env'] = {'XDG_CACHE_HOME': cache, 'PATH': f"{env}/bin:{os.environ['PATH']}"}
json.dump(spec, open(path, 'w'), indent=1)
print(f"Kernel ready: {path}")
EOF
