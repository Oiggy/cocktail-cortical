# Running the analysis on an Alliance cluster

## Per-condition notebooks: everything on the cluster

The per-condition notebooks in `analysis/` (`clean.ipynb`, `diotic.ipynb`,
`diotic_overt_masked_onsets.ipynb`, ... with shared settings in
`parameters.py`) never compute or save anything themselves. Every TRF fit
and model test is computed and kept on the cluster, in
`dataset/cocktail/bids/derivatives/results/`, and the notebooks only read
them. eelbrain's cache is never involved.

### From any browser, laptop closed: the cluster's JupyterHub

The notebooks run on the cluster itself and read the results from its
disk. Nothing runs or is stored on your own computer, so any browser works
(a phone included) and the laptop can stay closed.

1. **One-time setup** (in a cluster terminal, from the repo root, after
   `cluster/setup_env.sh`):
   ```
   bash cluster/setup_jupyterhub.sh
   ```
2. **Open the cluster's JupyterHub** in a browser and log in (Alliance
   account and Duo). Narval's address is listed on the Alliance
   documentation's "JupyterHub" page. Choose a session size and length:
   reading results and plotting needs little (for example 1 to 2 CPUs,
   8 GB, a few hours).
3. **Open** `projects/def-ibruce/iggy/cocktail-cortical/analysis/` and any
   notebook, and pick the **Python (cocktail-cortical)** kernel.
4. **Run All.** The `check_results(...)` cell at the top checks that every
   result the notebook reads is on the cluster. If any are missing it
   submits a job for that notebook by itself and stops the notebook there.
   The job runs on its own, so the browser can be closed meanwhile; the
   cluster emails when it ends, and the `cluster_queue()` cell lists it
   until then. Then Run All again.

   It does not submit when a job for the notebook is already queued, or
   when the last job already ran for the same results without saving them
   (it points to that job's log, shown by the `cluster_log()` cell, instead).
   After fixing such a failure, ask again with `submit_cluster_run('diotic')`
   (or `submit_cluster_run()` for every notebook).

Edits made in JupyterHub are in the cluster's copy of the repo and are
what submitted jobs run. To keep the Mac and GitHub in sync, commit and
push them from a JupyterHub terminal.

### From the Mac

The notebooks run on the Mac and read each result from the cluster over
SSH straight into memory: nothing is saved on the Mac and no local dataset
is needed.

1. **Connect** (in a Mac terminal, from the repo root; one Duo login,
   stays open 12 hours, run again after the laptop sleeps):
   ```
   bash cluster/connect.sh
   ```
2. **Run All** in any notebook, in VS Code or Jupyter. Missing results
   are asked for by the `check_results(...)` cell, as above. From the Mac,
   the cluster first updates its copy to GitHub's main branch, so the
   check refuses to submit while `parameters.py` or `experiment.py` have
   changes that are not pushed: commit and push them first.

### The cluster job

From a cluster terminal the job is
`sbatch --account=def-ibruce cluster/run_analysis.sbatch`, with
`--export=ALL,NOTEBOOKS="clean diotic"` to limit it.

The job saves each result as soon as it is done, so after a TIMEOUT just
submit again: what is already saved is read back in seconds. Fits already
in the cluster's eelbrain cache are read from it, so the first run after
the `cortical_analysis.ipynb` jobs is quick.

Each result's file name holds a hash of what defines it (model, condition,
`PARAMETERS`, `PARTITIONS`, `METRIC`, `PMIN`), with the details in a
matching `.json`. Rerunning with nothing changed reuses them; changing any
of those in `parameters.py` makes the next cluster run compute new ones.
ROI, peak windows and plot settings are applied in the notebook and need
no new run.

## cortical_analysis.ipynb: eelbrain cache

The cluster runs `analysis/cortical_analysis.ipynb` top to bottom on a
compute node, which fills the eelbrain cache (every TRF fit and model
comparison) under `dataset/cocktail/bids/derivatives/`. You copy that
back to the MacBook, and the notebook there loads the finished fits
instead of computing them.

Any of the general-purpose clusters works: **Fir, Nibi, Rorqual or
Narval**. Below, replace:

- `CLUSTER` with the cluster's address, e.g. `nibi.alliancecan.ca`
- `MAC_WORK` with the folder on your Mac that contains both
  `cocktail-cortical/` and `dataset/`

The cluster uses the same layout as the Mac: the repo and `dataset/`
side by side, in your project space:
`~/projects/def-ibruce/iggy/`.

## One-time setup

1. **Log in** (multi-factor authentication must be on in CCDB):
   ```
   ssh iggy@CLUSTER
   ```
2. **Get the code** into your project space:
   ```
   cd ~/projects/def-ibruce/iggy
   git clone https://github.com/Oiggy/cocktail-cortical.git
   ```
   If the repo is private, GitHub asks for a password: use a personal
   access token (GitHub > Settings > Developer settings > Tokens), or
   add an SSH key from the cluster to GitHub and clone with
   `git@github.com:Oiggy/cocktail-cortical.git`.
3. **Upload the dataset** from the Mac (in a Mac terminal). Check first
   that your ethics approval allows storing the data on Alliance servers.
   ```
   rsync -avP MAC_WORK/dataset/ iggy@CLUSTER:projects/def-ibruce/iggy/dataset/
   ```
   For a very large upload, Globus (globus.alliancecan.ca) is faster
   and resumes by itself.
4. **Find the eelbrain version on the Mac**, so the cluster uses the
   same one (the cache has to be read back by the same version):
   ```
   conda activate cocktail-cortical
   python -c "import eelbrain; print(eelbrain.__version__)"
   ```
5. **Build the environment on the cluster** (on the login node, takes
   10 to 20 minutes), passing that version:
   ```
   cd ~/projects/def-ibruce/iggy/cocktail-cortical
   bash cluster/setup_env.sh 0.42.0a7
   ```
   This installs micromamba and the `cocktail-cortical` environment
   from `environment.yml` next to the repo. (The Alliance generally
   recommends their own Python modules over conda; conda is used here
   because the eelbrain dev build is only published there.)

## Each run

1. **Update the code** on the cluster if it changed:
   ```
   cd ~/projects/def-ibruce/iggy/cocktail-cortical
   git pull
   ```
2. **Submit the job:**
   ```
   sbatch --account=def-ibruce cluster/run_notebook.sbatch
   ```
   The preprocessing cells ask whether to redo subjects that were
   already done. The job answers "skip" by default. To redo them (for
   example after changing the bad-channel or ICA settings), submit with:
   ```
   sbatch --account=def-ibruce --export=ALL,COCKTAIL_ANSWER=r cluster/run_notebook.sbatch
   ```
3. **Watch it:**
   ```
   squeue -u iggy                                  # queued / running
   tail -f cluster/runs/cocktail-trfs-JOBID.out    # live log
   ```
   You also get an email when it ends or fails. The executed notebook,
   with all outputs, is saved as `cluster/runs/cortical_analysis_JOBID.ipynb`.
4. **Copy the results back** (in a Mac terminal):
   ```
   rsync -avP iggy@CLUSTER:projects/def-ibruce/iggy/dataset/cocktail/bids/derivatives/ MAC_WORK/dataset/cocktail/bids/derivatives/
   ```
5. **On the Mac**, `git pull`, open the notebook in VS Code and run it
   as usual. The fits now load from the cache. Check once that the
   first `load_trfs` cell finishes in seconds; if it starts fitting
   instead, the two machines' settings or eelbrain versions differ.

## Resources and limits

The job asks for 32 cores, 128 GB of memory and 2 days
(`#SBATCH` lines in `run_notebook.sbatch`). After a run, `seff JOBID`
shows what it actually used; lower the request if it used much less
(smaller jobs start sooner), or raise `--mem` if it ran out. If a job
hits the time limit, just submit it again: finished fits are cached,
so it continues where it stopped.

Inside the job, eelbrain uses exactly the cores the job was given
(`experiment.py` reads `SLURM_CPUS_PER_TASK`), not every core on the
node.
