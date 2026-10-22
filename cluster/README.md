# Running the analysis on an Alliance cluster

The cluster runs `analysis/cortical_analysis.ipynb` top to bottom on a
compute node, which fills the eelbrain cache (every TRF fit and model
comparison) under `dataset/cocktail/bids/derivatives/`. You copy that
back to the MacBook, and the notebook there loads the finished fits
instead of computing them.

Any of the general-purpose clusters works: **Fir, Nibi, Rorqual or
Narval**. Below, replace:

- `CLUSTER` with the cluster's address, e.g. `nibi.alliancecan.ca`
- `USER` with your Alliance username
- `SUPERVISOR` with your supervisor's allocation name (the part after
  `def-`, shown in CCDB under "My Account")
- `MAC_WORK` with the folder on your Mac that contains both
  `cocktail-cortical/` and `dataset/`

The cluster uses the same layout as the Mac: the repo and `dataset/`
side by side, in your project space:
`~/projects/def-SUPERVISOR/USER/`.

## One-time setup

1. **Log in** (multi-factor authentication must be on in CCDB):
   ```
   ssh USER@CLUSTER
   ```
2. **Get the code** into your project space:
   ```
   cd ~/projects/def-SUPERVISOR/USER
   git clone https://github.com/Oiggy/cocktail-cortical.git
   ```
   If the repo is private, GitHub asks for a password: use a personal
   access token (GitHub > Settings > Developer settings > Tokens), or
   add an SSH key from the cluster to GitHub and clone with
   `git@github.com:Oiggy/cocktail-cortical.git`.
3. **Upload the dataset** from the Mac (in a Mac terminal). Check first
   that your ethics approval allows storing the data on Alliance servers.
   ```
   rsync -avP MAC_WORK/dataset/ USER@CLUSTER:projects/def-SUPERVISOR/USER/dataset/
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
   cd ~/projects/def-SUPERVISOR/USER/cocktail-cortical
   bash cluster/setup_env.sh 0.42.0a7
   ```
   This installs micromamba and the `cocktail-cortical` environment
   from `environment.yml` next to the repo. (The Alliance generally
   recommends their own Python modules over conda; conda is used here
   because the eelbrain dev build is only published there.)

## Each run

1. **Update the code** on the cluster if it changed:
   ```
   cd ~/projects/def-SUPERVISOR/USER/cocktail-cortical
   git pull
   ```
2. **Submit the job:**
   ```
   sbatch --account=def-SUPERVISOR cluster/run_notebook.sbatch
   ```
   The preprocessing cells ask whether to redo subjects that were
   already done. The job answers "skip" by default. To redo them (for
   example after changing the bad-channel or ICA settings), submit with:
   ```
   sbatch --account=def-SUPERVISOR --export=ALL,COCKTAIL_ANSWER=r cluster/run_notebook.sbatch
   ```
3. **Watch it:**
   ```
   squeue -u USER                                  # queued / running
   tail -f cluster/runs/cocktail-trfs-JOBID.out    # live log
   ```
   You also get an email when it ends or fails. The executed notebook,
   with all outputs, is saved as `cluster/runs/cortical_analysis_JOBID.ipynb`.
4. **Copy the results back** (in a Mac terminal):
   ```
   rsync -avP USER@CLUSTER:projects/def-SUPERVISOR/USER/dataset/cocktail/bids/derivatives/ MAC_WORK/dataset/cocktail/bids/derivatives/
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
