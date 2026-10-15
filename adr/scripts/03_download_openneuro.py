"""Download the OpenNeuro datasets: PhysioMotion (ds006386) and the phantom (ds004784).

Uses OpenNeuro's public GraphQL API to list a pinned snapshot's files and
their S3 URLs. OpenNeuro publishes sizes but not checksums, so each file is
checked against its listed size; downloads go to a .part file that is
renamed only when complete, so an interrupted run can be restarted. The full
file list (path, size, url) is written to results/logs for reproducibility.

What gets downloaded (PROJECT_PLAN.md §3.2-3.3):
- physiomotion: the raw BIDS EEG (sub-*/) and derivatives/Manual_Annotations,
  but not derivatives/preprocessed_BIDS (bipolar, filtered; ~14 GB, unused).
- phantom: the BIDS recordings, ground-truth stimuli, scripts and CAD, but not
  derivatives/Data/Imported or /Preprocessed (the authors' copies; ~10 GB).

    python scripts/03_download_openneuro.py physiomotion phantom
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from adr.config import load_paths, package_versions

API_URL = "https://openneuro.org/crn/graphql"
CHUNK = 1 << 20

DATASETS = {
    "physiomotion": {
        "id": "ds006386",
        "tag": "1.0.1",
        "exclude": ["derivatives/preprocessed_BIDS/*", ".datalad/*"],
    },
    "phantom": {
        "id": "ds004784",
        "tag": "1.0.4",
        # The authors' own imported/preprocessed copies (~10 GB); we work
        # from the BIDS recordings in sub-001/ instead.
        "exclude": ["derivatives/Data/Imported/*", "derivatives/Data/Preprocessed/*", ".datalad/*"],
    },
}

log = logging.getLogger("download_openneuro")


def query(dataset_id: str, tag: str, tree: str | None = None) -> list[dict]:
    files_arg = f'(tree: "{tree}")' if tree else ""
    q = (f'{{ snapshot(datasetId: "{dataset_id}", tag: "{tag}") '
         f'{{ files{files_arg} {{ id filename size directory urls }} }} }}')
    request = urllib.request.Request(
        API_URL, json.dumps({"query": q}).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        return json.load(response)["data"]["snapshot"]["files"]


def list_files(dataset_id: str, tag: str, tree: str | None = None, prefix: str = "") -> list[dict]:
    """All files in the snapshot, recursively, as {path, size, url}."""
    out = []
    for f in query(dataset_id, tag, tree):
        path = prefix + f["filename"]
        if f["directory"]:
            out += list_files(dataset_id, tag, f["id"], path + "/")
        elif f["urls"]:
            out.append({"path": path, "size": int(f["size"]), "url": f["urls"][0]})
    return out


def download(entry: dict, out_dir: Path) -> None:
    target = out_dir / entry["path"]
    if target.exists() and target.stat().st_size == entry["size"]:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    with urllib.request.urlopen(entry["url"]) as response, open(part, "wb") as f:
        while chunk := response.read(CHUNK):
            f.write(chunk)
    if part.stat().st_size != entry["size"]:
        part.unlink()
        raise RuntimeError(f"Size mismatch for {entry['path']}")
    part.rename(target)
    log.info("got %s (%.1f MB)", entry["path"], entry["size"] / 1e6)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="+", choices=sorted(DATASETS))
    parser.add_argument("--workers", type=int, default=4, help="parallel downloads")
    args = parser.parse_args(argv)

    paths = load_paths()
    log_dir = paths["results"] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_dir / "03_download_openneuro.log")],
    )
    log.info("package versions: %s", package_versions())

    for name in args.datasets:
        spec = DATASETS[name]
        out_dir = paths[name]
        files = [
            f for f in list_files(spec["id"], spec["tag"])
            if not any(fnmatch.fnmatch(f["path"], pattern) for pattern in spec["exclude"])
        ]
        with open(log_dir / f"03_{name}_{spec['id']}_{spec['tag']}_files.json", "w") as fh:
            json.dump(files, fh, indent=1)
        log.info("%s %s@%s: %d files, %.2f GB -> %s", name, spec["id"], spec["tag"],
                 len(files), sum(f["size"] for f in files) / 1e9, out_dir)
        with ThreadPoolExecutor(args.workers) as pool:
            list(pool.map(lambda entry: download(entry, out_dir), files))
        log.info("%s done", name)


if __name__ == "__main__":
    main()
