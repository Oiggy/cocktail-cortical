"""Download Das et al. Dataset I v2.0 (KU Leuven AAD dataset) from Zenodo.

Record 4004271 (DOI 10.5281/zenodo.4004271), CC BY-NC-SA 4.0. Never the
withdrawn v1 record 3377911 (PROJECT_PLAN.md §3).

File names, sizes and md5 checksums come from Zenodo's API, so nothing is
hard-coded. Files already present with the right checksum are skipped, and
each download goes to a .part file that is only renamed once its checksum
matches, so an interrupted run can simply be restarted.

    python scripts/01_download_dataset1.py                     # everything (~5.2 GB)
    python scripts/01_download_dataset1.py --only README.txt.txt S1.mat
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import urllib.request
from pathlib import Path

from adr.config import load_paths, package_versions

RECORD_ID = 4004271
API_URL = f"https://zenodo.org/api/records/{RECORD_ID}"
CHUNK = 1 << 20

log = logging.getLogger("download_dataset1")


def list_files() -> list[dict]:
    with urllib.request.urlopen(API_URL) as response:
        record = json.load(response)
    if record["metadata"].get("doi") != f"10.5281/zenodo.{RECORD_ID}":
        raise RuntimeError(f"Unexpected record metadata: {record['metadata'].get('doi')}")
    return [
        {
            "name": f["key"],
            "size": f["size"],
            "md5": f["checksum"].removeprefix("md5:"),
            "url": f["links"]["self"],
        }
        for f in record["files"]
    ]


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def download(entry: dict, out_dir: Path) -> None:
    target = out_dir / entry["name"]
    if target.exists() and md5sum(target) == entry["md5"]:
        log.info("%s already present, checksum ok", entry["name"])
        return

    part = target.with_name(target.name + ".part")
    log.info("downloading %s (%.1f MB)", entry["name"], entry["size"] / 1e6)
    digest = hashlib.md5()
    with urllib.request.urlopen(entry["url"]) as response, open(part, "wb") as f:
        while chunk := response.read(CHUNK):
            f.write(chunk)
            digest.update(chunk)
    if digest.hexdigest() != entry["md5"]:
        part.unlink()
        raise RuntimeError(f"Checksum mismatch for {entry['name']}")
    part.rename(target)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", metavar="FILE", help="download only these files")
    args = parser.parse_args(argv)

    out_dir = load_paths()["dataset1"]
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = load_paths()["results"] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_dir / "01_download_dataset1.log")],
    )
    log.info("package versions: %s", package_versions())

    files = list_files()
    if args.only:
        unknown = set(args.only) - {f["name"] for f in files}
        if unknown:
            sys.exit(f"Not in record {RECORD_ID}: {sorted(unknown)}")
        files = [f for f in files if f["name"] in args.only]

    log.info("%d files, %.2f GB -> %s", len(files), sum(f["size"] for f in files) / 1e9, out_dir)
    for entry in sorted(files, key=lambda f: f["name"]):
        download(entry, out_dir)
    log.info("done")


if __name__ == "__main__":
    main()
