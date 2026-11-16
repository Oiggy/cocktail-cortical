#!/bin/bash
# Downloads the Fuglsang, Dau & Hjortkjaer 2017 (NeuroImage 156:435-444)
# EEG and audio dataset for auditory attention decoding (Zenodo record
# 1199011) into dataset/dtu/ next to dataset/cocktail/, on the cluster.
#
# Runs on a LOGIN node: Narval's compute nodes have no internet. Started in
# the background by dtu_download() in a notebook (or by hand, from the repo
# root: nohup bash dtu/download.sh > dataset/dtu/download.log 2>&1 &).
# Safe to run again: finished files are skipped, a cut-off download resumes,
# every file is checked against Zenodo's md5 before it is unzipped, and each
# zip is deleted once unzipped (keeps the disk use to one copy, ~20 GB).
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(dirname "$REPO")"
DTU="$WORK/dataset/dtu"
RECORD=1199011
mkdir -p "$DTU/raw"
cd "$DTU/raw"
echo "=== $(date '+%F %T') downloading Zenodo record $RECORD into $DTU/raw"

# The record's files: name, md5, size, download link (from Zenodo's API)
curl -sSfL "https://zenodo.org/api/records/$RECORD" -o record.json || { echo "FAILED: cannot reach zenodo.org"; exit 1; }
python3 - > files.tsv <<'EOF'
import json
record = json.load(open('record.json'))
for f in record['files']:
    md5 = f['checksum'].split(':', 1)[1] if ':' in f['checksum'] else f['checksum']
    link = f['links'].get('self') or f['links'].get('content')
    print(f['key'], md5, f['size'], link, sep='\t')
EOF
cat files.tsv | awk -F'\t' '{printf "  %-28s %6.2f GB\n", $1, $3 / 1e9}'

status=0
while IFS=$'\t' read -r name md5 size link; do
    done_flag=".done-$name"
    if [ -e "$done_flag" ]; then echo "--- $name: already done"; continue; fi
    echo "--- $name ($(date '+%T'))"
    for attempt in 1 2 3 4 5; do
        curl -sSfL -C - --retry 5 --retry-delay 30 -o "$name" "$link" && break
        echo "    attempt $attempt failed; retrying in 60 s"; sleep 60
    done
    got=$(md5sum "$name" | cut -d' ' -f1)
    if [ "$got" != "$md5" ]; then
        echo "    FAILED: md5 $got, expected $md5 (deleted; run again)"; rm -f "$name"; status=1; continue
    fi
    echo "    md5 ok"
    case "$name" in
        *.zip) unzip -q -o "$name" -d "$DTU" && rm -f "$name" && echo "    unzipped into $DTU" \
                   || { echo "    FAILED: unzip"; status=1; continue; } ;;
        *) mv -f "$name" "$DTU/" ;;
    esac
    touch "$done_flag"
done < files.tsv

# mat-io reads MATLAB table objects (the trial table expinfo), which scipy cannot;
# installed here because only login nodes can reach PyPI
export MAMBA_ROOT_PREFIX="$WORK/micromamba"
"$WORK/bin/micromamba" run -n cocktail-cortical python -m pip install -q mat-io \
    && echo "--- mat-io installed" || echo "--- mat-io not installed (expinfo may not be readable)"

echo "=== $(date '+%F %T') finished; contents of $DTU:"
du -sh "$DTU"/* 2>/dev/null | sed 's/^/  /'
[ $status -eq 0 ] && echo "DOWNLOAD COMPLETE" || echo "DOWNLOAD INCOMPLETE: run dtu_download() again"
exit $status
