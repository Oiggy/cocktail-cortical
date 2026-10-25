"""Copy exact file modification times from one copy of the dataset to another.

rsync on macOS keeps only whole-second modification times, but eelbrain
fingerprints input files by their full-precision mtime, so after an rsync
every cached derivative (ICA, TRFs) looks stale on the other machine.

    python3 sync_mtimes.py dump  SOURCE_DIR > mtimes.tsv   # on the machine with the originals
    python3 sync_mtimes.py apply TARGET_DIR < mtimes.tsv   # on the copy

A time is only restored when the file has the same size and the same
whole-second mtime, i.e. it is the same file version rsync copied.
"""
import os
import stat
import sys
from pathlib import Path


def dump(root):
    root = Path(root)
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            path = Path(dirpath) / name
            st = path.lstat()
            if stat.S_ISREG(st.st_mode):
                sys.stdout.write(f"{st.st_mtime_ns}\t{st.st_size}\t{path.relative_to(root)}\n")


def apply(root):
    root = Path(root)
    n_set = n_same = n_skip = n_rounded = 0
    for line in sys.stdin:
        mtime_ns, size, rel = line.rstrip('\n').split('\t', 2)
        mtime_ns, size = int(mtime_ns), int(size)
        path = root / rel
        try:
            st = path.lstat()
        except FileNotFoundError:
            n_skip += 1
            continue
        if st.st_mtime_ns == mtime_ns:
            n_same += 1
        elif st.st_size == size and st.st_mtime_ns // 10**9 == mtime_ns // 10**9:
            os.utime(path, ns=(st.st_atime_ns, mtime_ns), follow_symlinks=False)
            if path.lstat().st_mtime_ns == mtime_ns:
                n_set += 1
            else:
                n_rounded += 1
        else:
            n_skip += 1
    print(f"mtimes restored: {n_set}, already exact: {n_same}, skipped (missing or changed): {n_skip}", file=sys.stderr)
    if n_rounded:
        print(f"not kept by this filesystem (whole seconds only): {n_rounded}", file=sys.stderr)


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] not in ('dump', 'apply'):
        sys.exit(__doc__)
    {'dump': dump, 'apply': apply}[sys.argv[1]](sys.argv[2])
