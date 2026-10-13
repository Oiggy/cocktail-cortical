"""Precision and miss rate of the automatic labels, from filled-in review sheets.

    python -m artifact_cleaning.scripts.summarize_reviews

Writes group/review_summary_<kind>.tsv.
"""
import argparse

import pandas as pd

from .. import config
from ..labels.review import summarize_reviews
from ..labels.store import KINDS
from ..paths import group_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--kinds', nargs='+', default=list(KINDS), choices=KINDS)
    args = parser.parse_args(argv)

    for kind in args.kinds:
        paths = sorted((config.DERIV_ROOT / 'review' / kind).glob('sub-*/review.csv'))
        sheets = [pd.read_csv(p, keep_default_na=False) for p in paths]
        if not sheets:
            print(f'{kind}: no review sheets found')
            continue
        summary = summarize_reviews(sheets)
        if summary.empty:
            print(f'{kind}: {len(sheets)} sheet(s), none with verdicts yet')
            continue
        out = group_path(f'review_summary_{kind}.tsv')
        summary.to_csv(out, sep='\t', index=False)
        print(f'{kind} ({len(sheets)} subjects):')
        print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
