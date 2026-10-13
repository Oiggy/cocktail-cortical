"""Write hand spot-check sheets for the automatic labels.

    python -m artifact_cleaning.scripts.make_review_sheets --subjects 01 05 09 --n-per-type 20

For each subject and label kind, writes PNGs plus review.csv to
<bids>/derivatives/artifact_cleaning/review/<kind>/sub-XX/. Open each PNG,
fill in the `verdict` column (see labels/review.py), then run
summarize_reviews. A sheet that already has verdicts is never overwritten
unless --overwrite is given - that's hand-made work.
"""
import argparse

import pandas as pd

from ..labels.review import make_review_sheet
from ..labels.store import KINDS, load_labels, review_dir
from ..preprocessing.loading import resolve_subjects
from ..preprocessing.preprocess import load_preprocessed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--subjects', nargs='+', default=['all'])
    parser.add_argument('--kinds', nargs='+', default=list(KINDS), choices=KINDS)
    parser.add_argument('--n-per-type', type=int, default=20)
    parser.add_argument('--n-unlabelled', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)

    for subject in resolve_subjects(args.subjects):
        raw, segments, _ = load_preprocessed(subject)
        for kind in args.kinds:
            out_dir = review_dir(kind, subject)
            sheet_path = out_dir / 'review.csv'
            if sheet_path.exists() and not args.overwrite:
                existing = pd.read_csv(sheet_path, keep_default_na=False)
                if (existing['verdict'].astype(str).str.strip() != '').any():
                    print(f'sub-{subject} {kind}: has verdicts already, skipping (use --overwrite to replace)')
                    continue
            events = load_labels(subject, kind, reviewed=False)
            sheet = make_review_sheet(raw, events, segments, subject, out_dir, args.n_per_type,
                                      args.n_unlabelled, seed=args.seed)
            print(f'sub-{subject} {kind}: {len(sheet)} items -> {sheet_path}')


if __name__ == '__main__':
    main()
