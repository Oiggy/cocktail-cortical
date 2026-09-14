"""
Validate the overt/masked onset predictors built by gammatone_predictors.py
against Brodbeck et al. 2020's own definitions:

    masked onset = max(source_onset - mixture_onset, 0)
    overt onset  = min(source_onset, mixture_onset)
    masked + overt == source_onset  (exact identity, for nonnegative onsets)

This doesn't re-derive anything new - it just re-applies the two equations
directly to the already-generated stimulus onsets and mixture onsets, and
checks the saved masked-8/overt-8 pickles against that, plus the near-zero
reconstruction error this implies. It also reports what fraction of total
onset magnitude ended up "masked" vs. "overt", to compare against Brodbeck's
reported ~33%/~67% split (Table/Methods, "masked and overt onset spectrograms").

Run this after gammatone_predictors.py has produced the masked-8/overt-8
predictor files.
"""
from pathlib import Path

import numpy
from eelbrain import load

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT.parent / "dataset" / "cocktail"
PREDICTOR_DIR = DATA_ROOT / "bids" / "derivatives" / "predictors"

SPEAKER = [
    ['male'] * 6 + ['female'] * 6,
    ['female'] * 6 + ['male'] * 6,
]

TOLERANCE = 1e-5


def _match_duration(a, b):
    tstop = min(a.time.tstop, b.time.tstop)
    window = (a.time.tmin, tstop)
    return a.sub(time=window), b.sub(time=window)


total_masked_norm = 0.0
total_overt_norm = 0.0
worst_masked_error = 0.0
worst_overt_error = 0.0
worst_reconstruction_error = 0.0
n_checked = 0
n_failed = 0

for list_id in (1, 2):
    fg_speakers = SPEAKER[list_id - 1]
    bg_speakers = SPEAKER[-list_id + 2]
    for i in range(1, 13):
        if i in (1, 7):
            continue  # clean segment: no competing speaker or mixture

        fg_stim = f"{fg_speakers[i - 1]}_{i}"
        bg_stim = f"{bg_speakers[i - 1]}_{i}"
        mix_stim = f"List_{list_id}_stim_{i}"

        mix_onset = load.unpickle(PREDICTOR_DIR / f"{mix_stim}~gammatone-on-8.pickle")

        for role, stim in (("attend", fg_stim), ("ignore", bg_stim)):
            source_onset = load.unpickle(PREDICTOR_DIR / f"{stim}~gammatone-on-8.pickle")
            masked = load.unpickle(PREDICTOR_DIR / f"{stim}~gammatone-on-masked-8.pickle")
            overt = load.unpickle(PREDICTOR_DIR / f"{stim}~gammatone-on-overt-8.pickle")

            source_aligned, mix_aligned = _match_duration(source_onset, mix_onset)
            expected_masked = (source_aligned - mix_aligned).clip(0, None)
            expected_overt = mix_aligned - (mix_aligned - source_aligned).clip(0, None)

            masked_aligned, _ = _match_duration(masked, expected_masked)
            overt_aligned, _ = _match_duration(overt, expected_overt)

            masked_error = float(numpy.abs(masked_aligned.x - expected_masked.x).max())
            overt_error = float(numpy.abs(overt_aligned.x - expected_overt.x).max())
            reconstruction_error = float(numpy.abs((masked_aligned.x + overt_aligned.x) - source_aligned.x).max())

            worst_masked_error = max(worst_masked_error, masked_error)
            worst_overt_error = max(worst_overt_error, overt_error)
            worst_reconstruction_error = max(worst_reconstruction_error, reconstruction_error)

            total_masked_norm += numpy.abs(masked_aligned.x).sum()
            total_overt_norm += numpy.abs(overt_aligned.x).sum()

            n_checked += 1
            failed = (
                masked_error > TOLERANCE
                or overt_error > TOLERANCE
                or reconstruction_error > TOLERANCE
            )
            if failed:
                n_failed += 1
                print(f"FAILED: list {list_id} seg {i} ({role}, {stim}): "
                      f"masked_error={masked_error:.3g}, overt_error={overt_error:.3g}, "
                      f"reconstruction_error={reconstruction_error:.3g}")

print()
print(f"pairs checked: {n_checked}")
print(f"pairs failed:  {n_failed}")
print(f"worst masked=max(source-mixture,0) error:  {worst_masked_error:.3g}")
print(f"worst overt=min(source,mixture) error:     {worst_overt_error:.3g}")
print(f"worst masked+overt == source error:        {worst_reconstruction_error:.3g}")
print()
total = total_masked_norm + total_overt_norm
print(f"overall magnitude split: overt {total_overt_norm / total:.1%}, masked {total_masked_norm / total:.1%}"
      f" (Brodbeck et al. 2020 report ~67%/~33%)")

if n_failed:
    raise SystemExit(1)
print("\nPASS: masked/overt onset predictors match the Brodbeck et al. 2020 equations.")
