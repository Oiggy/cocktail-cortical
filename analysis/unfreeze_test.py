"""Does eelbrain's boosting `unfreeze` option improve held-out prediction in the cocktail data?

A side experiment on the cluster. It touches none of the existing results:
- It runs in its own copy of the dataset root (made by cluster/make_unfreeze_root.sh: the preprocessed EEG
  copied, everything else linked), so eelbrain's cache and preprocessing files of the real dataset are
  never written to.
- Its fits and summary are saved in <root>_unfreeze/derivatives/results_unfreeze/.
- analysis/experiment.py, parameters.py and the notebooks are only imported, never changed.

For every cocktail condition one model is fitted with the same boosting settings as the main analysis
(BOOSTING_SETTINGS, PARTITIONS, PARAMETERS) twice, in the same eelbrain build: as is (`unfreeze=0`) and with
`unfreeze=-1` (all predictors included again after they have all stopped, until that does not lower the
validation error any more). The clean-speech model for 'clean', FULL_BASE for the two-talker conditions.
Compared: the held-out correlation r (the test partition of the cross-validation, which no fitting or
stopping decision has seen), per subject, averaged over the ROI (roi.txt of the real results) and over all sensors.

Run by cluster/run_unfreeze_test.sbatch (not meant to be run by hand). Each fit is saved as soon as it is done,
so after a TIMEOUT submit again and only what is missing is fitted.
UNFREEZE_EPOCHS="diotic binaural" limits the conditions.
"""
import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ['COCKTAIL_RESULTS'] = 'remote'  # experiment.py then builds no pipeline of its own

import numpy
from scipy import stats

import experiment as ex
from parameters import FULL_BASE, FULL_CLEAN, PARAMETERS, PARTITIONS, ROI_FALLBACK

ESTIMATORS = {'none': 'boosting', 'unfreeze': 'boosting_unfreeze'}
ALL_EPOCHS = ('diotic', 'dichotic', 'dichotic-left', 'dichotic-right', 'binaural', 'clean')
MODEL = {'clean': FULL_CLEAN}  # every other condition: FULL_BASE


def log(*args):
    print(time.strftime('%F %T'), *args, flush=True)


def main():
    real_root = Path(ex.DATA_ROOT)
    root = Path(os.environ.get('UNFREEZE_ROOT', f'{real_root}_unfreeze'))
    if not (root / 'derivatives' / 'mne').is_dir():
        sys.exit(f"{root} is not set up: run cluster/make_unfreeze_root.sh first")
    if root.resolve() == real_root.resolve():
        sys.exit("Refusing to run in the real dataset root")
    out_dir = root / 'derivatives' / 'results_unfreeze'
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = os.environ.get('UNFREEZE_EPOCHS', ' '.join(ALL_EPOCHS)).split()

    import eelbrain
    from eelbrain.pipeline import Boosting
    log(f"eelbrain {eelbrain.__version__} from {Path(eelbrain.__file__).parent}")
    import inspect
    if 'unfreeze' not in inspect.signature(eelbrain.boosting).parameters:
        sys.exit("This eelbrain has no `unfreeze` parameter: use the cocktail-cortical-unfreeze environment")

    # The pipeline works in the copy: experiment.py's methods read DATA_ROOT when they run
    ex.DATA_ROOT = str(root)
    settings = {**ex.BOOSTING_SETTINGS, 'partitions': PARTITIONS}
    estimators = {'boosting': Boosting(**settings), 'boosting_unfreeze': Boosting(**settings, unfreeze=-1)}
    cls = type(ex.BinauralCocktail.__name__, (ex.BinauralCocktail,), {'estimators': estimators, '__module__': ex.__name__})
    e = cls(str(root))
    log(f"settings: {settings}; PARAMETERS: {PARAMETERS}; results in {out_dir}")

    for epoch in epochs:
        model = MODEL.get(epoch, FULL_BASE)
        for label, estimator in ESTIMATORS.items():
            path = out_dir / f'fit-{epoch}-{label}.pickle'
            if path.exists():
                log(f"{path.name}: done already")
                continue
            log(f"fitting {epoch}, unfreeze={label}: {model}")
            t0 = time.time()
            ds = e.load_trfs('all', model, epoch=epoch, estimator=estimator, **PARAMETERS)
            r = ds['r']
            result = {
                'epoch': epoch, 'model': model, 'estimator': estimator, 'settings': settings, 'parameters': PARAMETERS,
                'subjects': [str(s) for s in ds['subject']], 'sensors': list(r.sensor.names),
                'r': numpy.asarray(r.x), 'ev': numpy.asarray(ds['ev'].x), 'seconds': time.time() - t0,
                'eelbrain': eelbrain.__version__,
            }
            part = path.with_name(path.name + '.part')
            with open(part, 'wb') as file:
                pickle.dump(result, file, protocol=pickle.HIGHEST_PROTOCOL)
            part.replace(path)
            log(f"saved {path.name} ({result['seconds'] / 60:.1f} min); mean r over subjects and sensors {result['r'].mean():.5f}")

    summarize(real_root, out_dir, epochs)


def read_roi(real_root):
    "The ROI of the main analysis (read only), else the a priori ROI"
    path = real_root / 'derivatives' / 'results' / 'roi.txt'
    if path.exists():
        roi = [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith('#')]
        if roi:
            return roi, str(path)
    return list(ROI_FALLBACK), 'ROI_FALLBACK (no roi.txt found)'


def summarize(real_root, out_dir, epochs):
    roi, roi_source = read_roi(real_root)
    lines = [
        "Boosting with and without `unfreeze=-1`: held-out correlation r (test partition), per subject",
        f"ROI: {len(roi)} sensors from {roi_source}",
        "Mean r over the region, mean over subjects; difference = unfreeze - none, paired t-test over subjects (two-sided);",
        "'better' = subjects whose r is higher with unfreeze; 'sensors' = share of sensors with a higher group-mean r.",
        '',
    ]
    table = {}
    for epoch in epochs:
        fits = {}
        for label in ESTIMATORS:
            path = out_dir / f'fit-{epoch}-{label}.pickle'
            if path.exists():
                fits[label] = pickle.loads(path.read_bytes())
        if len(fits) < 2:
            lines.append(f"{epoch}: not finished")
            continue
        a, b = fits['none'], fits['unfreeze']
        if a['subjects'] != b['subjects'] or a['sensors'] != b['sensors']:
            lines.append(f"{epoch}: the two fits differ in subjects or sensors, not compared")
            continue
        index = [i for i, name in enumerate(a['sensors']) if name in roi]
        table[epoch] = {}
        for region, sel in (('ROI', index), ('all sensors', slice(None))):
            ra, rb = a['r'][:, sel].mean(axis=1), b['r'][:, sel].mean(axis=1)
            d = rb - ra
            t, p = stats.ttest_rel(rb, ra)
            sensors = float((b['r'].mean(axis=0) > a['r'].mean(axis=0)).mean())
            table[epoch][region] = dict(none=float(ra.mean()), unfreeze=float(rb.mean()), diff=float(d.mean()), t=float(t), p=float(p),
                                        better=int((d > 0).sum()), n=len(d), sensors_better=sensors)
            lines.append(
                f"{epoch:15s} {region:12s} none {ra.mean():.5f}  unfreeze {rb.mean():.5f}  diff {d.mean():+.5f} "
                f"({100 * d.mean() / ra.mean():+.2f}%)  t({len(d) - 1}) = {t:.2f}, p = {p:.3f}  "
                f"better in {int((d > 0).sum())}/{len(d)} subjects, {100 * sensors:.0f}% of sensors")
        lines.append(f"{'':15s} fit time: none {a['seconds'] / 60:.0f} min, unfreeze {b['seconds'] / 60:.0f} min")
    text = '\n'.join(lines)
    (out_dir / 'summary.txt').write_text(text + '\n')
    (out_dir / 'summary.json').write_text(json.dumps({'roi': roi, 'table': table}, indent=1))
    log('\n' + text)


if __name__ == '__main__':
    main()
