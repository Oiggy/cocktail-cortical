"""Detector metrics (PROJECT_PLAN.md §7): window-level and event-level."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def window_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    y_true = np.asarray(y_true, bool)
    y_pred = np.asarray(score) >= threshold
    tp, fp = np.sum(y_pred & y_true), np.sum(y_pred & ~y_true)
    fn, tn = np.sum(~y_pred & y_true), np.sum(~y_pred & ~y_true)
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * prec * sens / (prec + sens) if prec + sens and not np.isnan(prec) else 0.0
    both = y_true.any() and (~y_true).any()
    return {
        "pr_auc": average_precision_score(y_true, score) if both else np.nan,
        "roc_auc": roc_auc_score(y_true, score) if both else np.nan,
        "f1": f1, "sensitivity": sens, "specificity": spec,
        "n_pos": int(y_true.sum()), "n_neg": int((~y_true).sum()),
    }


def runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, stop) index pairs of consecutive True values."""
    padded = np.concatenate([[False], np.asarray(mask, bool), [False]])
    edges = np.flatnonzero(np.diff(padded.astype(int)))
    return list(zip(edges[::2], edges[1::2]))


def event_metrics(y_true: np.ndarray, y_pred: np.ndarray, rest: np.ndarray, step_s: float) -> dict[str, float]:
    """Event-level sensitivity and false alarms per hour of rest data, for one run.

    Inputs are per window, in time order. A true event is a run of
    consecutive positive windows; it is detected if any of its windows is
    predicted positive. A false alarm is a run of predicted-positive windows
    inside the rest blocks (labelled clean), so unannotated task periods are
    never counted against the detector.
    """
    events = runs_of(y_true)
    detected = sum(np.asarray(y_pred)[a:b].any() for a, b in events)
    false_alarms = len(runs_of(np.asarray(y_pred, bool) & np.asarray(rest, bool)))
    rest_hours = np.sum(rest) * step_s / 3600
    return {"events": len(events), "detected": int(detected), "false_alarms": false_alarms, "rest_hours": rest_hours}


def summarize_events(rows: list[dict]) -> dict[str, float]:
    events = sum(r["events"] for r in rows)
    hours = sum(r["rest_hours"] for r in rows)
    return {
        "event_sensitivity": sum(r["detected"] for r in rows) / events if events else np.nan,
        "false_alarms_per_hour": sum(r["false_alarms"] for r in rows) / hours if hours else np.nan,
    }
