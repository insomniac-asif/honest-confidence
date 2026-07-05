"""Evaluation metrics for the honesty layer — calibration, discrimination, and cost.

These are the numbers the eval reports for **raw** vs. **honest** arms on a labeled
benchmark. Nothing here is model- or agent-specific: every function takes plain arrays of
confidences / scores / booleans and returns a float (or per-bin tuples). The metrics answer
three separate questions the README poses:

  * **Is confidence honest?** ``ece`` (expected calibration error) and ``reliability_bins``
    measure the gap between stated confidence and observed accuracy.
  * **Does the abstain signal discriminate?** ``auroc`` measures whether the score used to
    answer-vs-abstain separates right answers from wrong ones.
  * **What does honesty cost, and what does it buy?** ``abstention_rate``,
    ``accuracy_on_answered``, and ``confident_falsehood_rate`` measure the trade the layer
    makes — correct answers withheld vs. confident wrong answers avoided.

THE HONEST GENERALIZATION: a method that only reports its wins is not a measurement. Every
metric here is defined so it can show a *regression* (higher ECE, lower accuracy-on-answered,
over-abstention) just as readily as an improvement. That is the point.

Fail-safe: every function returns ``float('nan')`` on degenerate input (empty arrays, a
single class, mismatched lengths) instead of raising, so a run over a thin or malformed slice
degrades to a reported NaN rather than crashing the harness.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

NAN = float("nan")


def _as_arrays(*seqs: Sequence) -> Tuple[np.ndarray, ...]:
    """Coerce inputs to equal-length float arrays, else raise ValueError.

    Callers wrap this in try/except so a length mismatch becomes a NaN, not a crash.
    """
    arrs = [np.asarray(list(s), dtype=float) for s in seqs]
    n = len(arrs[0]) if arrs else 0
    if n == 0 or any(len(a) != n for a in arrs):
        raise ValueError("empty or mismatched-length inputs")
    return tuple(arrs)


def ece(confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10) -> float:
    """Expected Calibration Error over ``n_bins`` equal-width confidence bins.

    For each bin, the absolute gap between mean confidence and observed accuracy, weighted by
    the fraction of samples in the bin, then summed. ``0.0`` is perfect calibration; larger is
    worse. This is the headline calibration number (raw vs. honest).

    Returns ``nan`` on empty / mismatched input.
    """
    try:
        conf, corr = _as_arrays(confidences, correct)
    except Exception:
        return NAN
    conf = np.clip(conf, 0.0, 1.0)
    corr = corr.astype(bool)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(conf)
    err = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        # first bin is closed on the left so conf == 0.0 counts; all bins closed on the right
        if i == 0:
            in_bin = (conf >= lo) & (conf <= hi)
        else:
            in_bin = (conf > lo) & (conf <= hi)
        count = int(in_bin.sum())
        if count == 0:
            continue
        bin_conf = float(conf[in_bin].mean())
        bin_acc = float(corr[in_bin].mean())
        err += (count / total) * abs(bin_conf - bin_acc)
    return float(err)


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Area under the ROC curve — does ``scores`` separate the two classes in ``labels``?

    In this eval, ``scores`` is the confidence/answer signal and ``labels`` is correctness:
    a good abstain signal ranks correct answers above wrong ones (AUROC toward 1.0; 0.5 is
    chance). Uses ``sklearn.metrics.roc_auc_score``.

    Fail-safe: returns ``nan`` on empty input, on a single class present (AUROC undefined), or
    if scikit-learn is unavailable — never raises.
    """
    try:
        sc, lab = _as_arrays(scores, labels)
    except Exception:
        return NAN
    lab = lab.astype(int)
    if len(np.unique(lab)) < 2:          # single-class: AUROC is undefined
        return NAN
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        return NAN
    try:
        return float(roc_auc_score(lab, sc))
    except Exception:
        return NAN


def abstention_rate(decisions: Sequence[dict]) -> float:
    """Fraction of decisions that abstained.

    ``decisions`` is a list of ``decide()`` outputs (dicts with a truthy ``abstain`` key).
    This is the cost axis: how often the honesty layer declined to answer. Returns ``nan`` on
    empty input.
    """
    try:
        items = list(decisions)
        if not items:
            return NAN
        n_abstain = sum(1 for d in items if bool((d or {}).get("abstain")))
        return float(n_abstain) / float(len(items))
    except Exception:
        return NAN


def accuracy_on_answered(
    answers: Sequence[bool], correct: Sequence[bool], abstained: Sequence[bool]
) -> float:
    """Accuracy computed **only over items the layer actually answered**.

    ``correct[i]`` is whether item ``i`` was right; ``abstained[i]`` marks a withheld answer.
    Items where ``abstained`` is true are excluded from both numerator and denominator. This
    is the flip side of ``abstention_rate``: dropping ungrounded claims should *raise* this.

    ``answers`` is accepted for call-site symmetry with the other metrics but only its length
    is used; correctness is read from ``correct``. Returns ``nan`` if nothing was answered.
    """
    try:
        _, corr, abst = _as_arrays(answers, correct, abstained)
    except Exception:
        return NAN
    answered = ~abst.astype(bool)
    denom = int(answered.sum())
    if denom == 0:                        # everything abstained: accuracy undefined
        return NAN
    return float(corr.astype(bool)[answered].sum()) / float(denom)


def confident_falsehood_rate(
    confidences: Sequence[float], correct: Sequence[bool], threshold: float = 0.7
) -> float:
    """Fraction of **high-confidence** answers that are actually wrong.

    Of the answers stated with confidence ``>= threshold``, how many were incorrect. This is
    the harm the honesty layer exists to reduce: confident falsehoods. ``0.0`` means no
    high-confidence answer was wrong. Returns ``nan`` if no answer clears the threshold.
    """
    try:
        conf, corr = _as_arrays(confidences, correct)
    except Exception:
        return NAN
    conf = np.clip(conf, 0.0, 1.0)
    high = conf >= float(threshold)
    denom = int(high.sum())
    if denom == 0:                        # no confident answers to judge
        return NAN
    wrong = ~corr.astype(bool)
    return float((high & wrong).sum()) / float(denom)


def reliability_bins(
    confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10
) -> List[Tuple[float, float, int]]:
    """Per-bin ``(mean_confidence, accuracy, count)`` for a reliability diagram.

    Same equal-width binning as ``ece``. Empty bins are emitted as ``(nan, nan, 0)`` so the
    returned list always has ``n_bins`` entries in ascending-confidence order — a plotter can
    map it straight onto the diagonal. Returns an empty list on degenerate input.
    """
    out: List[Tuple[float, float, int]] = []
    try:
        conf, corr = _as_arrays(confidences, correct)
    except Exception:
        return out
    conf = np.clip(conf, 0.0, 1.0)
    corr = corr.astype(bool)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if i == 0:
            in_bin = (conf >= lo) & (conf <= hi)
        else:
            in_bin = (conf > lo) & (conf <= hi)
        count = int(in_bin.sum())
        if count == 0:
            out.append((NAN, NAN, 0))
        else:
            out.append((float(conf[in_bin].mean()), float(corr[in_bin].mean()), count))
    return out
