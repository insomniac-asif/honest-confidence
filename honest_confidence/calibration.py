"""Confidence calibration — deflate a self-reported confidence toward a MEASURED accuracy.

Extracted from a running local AI agent (an internal reasoning module) and generalized for public,
reproducible use. The core rule is one line:

    calibrated = min(raw, measured_rate + margin)

with a hard invariant: **it never inflates** (calibrated <= raw, always). The point is
epistemic humility — an agent should not state 0.95 confidence when the class of thing it
is doing is only right ~62% of the time.

THE HONEST GENERALIZATION (vs. the source agent): in the original system `measured_rate`
was the *owner's subjective approve-rate* — useful in situ, but not ground truth. Here it
is a PARAMETER you fit on a held-out validation split (`fit_measured_rate`), i.e. the
model's *actual measured accuracy*. That swap is what makes calibrated confidence honest
rather than self-referential, and it is exactly what the eval measures (raw vs. calibrated
ECE on a labeled benchmark).
"""
from __future__ import annotations

from typing import Iterable, Tuple

DEFAULT_MARGIN = 0.15        # measured_rate + margin is a CAP, so cal can only be pulled down
DEFAULT_MIN_GRADED = 8       # below this many graded items, treat the rate as thin history
COLD_START_PRIOR = 0.30      # cautious prior when we have too little data to trust the rate


def _clamp01(x) -> float:
    try:
        return max(0.0, min(1.0, float(x or 0.0)))
    except Exception:
        return 0.0


def calibrate_confidence(
    raw_conf: float,
    measured_rate: float,
    graded: int | None = None,
    margin: float = DEFAULT_MARGIN,
    min_graded: int = DEFAULT_MIN_GRADED,
    cold_start_prior: float = COLD_START_PRIOR,
) -> Tuple[float, str]:
    """Return ``(calibrated, note)``.

    ``calibrated`` is ``raw_conf`` deflated toward ``measured_rate`` and is GUARANTEED
    ``<= raw_conf`` (never inflates). If ``graded`` is given and below ``min_graded``, the
    rate is treated as thin history and capped at ``cold_start_prior`` so a scant sample
    cannot oversell. ``note`` discloses which regime applied (for transparency).

    Fail-safe: on any internal error it returns ``(clamped_raw, "")`` — it never raises and
    never oversells, so a caller can always trust the value.
    """
    try:
        raw = _clamp01(raw_conf)
        rate = _clamp01(measured_rate)
        if graded is not None and graded < min_graded:
            rate = min(rate, cold_start_prior)
            note = "thin history (n=%s) — capped at cautious %.0f%% prior" % (graded, rate * 100)
        else:
            n = "" if graded is None else " (n=%s)" % graded
            note = "deflated toward measured %.0f%% accuracy%s" % (rate * 100, n)
        cal = round(min(raw, rate + margin), 2)
        if cal > raw:                       # belt-and-suspenders: never inflate
            cal = raw
        return cal, note
    except Exception:
        return _clamp01(raw_conf), ""


def fit_measured_rate(preds: Iterable[Tuple[float, bool]]) -> Tuple[float, int]:
    """Fit ``measured_rate`` from a held-out validation set.

    ``preds`` is an iterable of ``(confidence, is_correct)`` pairs. Returns
    ``(accuracy, n)`` where ``accuracy`` is the fraction correct — the value the cap
    deflates confidence toward. This is the HONEST swap versus the source agent, which used
    a subjective owner approve-rate instead of measured accuracy. Empty input returns the
    cold-start prior.
    """
    pairs = [(c, bool(ok)) for c, ok in preds]
    if not pairs:
        return COLD_START_PRIOR, 0
    acc = sum(1 for _, ok in pairs if ok) / len(pairs)
    return acc, len(pairs)


def corroborated_cap(
    raw_conf: float,
    measured_rate: float,
    n_endpoints: int,
    base_margin: float = DEFAULT_MARGIN,
    per_endpoint: float = 0.03,
    max_margin: float = 0.30,
) -> Tuple[float, str]:
    """EXPERIMENTAL / NOT YET MEASURED — a corroboration-aware variant of the flat cap.

    Idea: a claim backed by MORE distinct real endpoints should be allowed a slightly
    higher confidence ceiling than one backed by the bare minimum. The margin grows a
    little per supporting endpoint beyond the required 2, bounded by ``max_margin``::

        margin = min(max_margin, base_margin + per_endpoint * max(0, n_endpoints - 2))
        calibrated = min(raw, measured_rate + margin)     # still never inflates

    The motivation is the eval's central negative result: the flat cap fixes ECE but
    flattens per-item discrimination (AUROC → chance). If endpoint count correlates with
    correctness, letting it modulate the cap would restore some ranking signal — the
    thing a single global cap throws away.

    ⚠️ **This is not validated.** On the closed-book TruthfulQA eval the grounding gate
    fires rarely (evidence is the model's own justifications, not real retrieval), so
    endpoint count barely varies and this cannot be measured there. It needs a
    retrieval-grounded eval, and it must be reported as a WIN only after AUROC-vs-ECE is
    measured against the flat cap (see WRITEUP §7). Until then it is a designed,
    implemented, *unmeasured* direction — do not claim it improves anything.

    Returns ``(calibrated, note)``. Never inflates; fail-safe returns clamped raw.
    """
    try:
        raw = _clamp01(raw_conf)
        rate = _clamp01(measured_rate)
        n = max(0, int(n_endpoints))
        margin = min(max_margin, base_margin + per_endpoint * max(0, n - 2))
        cal = round(min(raw, rate + margin), 2)
        if cal > raw:
            cal = raw
        note = ("[experimental] corroboration-aware: %d endpoints -> margin %.2f, "
                "capped at measured %.0f%%+margin (UNMEASURED)" % (n, margin, rate * 100))
        return cal, note
    except Exception:
        return _clamp01(raw_conf), ""


def calibration_health(measured_rate: float, graded: int,
                       margin: float = DEFAULT_MARGIN,
                       min_graded: int = DEFAULT_MIN_GRADED) -> dict:
    """LLM-free snapshot of the calibration signal (for a dashboard / report).

    Honest: ``cold_start`` stays True until ``graded >= min_graded``, and ``note`` says so.
    """
    rate = _clamp01(measured_rate)
    cold = graded < min_graded
    if cold:
        note = ("thin history (n=%s) — confidence capped at a cautious %.0f%% prior; earns "
                "trust as more items are graded" % (graded, min(rate, COLD_START_PRIOR) * 100))
    else:
        note = "confidence is deflated toward the measured %.0f%% accuracy (n=%s)" % (rate * 100, graded)
    return {
        "measured_rate": round(rate, 3),
        "graded": graded,
        "cold_start": cold,
        "min_graded": min_graded,
        "margin": margin,
        "rule": "calibrated = min(raw, measured_rate + margin)",
        "note": note,
    }
