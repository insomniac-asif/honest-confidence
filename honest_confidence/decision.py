"""Decision glue — compose grounding + refuter + calibration into one honest gate.

This is the layer an agent actually calls before it is allowed to assert an answer. It
sequences the three deterministic honesty mechanisms in the order that fails safest:

    1. GROUNDING (grounding.is_grounded) — does the answer resolve to enough real, distinct
       supporting endpoints? If not, ABSTAIN. You cannot calibrate confidence in a claim
       that is not anchored to anything.
    2. REFUTATION (refuter.refute) — a default-drop skeptic. If the claim is spurious,
       trivial, coincidental, or an unfalsifiable cross-domain analogy, ABSTAIN.
    3. CALIBRATION (calibration.calibrate_confidence) — only for a claim that survived both
       gates, deflate the stated confidence toward the model's MEASURED accuracy so it never
       oversells.

THE HONEST GENERALIZATION (vs. the source agent): in the running agent these three checks
were entangled with each other and with infrastructure (an Ollama skeptic, a SQLite KB, a
mind_cache). Here every model / store dependency is pushed down into the composed modules as
injectable params (resolver, judge_fn) — decide itself is pure control flow with
no I/O, so the whole gate is reproducible and the eval can measure it directly.

decide is a GATE, not an oracle: it never invents an answer. The caller supplies the
actual answer text via answer= and decide returns it unchanged when the claim clears
both gates, or None (with a legible reason) when the honest move is to abstain.

Fail-safe: on any internal error decide ABSTAINS — the safe direction for an honesty
layer — and never raises into the caller.
"""
from __future__ import annotations

from typing import Callable, Iterable, List, Optional

from . import calibration, grounding, refuter


def decide(
    question: str,
    raw_conf: float,
    evidence: Iterable,
    measured_rate: float,
    graded: Optional[int] = None,
    judge_fn: Optional[Callable[[str, List], dict]] = None,
    resolver: Optional[Callable[[object], bool]] = None,
    min_endpoints: int = 2,
    margin: float = 0.15,
    answer: Optional[object] = None,
    rooms: Optional[List] = None,
) -> dict:
    """Gate and calibrate an answer, returning a full legible verdict.

    Returns a dict::

        {answer: <answer or None>, abstain: bool, calibrated_conf: float, reason: str}

    Flow (each stage can only ABSTAIN, never upgrade a later stage's doubt):

      * grounding.is_grounded(question, evidence, resolver, min_endpoints) — if the claim
        does not resolve to >= min_endpoints distinct real supports, ABSTAIN with
        calibrated_conf=0.0 and a "ungrounded: ..." reason.
      * refuter.refute(question, evidence, judge_fn, rooms) — if the claim is spurious or
        analogy-quarantined, ABSTAIN with calibrated_conf=0.0 and the refuter's reason.
      * otherwise ANSWER: calibrated_conf is
        calibration.calibrate_confidence(raw_conf, measured_rate, graded, margin)[0] and
        answer is passed through unchanged (the caller owns the answer text; this gate
        only decides whether it may be asserted and at what confidence).

    question is used as the claim text for both gates. evidence is the list of cited
    supports (also the default endpoints handed to the refuter). rooms optionally scopes
    the analogy quarantine; if omitted the refuter derives rooms from evidence.

    Fail-safe: any error ABSTAINS (answer=None, calibrated_conf=0.0) with a legible
    reason — it never raises, and it fails toward silence.
    """
    try:
        claim = str(question or "").strip()
        if not claim:
            return {"answer": None, "abstain": True, "calibrated_conf": 0.0,
                    "reason": "ungrounded: empty question — nothing to assert, abstaining"}

        # -- 1. grounding gate --------------------------------------------------
        grounded, ground_reason = grounding.is_grounded(
            claim, evidence, resolver=resolver, min_endpoints=min_endpoints)
        if not grounded:
            return {"answer": None, "abstain": True, "calibrated_conf": 0.0,
                    "reason": "ungrounded: " + ground_reason}

        # -- 2. refutation gate (default-drop) ---------------------------------
        endpoints = grounding._as_list(evidence)
        verdict = refuter.refute(claim, endpoints, judge_fn=judge_fn, rooms=rooms)
        if verdict.get("spurious") or verdict.get("quarantined"):
            tag = "quarantined" if verdict.get("quarantined") else "refuted"
            return {"answer": None, "abstain": True, "calibrated_conf": 0.0,
                    "reason": "%s: %s" % (tag, verdict.get("reason") or "default-drop")}

        # -- 3. answer + calibrate ---------------------------------------------
        cal, note = calibration.calibrate_confidence(
            raw_conf, measured_rate, graded=graded, margin=margin)
        # Surface the grounding verdict on the answer too (not just on abstain), so a
        # caller can SEE what supported the claim and why it earned this confidence —
        # legibility, not a change to the number.
        return {"answer": answer, "abstain": False, "calibrated_conf": cal,
                "grounding": ground_reason,
                "reason": "answered; grounded and survived refutation; %s" % note}
    except Exception as exc:
        return {"answer": None, "abstain": True, "calibrated_conf": 0.0,
                "reason": "decision-error (%s) — abstaining (fail toward silence)"
                          % type(exc).__name__}
