"""Adversarial refuter — a DEFAULT-DROP gate that dismisses spurious claims before an
agent is allowed to assert them.

Extracted from a running local AI agent (the ``_mind_refute_verdict`` skeptic in
an internal privacy module / an internal reasoning module) and generalized for public, reproducible use. The source
agent proposed connections between two items in its knowledge base and handed each proposal
to a skeptic model whose only job was to argue the link was random / coincidental / trivial
/ already-obvious. Two ideas survive that generalization:

  1. **Default-drop.** A claim survives ONLY on an explicit, unambiguous not-spurious
     verdict. Anything missing, unsure, timed-out, or errored is treated as spurious — the
     bias is toward silence over a confident-but-wrong assertion. This is the honesty move:
     it costs real claims, and that cost is the point.

  2. **Deterministic analogy quarantine (the "Red Moonlight" class).** A metaphor cannot be
     refuted: "X represents / mirrors Y" is unfalsifiable, so a skeptic model kept passing
     poetic cross-domain glue. If a claim links two DIFFERENT rooms (domains) by resemblance
     language, it is dropped with a legible reason and NO model call — a zero-model gate.

THE HONEST GENERALIZATION (vs. the source agent): the skeptic was one Ollama call
(``_ollama_json``). Here the model judge is an INJECTABLE ``judge_fn`` you can supply (or
omit). With ``judge_fn=None`` the refuter runs DETERMINISTIC-ONLY — analogy quarantine plus
trivial / coincidental heuristics — so the gate is fully reproducible with no model at all,
and any model you plug in is measured against that deterministic floor rather than assumed.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

# Resemblance / metaphor language. If a claim connects two distinct rooms (domains) using
# any of these, it is an unfalsifiable analogy — quarantined deterministically, no model.
ANALOGY_MARKERS: Tuple[str, ...] = (
    "represents", "mirrors", "just as ", "signifies", "symboliz",
    "echoes", "parallels", "is like ", "much like ", "akin to",
    "in the same way", "metaphor", "resembles", "reminiscent",
)

# Phrases that make a claim TRIVIAL / already-obvious — true but contentless.
TRIVIAL_MARKERS: Tuple[str, ...] = (
    "both are", "both exist", "are both", "as expected", "obviously",
    "of course", "it is well known", "everyone knows", "by definition",
)

# Hedge words that signal a merely COINCIDENTAL link — no stated shared datum.
COINCIDENTAL_MARKERS: Tuple[str, ...] = (
    "coincidentally", "happen to", "by chance", "randomly", "might be related",
    "could be connected", "possibly linked", "seems related",
)

# What a refute() dict looks like on the fail-closed path (also the shape judge_fn returns).
_SPURIOUS_SHAPE = {
    "spurious": True, "is_trivial": False, "is_coincidental": False,
    "quarantined": False, "reason": "",
}


def _rooms_of(endpoints: Optional[List]) -> List[str]:
    """Distinct 'room' (domain) labels across endpoints. An endpoint may be a dict with a
    ``room`` key or a bare string; strings without a room contribute nothing. Never raises."""
    try:
        rooms = set()
        for e in (endpoints or []):
            if isinstance(e, dict) and e.get("room"):
                rooms.add(str(e.get("room")))
        return sorted(rooms)
    except Exception:
        return []


def is_analogy_quarantined(claim: str, rooms: Optional[List] = None) -> Tuple[bool, str]:
    """Zero-model gate: is ``claim`` an unfalsifiable cross-domain analogy?

    Returns ``(quarantined, reason)``. Quarantined iff the claim uses resemblance language
    AND spans >= 2 distinct rooms — a metaphor gluing two domains states a vibe, not a
    shared datum, so it can never be refuted and is dropped deterministically. ``rooms`` may
    be a list of room labels (strings) or of endpoint dicts (their ``room`` fields are used).
    Fail-safe: on any error returns ``(False, "")`` — a gate failure never blocks a claim by
    accident (the model judge / default-drop still guard it downstream).
    """
    try:
        low = str(claim or "").lower()
        if isinstance(rooms, (list, tuple, set)):
            room_list = sorted({str(r.get("room")) if isinstance(r, dict) else str(r)
                                for r in rooms if r})
        else:
            room_list = []
        hit = next((m for m in ANALOGY_MARKERS if m in low), None)
        if hit and len(room_list) >= 2:
            reason = ("cross-domain analogy (%s): '%s' links by resemblance, not by a shared "
                      "datum — poetic and unfalsifiable, quarantined deterministically; it "
                      "would need a concrete datum tying these domains together before it "
                      "can be considered" % (" <-> ".join(room_list), hit.strip()))
            return True, reason
        return False, ""
    except Exception:
        return False, ""


def _deterministic_flags(claim: str) -> Tuple[bool, bool, Optional[str]]:
    """Heuristic trivial / coincidental detection from claim text alone (no model).
    Returns ``(is_trivial, is_coincidental, reason_or_None)``. Never raises."""
    try:
        low = str(claim or "").lower()
        trivial = any(m in low for m in TRIVIAL_MARKERS)
        coincidental = any(m in low for m in COINCIDENTAL_MARKERS)
        reason = None
        if trivial and coincidental:
            reason = "reads as trivial/already-obvious AND merely coincidental"
        elif trivial:
            reason = "trivial/already-obvious phrasing — true but contentless"
        elif coincidental:
            reason = "hedged as coincidental — no shared datum stated"
        return trivial, coincidental, reason
    except Exception:
        return False, False, None


def refute(
    claim: str,
    endpoints: List,
    judge_fn: Optional[Callable[[str, List], dict]] = None,
    rooms: Optional[List] = None,
) -> dict:
    """Adversarially refute ``claim`` and return a full, legible verdict.

    Returns a dict with::

        {spurious: bool, is_trivial: bool, is_coincidental: bool,
         quarantined: bool, reason: str}

    A claim is cleared (``spurious=False``) ONLY when nothing flags it. DEFAULT-DROP
    governs everything else:

      * If the claim is an analogy spanning >= 2 rooms → ``quarantined=True``, spurious
        (zero-model, deterministic — see :func:`is_analogy_quarantined`).
      * Deterministic trivial / coincidental heuristics fire → spurious.
      * If ``judge_fn`` is supplied, it is called ``judge_fn(claim, endpoints) -> dict`` and
        must AFFIRMATIVELY clear all three of ``spurious``/``is_trivial``/``is_coincidental``
        as falsey. Any missing/None flag, or any truthy flag, → spurious (the skeptic did
        not clear it). If ``judge_fn`` is ``None``, the refuter is deterministic-only.

    Fail-safe: on ANY error the verdict is spurious=True (fail closed) with a reason — it
    never raises into the caller and never lets an unjudged claim through.
    """
    try:
        text = str(claim or "").strip()
        if not text:
            return {**_SPURIOUS_SHAPE, "reason": "empty claim — nothing to assert, dropped"}

        # rooms default to the endpoints themselves (they may carry room labels).
        room_source = rooms if rooms is not None else endpoints

        # -- 1. deterministic analogy quarantine (zero-model, always runs first) --
        quarantined, q_reason = is_analogy_quarantined(text, room_source)
        if quarantined:
            return {"spurious": True, "is_trivial": False, "is_coincidental": True,
                    "quarantined": True, "reason": q_reason}

        # -- 2. deterministic trivial / coincidental heuristics --
        trivial, coincidental, det_reason = _deterministic_flags(text)
        if trivial or coincidental:
            return {"spurious": True, "is_trivial": trivial, "is_coincidental": coincidental,
                    "quarantined": False,
                    "reason": det_reason or "deterministic heuristic flagged the claim"}

        # -- 3. optional injectable model judge (default-drop on anything unsure) --
        if judge_fn is None:
            return {"spurious": False, "is_trivial": False, "is_coincidental": False,
                    "quarantined": False,
                    "reason": "deterministic-only pass: no analogy, trivial, or coincidental "
                              "marker — cleared (no model judge supplied)"}

        data = judge_fn(text, endpoints)
        if not isinstance(data, dict):
            return {**_SPURIOUS_SHAPE, "is_coincidental": False,
                    "reason": "the judge returned no usable verdict (non-dict / timed out) — "
                              "default-drop, bias to silence over noise"}

        rsn = str(data.get("reason") or "").strip()
        spurious = data.get("spurious")
        coincidental_j = data.get("is_coincidental")
        trivial_j = data.get("is_trivial")

        # A missing/None flag = judge did not AFFIRMATIVELY clear it → unsure → DROP.
        if spurious is None or coincidental_j is None or trivial_j is None:
            return {"spurious": True, "is_trivial": bool(trivial_j),
                    "is_coincidental": bool(coincidental_j), "quarantined": False,
                    "reason": rsn or "the judge didn't affirmatively clear all three checks "
                              "(spurious / trivial / coincidental) — unsure, so default-drop"}

        flags = []
        if bool(spurious): flags.append("spurious")
        if bool(coincidental_j): flags.append("coincidental")
        if bool(trivial_j): flags.append("trivial/already-obvious")
        if flags:
            return {"spurious": True, "is_trivial": bool(trivial_j),
                    "is_coincidental": bool(coincidental_j), "quarantined": False,
                    "reason": rsn or ("the judge judged the link " + ", ".join(flags))}

        return {"spurious": False, "is_trivial": False, "is_coincidental": False,
                "quarantined": False,
                "reason": rsn or "the judge tried to argue the link is random or trivial and "
                          "couldn't — it holds"}
    except Exception as exc:
        return {**_SPURIOUS_SHAPE,
                "reason": "refuter error (%s) — default-drop, bias to silence"
                          % type(exc).__name__}
