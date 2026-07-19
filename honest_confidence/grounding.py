"""Grounding — abstain unless a claim resolves to enough REAL, DISTINCT supporting endpoints.

Extracted from a running local AI agent (an internal reasoning module) and generalized for public,
reproducible use. The portable core is a one-line decision rule:

    a claim is grounded IFF it cites >= min_endpoints references that each resolve to a real
    thing, AND those resolved supports are DISTINCT (not one source cited twice).

In the source agent the resolvers were tangled up with infrastructure — a SQLite knowledge-base
lookup, a filesystem/grep code probe, and a type-strict dispatcher. Those are all
implementation-specific ways of asking one generic question: **does this reference point at
something real?**

THE HONEST GENERALIZATION: that question is hoisted into a single injectable ``resolver``
callable ``(ref) -> bool``. The default resolver is deliberately dumb and transparent —
"the reference appears in the caller-supplied ``evidence`` list" — so the rule is fully
reproducible with no hidden database. A caller with a real KB, a retrieval index, or a code
repo passes their own resolver; the DECISION RULE (>= N distinct real supports, else drop)
is unchanged and is the thing the eval measures.

Fail CLOSED: any error, any empty/None input, anything ambiguous resolves to NOT grounded.
Abstaining is the safe direction for an honesty layer, so on doubt we never claim grounded.
"""
from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Tuple

DEFAULT_MIN_ENDPOINTS = 2       # a claim needs at least this many real, distinct supports


def _as_list(evidence) -> List:
    """Coerce arbitrary evidence input into a plain list, never raising."""
    if evidence is None:
        return []
    if isinstance(evidence, (str, bytes)):
        return [evidence]
    try:
        return list(evidence)
    except Exception:
        return [evidence]


def _default_resolver(evidence: Iterable) -> Callable[[object], bool]:
    """Build the stdlib default resolver: a ref is 'real' iff it is present in ``evidence``.

    This is the transparent, database-free stand-in for the source agent's KB/code probes.
    Membership is checked both directly and by string form, so callers can pass either the
    raw supports or their string ids as evidence.
    """
    items = _as_list(evidence)
    str_items = {str(e) for e in items}

    def _resolve(ref) -> bool:
        try:
            if ref in items:
                return True
            return str(ref) in str_items
        except Exception:
            return False

    return _resolve


def _distinct_key(ref) -> str:
    """Identity used to decide whether two resolved supports are actually the same thing."""
    try:
        return str(ref).strip().lower()
    except Exception:
        return repr(ref)


def is_grounded(
    claim: str,
    evidence: Iterable,
    resolver: Optional[Callable[[object], bool]] = None,
    min_endpoints: int = DEFAULT_MIN_ENDPOINTS,
) -> Tuple[bool, str]:
    """Return ``(grounded, reason)`` for ``claim`` against its cited ``evidence``.

    ``evidence`` is the list of references the claim cites as support. Each is checked with
    ``resolver`` — a callable ``ref -> bool`` that answers "does this reference point at
    something real?". If ``resolver`` is None, the default is used: a ref is real iff it
    appears in ``evidence`` (transparent, stdlib-only, no hidden store).

    RULE (as implemented in the source agent): the claim is grounded only if at
    least ``min_endpoints`` references resolve as real AND they are DISTINCT (the same source
    cited twice counts once). Otherwise it is NOT grounded and the caller should ABSTAIN.

    ``reason`` is always a legible plain-English string naming what carried or sank the claim.

    Fail-safe: on any internal error returns ``(False, "grounding-error")`` — it never raises
    and it fails CLOSED (abstain on doubt), so a caller can always trust a True result.
    """
    try:
        refs = _as_list(evidence)
        resolve = resolver if resolver is not None else _default_resolver(refs)

        real: List = []
        seen: set = set()
        for ref in refs:
            if ref is None or (isinstance(ref, str) and not ref.strip()):
                continue
            try:
                ok = bool(resolve(ref))
            except Exception:
                ok = False          # a resolver that blows up on a ref => that ref is not real
            if not ok:
                continue
            key = _distinct_key(ref)
            if key in seen:
                continue            # distinct-endpoint guard: don't count one source twice
            seen.add(key)
            real.append(ref)

        need = max(1, int(min_endpoints))
        if len(real) >= need:
            named = ", ".join(str(r) for r in real[:need])
            return True, ground_reason(True, len(real), need, named)
        return False, ground_reason(False, len(real), need, "")
    except Exception:
        return False, "grounding-error"


def ground_reason(grounded: bool, n_real: int, need: int, named: str) -> str:
    """Compose the legible reason string for a grounding verdict (kept separate so
    ``decision.py`` can reuse the same phrasing). Never raises."""
    try:
        if grounded:
            tail = (": " + named) if named else ""
            return ("grounded — %d distinct real endpoint%s resolved (needed %d)%s"
                    % (n_real, "" if n_real == 1 else "s", need, tail))
        return ("not grounded — only %d distinct real endpoint%s resolved, need %d "
                "(abstaining)" % (n_real, "" if n_real == 1 else "s", need))
    except Exception:
        return "grounding-error"
