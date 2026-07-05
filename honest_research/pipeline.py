"""Deep-research pipeline — the runnable HONESTY DEMO.

This is the end-to-end tool that ties the whole package together: it takes a
video / social URL, turns it into text (:mod:`honest_research.sources`), asks a
local model to summarize and extract factual CLAIMS, then — for every claim — runs
it through the honesty gate (:func:`honest_confidence.decision.decide`) so that
ungrounded claims ABSTAIN and every surviving claim carries a CALIBRATED confidence
instead of the model's raw self-report.

    research(url, ...)  -> {title, summary, claims:[{claim, calibrated_conf,
                            abstain, reason}], source}
    check(claim, ...)   -> thin wrapper over decide() for the "check my work for
                           honesty / confidence" mode.

WHY THIS IS THE DEMO: a normal "summarize this video" tool asserts every claim it
extracts at whatever confidence the model feels. This one refuses to. Each claim's
supporting EVIDENCE is built by finding which chunks of the ACTUAL source text
mention the claim's key terms; a claim that isn't echoed in at least two distinct
source chunks is ungrounded and abstains. Confidence on the rest is deflated toward
a measured accuracy rate. The output makes the honest move legible, per claim.

The techniques here (map-reduce summary over a local /v1 model, key-term evidence
grounding, decision-gated confidence) are reused, not privately imported: no server
internals, no hard-wired paths, no secrets. The one external dependency is a local
OpenAI-compatible model endpoint (Ollama by default), and it is fully optional —
every model call is guarded and the pipeline degrades to a legible empty result
rather than raising.

GOTCHA (inherited from model_client): the DEFAULT model is a NON-thinking model
(``huihui_ai/qwen2.5-abliterate:7b``). qwen3.x *thinking* models return empty
content over the OpenAI ``/v1`` route, which would silently blank every summary and
claim. Point this at a thinking model only if you disable its thinking mode.

Fail-safe throughout: any failure (dead endpoint, empty reply, bad URL) yields a
well-formed result dict with empty ``claims`` and a legible ``error``/reason — it
never raises into the caller.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

from honest_confidence import decision
from . import sources
from .model_client import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    parse_confidence,
)

# -- honesty tunables ---------------------------------------------------------------
# measured_rate / graded are the accuracy the model has been MEASURED at on a held-out
# split (see honest_confidence.calibration.fit_measured_rate). These are demo defaults;
# a real deployment fits them on its own eval. 0.62 over 490 graded items mirrors the
# package's documented example so calibrated_conf is never oversold.
DEFAULT_MEASURED_RATE = 0.62
DEFAULT_GRADED = 490

_CHUNK_WORDS = 800          # ~800-word windows for map-reduce summary + grounding
_MIN_TERM_LEN = 4           # ignore short/stopword-ish key terms when matching evidence
_STOPWORDS = frozenset(
    "the and for that this with from have will your they them then than into"
    " over more most some such been being about which while where when what"
    " youre their there these those here onto also just like only".split()
)


# ---------------------------------------------------------------------------
# local model call (guarded; degrades to "" on any failure)
# ---------------------------------------------------------------------------
def _chat(
    prompt: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = DEFAULT_API_KEY,
    timeout: float = DEFAULT_TIMEOUT,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    """One-shot chat completion over the local OpenAI-compatible endpoint.

    Same wire pattern as :func:`model_client.answer_with_confidence`, but for free-form
    prompts (summary / claim-extraction / confidence). Always passes an explicit request
    timeout so a wedged local model cannot hang the pipeline.

    Fail-safe: on ANY error (missing SDK, dead endpoint, blank reply — including the
    thinking-model empty-content gotcha) it returns ``""`` and never raises.
    """
    try:
        from openai import OpenAI  # lazy so the module imports without the SDK
    except Exception:
        return ""
    try:
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=float(timeout))
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            timeout=float(timeout),
        )
        try:
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            return ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# text helpers (pure, deterministic, never raise)
# ---------------------------------------------------------------------------
def _chunk_words(text: str, size: int = _CHUNK_WORDS) -> List[str]:
    """Split text into ~``size``-word windows. Returns [] for empty text."""
    try:
        words = (text or "").split()
        if not words:
            return []
        return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]
    except Exception:
        return []


def _key_terms(claim: str) -> List[str]:
    """Extract distinctive lower-cased key terms from a claim for evidence matching.

    Keeps alphanumeric tokens >= _MIN_TERM_LEN that are not obvious stopwords, plus any
    4+ digit number (years, counts). De-duplicated, order-preserving. [] on error.
    """
    try:
        toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]+", (claim or "").lower())
        out: List[str] = []
        seen = set()
        for t in toks:
            keep = (t.isdigit() and len(t) >= 4) or (
                len(t) >= _MIN_TERM_LEN and t not in _STOPWORDS)
            if keep and t not in seen:
                seen.add(t)
                out.append(t)
        return out
    except Exception:
        return []


def _evidence_for(claim: str, chunks: List[str]) -> List[str]:
    """Return the source chunks that support ``claim`` as distinct grounding endpoints.

    A chunk SUPPORTS the claim if it contains at least two of the claim's distinct key
    terms (one shared word is coincidence; two is corroboration). Supporting chunks become
    the endpoints decide() counts — >= min_endpoints DISTINCT ones are required to clear the
    grounding gate, so a claim the source never actually echoes will abstain. Each endpoint
    is prefixed ``src#N:`` so grounding's distinct-key de-dup counts two matching chunks as
    two supports. Fail-safe: [] on any error.
    """
    try:
        terms = _key_terms(claim)
        if not terms:
            return []
        support: List[str] = []
        for ch in chunks:
            low = ch.lower()
            hits = sum(1 for t in terms if t in low)
            if hits >= 2:
                support.append("src#%d: %s" % (len(support), ch[:200]))
        return support
    except Exception:
        return []


# ---------------------------------------------------------------------------
# model steps (all guarded; degrade to "" / [] on any failure)
# ---------------------------------------------------------------------------
def _summarize(text: str, model: str, base_url: str, timeout: float) -> str:
    """Map-reduce summary of ``text`` via the local model. "" if unavailable.

    MAP: summarize each ~800-word chunk to a few factual sentences.
    REDUCE: fuse the chunk summaries into one tight factual summary. A single chunk skips
    the reduce call and returns the map result directly.
    """
    chunks = _chunk_words(text)
    if not chunks:
        return ""
    partials: List[str] = []
    for ch in chunks:
        s = _chat(
            "Summarize the factual content below in 2-4 plain sentences. State only what "
            "the text actually claims; do not add outside information.\n\n" + ch,
            model=model, base_url=base_url, timeout=timeout,
        )
        if s:
            partials.append(s.strip())
    if not partials:
        return ""
    if len(partials) == 1:
        return partials[0]
    joined = "\n".join("- " + p for p in partials)
    reduced = _chat(
        "Fuse these partial summaries into one tight factual summary (4-6 sentences). "
        "Keep only claims present in the partials; drop repetition.\n\n" + joined,
        model=model, base_url=base_url, timeout=timeout,
    )
    return (reduced or "\n".join(partials)).strip()


def _extract_claims(summary: str, max_claims: int, model: str,
                    base_url: str, timeout: float) -> List[str]:
    """Ask the model for up to ``max_claims`` discrete factual claims from the summary.

    Returns a list of one-line claim strings (bullets/numbering parsed out). [] if the
    model is unavailable or the summary is empty. Never raises.
    """
    if not (summary or "").strip():
        return []
    raw = _chat(
        "From the summary below, list up to %d DISCRETE factual claims — each a single "
        "verifiable statement on its own line, prefixed with '- '. No commentary, no "
        "numbering, just the bulleted claims.\n\nSUMMARY:\n%s" % (max_claims, summary),
        model=model, base_url=base_url, timeout=timeout,
    )
    if not raw:
        return []
    claims: List[str] = []
    seen = set()
    for line in raw.splitlines():
        m = re.match(r"^\s*(?:[-*•]|\d+[.)])\s*(.+)$", line)
        cand = (m.group(1) if m else line).strip()
        if len(cand) >= 8 and cand.lower() not in seen:
            seen.add(cand.lower())
            claims.append(cand)
        if len(claims) >= max_claims:
            break
    return claims


def _raw_conf_for(claim: str, model: str, base_url: str, timeout: float) -> float:
    """Get the model's SELF-REPORTED confidence (0-1) that ``claim`` is true.

    This is the raw, uncalibrated number the honesty layer later deflates toward the
    measured accuracy rate. Falls back to 0.5 when the model is silent/unparseable.
    """
    txt = _chat(
        "How confident are you that the following statement is factually true? Reply with "
        "a single number from 0 to 1 on a line 'CONFIDENCE: <n>'.\n\n" + (claim or ""),
        model=model, base_url=base_url, timeout=timeout,
    )
    return parse_confidence(txt or "")


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def research(
    url: str,
    model: str = DEFAULT_MODEL,
    max_claims: int = 8,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = DEFAULT_TIMEOUT,
    measured_rate: float = DEFAULT_MEASURED_RATE,
    graded: Optional[int] = DEFAULT_GRADED,
) -> Dict[str, Any]:
    """Research a URL end-to-end and return honesty-gated claims.

    Returns::

        {
          "source":  "youtube" | "tiktok" | "unknown",
          "title":   str,          # best-effort source title ("" if none)
          "summary": str,          # map-reduce factual summary ("" if no text/model)
          "claims":  [ {claim, calibrated_conf, abstain, reason}, ... ],
          "url":     str,
          "error":   str,          # present only when something degraded the run
        }

    Flow: :func:`honest_research.sources.fetch` -> chunk -> map-reduce summary -> extract
    up to ``max_claims`` claims -> for EACH claim build evidence from the source chunks that
    echo its key terms, get the model's raw confidence, and call
    :func:`honest_confidence.decision.decide` so ungrounded claims abstain and the rest
    carry a calibrated confidence.

    Fail-safe: any failure yields a well-formed dict with ``claims: []`` and an ``error``;
    it never raises.
    """
    result: Dict[str, Any] = {
        "source": "unknown", "title": "", "summary": "", "claims": [], "url": url,
    }
    try:
        fetched = sources.fetch(url)
        result["source"] = fetched.get("source", "unknown")
        result["title"] = fetched.get("title", "") or ""
        text = (fetched.get("text") or "").strip()
        if fetched.get("error"):
            result["error"] = str(fetched["error"])
        if not text:
            result.setdefault("error", "no source text extracted")
            return result

        chunks = _chunk_words(text)
        summary = _summarize(text, model, base_url, timeout)
        result["summary"] = summary
        if not summary:
            result.setdefault("error", "model unavailable or produced no summary")
            return result

        claims = _extract_claims(summary, max_claims, model, base_url, timeout)
        graded_claims: List[Dict[str, Any]] = []
        for claim in claims:
            evidence = _evidence_for(claim, chunks)
            raw_conf = _raw_conf_for(claim, model, base_url, timeout)
            verdict = decision.decide(
                claim, raw_conf, evidence,
                measured_rate=measured_rate, graded=graded, answer=claim,
            )
            graded_claims.append({
                "claim": claim,
                "calibrated_conf": verdict.get("calibrated_conf", 0.0),
                "abstain": verdict.get("abstain", True),
                "reason": verdict.get("reason", ""),
            })
        result["claims"] = graded_claims
        return result
    except Exception as exc:   # pragma: no cover - defensive
        result.setdefault("error",
                          "research-error (%s) — returning partial result"
                          % type(exc).__name__)
        return result


def check(
    claim: str,
    answer: Optional[object] = None,
    evidence: Optional[Iterable] = None,
    raw_conf: float = 0.7,
    measured_rate: float = DEFAULT_MEASURED_RATE,
    graded: Optional[int] = DEFAULT_GRADED,
) -> Dict[str, Any]:
    """Check-my-work mode: gate one claim/answer for honesty + confidence.

    A thin wrapper over :func:`honest_confidence.decision.decide` for callers that already
    have a claim (and optionally an answer + cited evidence) and just want the honest
    verdict: is it grounded enough to assert, and at what CALIBRATED confidence?

    Returns decide()'s dict: ``{answer, abstain, calibrated_conf, reason}``. If ``answer``
    is omitted the claim text is used as the answer. Evidence defaults to an empty list
    (which, with the default ``min_endpoints=2``, ABSTAINS as ungrounded — the safe default
    when no supports are supplied). Fail-safe via decide(); never raises.
    """
    return decision.decide(
        claim, raw_conf, list(evidence or []),
        measured_rate=measured_rate, graded=graded,
        answer=claim if answer is None else answer,
    )
