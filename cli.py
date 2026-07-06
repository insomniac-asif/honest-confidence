"""honest-research — runnable CLI demo of the honest-confidence library.

Two subcommands, both fail-safe (they print a legible verdict, never a traceback):

    check    gate one claim/answer for honesty + calibrated confidence.
             $ python cli.py check "the earth is 4.5 billion years old" \
                   --evidence "radiometric dating of meteorites" \
                   --evidence "oldest zircon crystals ~4.4 Gyr" --raw-conf 0.85

    research turn a video / social URL into a factual summary + a table of claims,
             each carrying a CALIBRATED confidence and marked [ABSTAINED] where the
             source does not actually ground it.
             $ python cli.py research "https://youtu.be/<id>" --max-claims 6

The commands are thin, human-readable front-ends over ``honest_research.check`` and
``honest_research.research`` (which themselves gate every claim through
``honest_confidence.decision.decide``). Model calls hit a local OpenAI-compatible
endpoint (Ollama by default); the DEFAULT model is a NON-thinking one because qwen3.x
*thinking* models return empty content over ``/v1`` and would blank every summary.

Design mirrors ``honest_confidence/calibration.py``: module docstring, type hints,
stdlib-only (argparse), and no path that raises into the user — a dead endpoint or a
bad URL degrades to a printed ``error`` line and a non-zero exit code.
"""
from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Optional, Sequence

from honest_research import DEFAULT_NONTHINKING, check, research


# ---------------------------------------------------------------------------
# formatting helpers (pure; never raise)
# ---------------------------------------------------------------------------
def _pct(x: Any) -> str:
    """Render a 0-1 confidence as a percent string; ""/None -> "n/a"."""
    try:
        return "%d%%" % round(float(x) * 100)
    except Exception:
        return "n/a"


def _wrap(text: str, width: int = 78, indent: str = "  ") -> str:
    """Soft word-wrap for readable multi-line output. Never raises."""
    try:
        words = (text or "").split()
        if not words:
            return ""
        lines: List[str] = []
        cur = indent
        for w in words:
            if len(cur) + len(w) + 1 > width and cur.strip():
                lines.append(cur.rstrip())
                cur = indent + w
            else:
                cur = (cur + " " + w) if cur.strip() else (indent + w)
        if cur.strip():
            lines.append(cur.rstrip())
        return "\n".join(lines)
    except Exception:
        return indent + (text or "")


def _rule(char: str = "-", width: int = 78) -> str:
    """A horizontal rule line."""
    return char * width


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------
def _render_check(claim: str, verdict: Dict[str, Any]) -> str:
    """Human-readable block for a single decide() verdict."""
    abstain = bool(verdict.get("abstain", True))
    status = "ABSTAIN" if abstain else "ANSWER"
    cal = _pct(verdict.get("calibrated_conf", 0.0))
    grounded = "no (abstained)" if abstain else "yes"
    out = [
        _rule("="),
        "CLAIM:   %s" % claim,
        _rule("="),
        "verdict:     %s" % status,
        "grounded?:   %s" % grounded,
        "confidence:  %s  (calibrated, never inflated)" % cal,
    ]
    support = verdict.get("grounding")          # what actually grounded the claim
    if support:
        out.append("support:     %s" % _wrap(str(support), indent="             ").strip())
    out += [
        "reason:",
        _wrap(str(verdict.get("reason", "")), indent="    "),
        _rule("="),
    ]
    return "\n".join(out)


def _cmd_check(args: argparse.Namespace) -> int:
    """Run the check subcommand. Returns a process exit code (0 answered, 2 abstained)."""
    evidence: List[str] = list(args.evidence or [])
    verdict = check(
        args.claim,
        answer=args.claim,
        evidence=evidence,
        raw_conf=args.raw_conf,
    )
    print(_render_check(args.claim, verdict))
    return 0 if not verdict.get("abstain", True) else 2


# ---------------------------------------------------------------------------
# research
# ---------------------------------------------------------------------------
def _render_research(res: Dict[str, Any]) -> str:
    """Human-readable summary + claims table for a research() result."""
    lines: List[str] = []
    lines.append(_rule("="))
    lines.append("SOURCE:  %s" % (res.get("source") or "unknown"))
    if res.get("title"):
        lines.append("TITLE:   %s" % res["title"])
    lines.append("URL:     %s" % (res.get("url") or ""))
    lines.append(_rule("="))

    summary = (res.get("summary") or "").strip()
    lines.append("SUMMARY:")
    lines.append(_wrap(summary) if summary else "  (no summary)")
    lines.append("")

    claims: List[Dict[str, Any]] = list(res.get("claims") or [])
    answered = sum(1 for c in claims if not c.get("abstain", True))
    lines.append("CLAIMS:  %d total  |  %d answered  |  %d abstained"
                 % (len(claims), answered, len(claims) - answered))
    lines.append(_rule("-"))
    if not claims:
        lines.append("  (no claims extracted)")
    for i, c in enumerate(claims, 1):
        abstain = bool(c.get("abstain", True))
        tag = "[ABSTAINED]" if abstain else "[%s]" % _pct(c.get("calibrated_conf", 0.0))
        lines.append("%2d. %-12s %s" % (i, tag, c.get("claim", "")))
        reason = (c.get("reason") or "").strip()
        if reason:
            lines.append(_wrap(reason, indent="      "))
    lines.append(_rule("="))

    if res.get("error"):
        lines.append("note: %s" % res["error"])
    return "\n".join(lines)


def _cmd_research(args: argparse.Namespace) -> int:
    """Run the research subcommand. Returns 0 on any well-formed result, 1 on hard error."""
    res = research(
        args.url,
        model=args.model,
        max_claims=args.max_claims,
    )
    print(_render_research(res))
    # a result with no summary AND no claims but carrying an error is a hard degrade
    if res.get("error") and not res.get("summary") and not res.get("claims"):
        return 1
    return 0


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser for the honest-research CLI."""
    p = argparse.ArgumentParser(
        prog="honest-research",
        description="Runnable demo of the honest-confidence library: gate claims so "
                    "ungrounded ones abstain and the rest carry a calibrated confidence.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("check", help="gate one claim/answer for honesty + confidence")
    c.add_argument("claim", help="the claim / answer to check")
    c.add_argument("--evidence", action="append", default=[], metavar="TEXT",
                   help="a supporting evidence string (repeatable; >=2 distinct needed "
                        "to clear the grounding gate)")
    c.add_argument("--raw-conf", type=float, default=0.7, metavar="0-1",
                   help="the raw self-reported confidence to calibrate (default 0.7)")
    c.set_defaults(func=_cmd_check)

    r = sub.add_parser("research", help="URL -> factual summary + honesty-gated claims")
    r.add_argument("url", help="a YouTube or TikTok URL to research")
    r.add_argument("--model", default=DEFAULT_NONTHINKING, metavar="NAME",
                   help="local model name (default: %(default)s — a NON-thinking model; "
                        "thinking models blank out over /v1)")
    r.add_argument("--max-claims", type=int, default=8, metavar="N",
                   help="max discrete claims to extract (default 8)")
    r.set_defaults(func=_cmd_research)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Parses args and dispatches; returns a process exit code.

    Fail-safe: any unexpected error is caught and printed as a legible line rather than
    a traceback, and yields exit code 1.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\ninterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # pragma: no cover - defensive
        print("error: %s (%s)" % (exc, type(exc).__name__), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
