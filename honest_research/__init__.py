"""honest_research — the runnable demo of the honest-confidence library.

A small, reproducible deep-research tool that turns a video / social URL into a
factual summary and a set of CLAIMS, each gated through the honesty layer so that
ungrounded claims ABSTAIN and surviving claims carry a CALIBRATED confidence.

Public surface::

    research(url, model=DEFAULT_NONTHINKING, max_claims=8) -> {title, summary,
             claims:[{claim, calibrated_conf, abstain, reason}], source}
    check(claim, answer=None, evidence=None, ...)          -> honest verdict dict

Source fetchers live in :mod:`honest_research.sources`; the local model wrapper in
:mod:`honest_research.model_client`. Everything is fail-safe and stdlib-first; the
only external needs are optional local CLIs (yt-dlp / ffmpeg / an OCR backend) and
an optional local OpenAI-compatible model endpoint.
"""
from __future__ import annotations

from .model_client import DEFAULT_MODEL
from .model_client import DEFAULT_MODEL as DEFAULT_NONTHINKING
from .pipeline import check, research

__all__ = ["research", "check", "DEFAULT_MODEL", "DEFAULT_NONTHINKING"]
