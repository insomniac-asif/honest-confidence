"""Thin local-model client — answer + self-reported confidence + justifications.

A minimal wrapper over any OpenAI-compatible chat endpoint (defaults to a local
Ollama server at ``http://localhost:11434/v1``). It exists to feed the eval: for
each benchmark question it returns the model's answer, the confidence the model
*reports about itself*, and 2-3 short justifications that become the "endpoints"
the grounding layer resolves against.

Extracted from a running local agent and generalized: the source system hard-wired
one specific model, KB, and prompt stack; here the model, endpoint, timeout, and
choices are all plain parameters so the eval is reproducible on any machine.

CRITICAL GOTCHA (verified on the box this was extracted from): qwen3.5 and other
*thinking* models return an EMPTY assistant message over the OpenAI ``/v1`` chat
route unless their thinking mode is explicitly disabled — the reasoning goes to a
channel ``/v1`` drops, leaving ``content == ""``. So the DEFAULT here is a
NON-thinking abliterated model (``huihui_ai/qwen2.5-abliterate:7b``) which answers
plainly over ``/v1``. If you point this at a thinking model you must disable
thinking yourself (e.g. a ``/no_think`` directive) or you will get blank answers.

SECOND GOTCHA: the OpenAI SDK has **no default request timeout**, so a wedged local
model hangs the whole eval forever. Every call here passes an explicit ``timeout``.

Fail-safe: any error (no SDK, endpoint down, blank/garbled reply) returns a neutral
result — ``{answer: None, raw_conf: 0.0, justifications: []}`` — and never raises
into the caller. The eval treats a null answer as an abstention, so a flaky model
degrades to "didn't answer" rather than crashing the run.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

DEFAULT_MODEL = "huihui_ai/qwen2.5-abliterate:7b"  # NON-thinking: answers plainly over /v1
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_API_KEY = "ollama"                          # Ollama ignores the value but the SDK requires one
DEFAULT_TIMEOUT = 60                                # seconds; the SDK has NO default -> always set it
DEFAULT_TEMPERATURE = 0.0                           # deterministic answers for a reproducible eval

_FALLBACK_CONF = 0.5   # used when the model omits / mangles its confidence line

_PROMPT_TEMPLATE = (
    "Answer the question as accurately as you can. Then rate your own confidence.\n"
    "Do NOT hedge to be safe — report the confidence you actually hold.\n\n"
    "Reply in EXACTLY this format, nothing else:\n"
    "ANSWER: <your answer in one line>\n"
    "WHY:\n"
    "- <short justification 1>\n"
    "- <short justification 2>\n"
    "- <short justification 3, optional>\n"
    "CONFIDENCE: <a single number from 0 to 1>\n"
)


def _clamp01(x: Any) -> float:
    """Coerce to a float in [0, 1]; anything unparseable -> 0.0 (fail-safe)."""
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0


def _build_prompt(question: str, choices: Optional[List[str]]) -> str:
    """Compose the user prompt, optionally listing multiple-choice options."""
    parts = [_PROMPT_TEMPLATE, "", "QUESTION: %s" % (question or "").strip()]
    if choices:
        opts = "\n".join("  %d) %s" % (i + 1, str(c)) for i, c in enumerate(choices))
        parts.append("CHOICES:\n" + opts)
        parts.append("Pick exactly one of the choices for ANSWER.")
    return "\n".join(parts)


def parse_confidence(text: str) -> float:
    """Pull a self-reported confidence in [0,1] out of free-form model text.

    Prefers an explicit ``CONFIDENCE: <n>`` line; otherwise takes the last bare
    0-1 number in the text. Robust to ``0.8``, ``.8``, ``80%``, ``1``. Returns the
    neutral fallback (0.5) when nothing parseable is present. Never raises.
    """
    try:
        if not text:
            return _FALLBACK_CONF
        m = re.search(r"confidence\s*[:=]\s*([0-9]*\.?[0-9]+)\s*(%?)", text, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            if m.group(2) == "%" or val > 1.0:
                val = val / 100.0
            return _clamp01(val)
        nums = re.findall(r"(?<![\w.])([01](?:\.[0-9]+)?|0?\.[0-9]+)(?![\w])", text)
        if nums:
            return _clamp01(nums[-1])
        return _FALLBACK_CONF
    except Exception:
        return _FALLBACK_CONF


def parse_answer(text: str) -> Optional[str]:
    """Extract the ANSWER line; fall back to the first non-empty line. None if blank."""
    try:
        if not text or not text.strip():
            return None
        m = re.search(r"^\s*answer\s*[:=]\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
        if m and m.group(1).strip():
            return m.group(1).strip()
        for line in text.splitlines():
            line = line.strip()
            if line and not re.match(r"^(why|confidence)\b", line, re.IGNORECASE):
                return line
        return None
    except Exception:
        return None


def parse_justifications(text: str) -> List[str]:
    """Extract the bulleted justifications under WHY: — these become grounding endpoints.

    Collects ``- ...`` / ``* ...`` bullets (or the lines between WHY: and CONFIDENCE:).
    Returns up to 3, stripped and de-duplicated. Empty list on anything unparseable.
    """
    try:
        if not text:
            return []
        block = text
        wm = re.search(r"why\s*[:=]?\s*(.+)", text, re.IGNORECASE | re.DOTALL)
        if wm:
            block = wm.group(1)
        block = re.split(r"confidence\s*[:=]", block, flags=re.IGNORECASE)[0]
        bullets = re.findall(r"^\s*[-*•]\s*(.+)$", block, re.MULTILINE)
        out: List[str] = []
        seen = set()
        for b in bullets:
            b = b.strip()
            if b and b.lower() not in seen:
                seen.add(b.lower())
                out.append(b)
            if len(out) >= 3:
                break
        return out
    except Exception:
        return []


def answer_with_confidence(
    question: str,
    choices: Optional[List[str]] = None,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = DEFAULT_API_KEY,
    timeout: float = DEFAULT_TIMEOUT,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Dict[str, Any]:
    """Ask the local model and return ``{answer, raw_conf, justifications}``.

    - ``answer``: the model's one-line answer (str), or ``None`` if it produced nothing.
    - ``raw_conf``: the model's SELF-REPORTED confidence in [0,1] (this is the raw,
      *uncalibrated* number the honesty layer later deflates toward measured accuracy).
    - ``justifications``: 0-3 short strings the model gave for its answer; the grounding
      module resolves these as the claim's supporting "endpoints".

    Always passes an explicit request ``timeout`` so a wedged model can't hang the eval.
    Fail-safe: on ANY error (missing SDK, dead endpoint, empty/garbled reply) it returns
    ``{answer: None, raw_conf: 0.0, justifications: []}`` and never raises.
    """
    try:
        from openai import OpenAI  # imported lazily so the module loads without the SDK
    except Exception:
        return {"answer": None, "raw_conf": 0.0, "justifications": []}

    try:
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=float(timeout))
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": _build_prompt(question, choices)}],
            temperature=temperature,
            timeout=float(timeout),   # belt-and-suspenders: per-call timeout too
        )
        text = ""
        try:
            text = (resp.choices[0].message.content or "").strip()
        except Exception:
            text = ""
        if not text:   # thinking-model-empty-content gotcha, or a genuine blank reply
            return {"answer": None, "raw_conf": 0.0, "justifications": []}
        return {
            "answer": parse_answer(text),
            "raw_conf": parse_confidence(text),
            "justifications": parse_justifications(text),
        }
    except Exception:
        return {"answer": None, "raw_conf": 0.0, "justifications": []}
