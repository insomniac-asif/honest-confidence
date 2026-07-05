"""Source fetchers — turn a social/video URL into plain text an LLM can reason over.

Two fetchers, one dispatcher, all fail-safe (they return a dict with an ``error``
key instead of raising, so a caller can always trust the return value):

    fetch_youtube(url) -> spoken transcript, from yt-dlp auto-captions (VTT -> text)
    fetch_tiktok(url)  -> ON-SCREEN text, by sampling frames and OCR-ing each
    fetch(url)         -> dispatch by host (youtube/youtu.be -> youtube; tiktok -> tiktok)

The technique is the same one a running local agent uses to harvest knowledge from
videos, generalized for public / reproducible use: no private imports, no hard-wired
paths, no secrets — just the external CLIs the user already has on PATH.

EXTERNAL CLIS (must be on PATH; each is guarded and optional):
    * ``yt-dlp``   — downloads captions / video. Required for both fetchers.
    * ``ffmpeg``   — samples frames from the TikTok video. Required for fetch_tiktok.
    * OCR backend  — EITHER the ``pytesseract`` Python package (preferred, offline,
                     no server) OR a local Ollama vision model (``qwen2.5vl:3b`` by
                     default) reached over ``http://localhost:11434``. If neither is
                     available, fetch_tiktok returns empty on-screen text (not an error).

WHY OCR FOR TIKTOK, CAPTIONS FOR YOUTUBE: YouTube reliably auto-generates caption
tracks yt-dlp can pull; TikTok usually has none, and its *information* often lives as
burned-in on-screen text (URLs, handles, quotes) — which OCR recovers and captions
never would.

Every subprocess call passes an explicit ``timeout`` so a wedged download cannot hang
the caller. Nothing here raises: on any failure the fetchers degrade to empty text.
"""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
from typing import Any, Dict, List, Set

# --- tunables (all env-overridable so the demo is reproducible on any box) ----------
YTDLP = shutil.which("yt-dlp") or "yt-dlp"
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
VISION_MODEL = os.environ.get("VISION_MODEL", "qwen2.5vl:3b")

_OCR_PROMPT = (
    "Transcribe EVERY piece of visible on-screen text in this image verbatim - "
    "especially URLs, @usernames, hashtags, and quotes. If there is no text, reply "
    "with an empty line."
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _run(cmd: List[str], timeout: float) -> subprocess.CompletedProcess:
    """Run a subprocess capturing text output; never raises (returns a dummy on error)."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception:
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="")


def _vtt_to_text(vtt_path: str) -> str:
    """Flatten a WebVTT caption file to a single de-duplicated line of spoken text.

    Drops cue-timing lines (``-->``), the ``WEBVTT`` / ``Kind:`` / ``Language:`` header,
    bare cue numbers, inline ``<...>`` timing tags, and repeated lines (auto-captions
    repeat heavily). Fail-safe: returns "" on any read error.
    """
    try:
        raw = open(vtt_path, encoding="utf-8", errors="replace").read()
    except Exception:
        return ""
    seen: Set[str] = set()
    lines: List[str] = []
    for ln in raw.splitlines():
        if ("-->" in ln or ln.startswith(("WEBVTT", "Kind:", "Language:"))
                or ln.strip().isdigit() or not ln.strip()):
            continue
        ln = re.sub(r"<[^>]+>", "", ln).strip()   # strip inline timing tags
        if ln and ln not in seen:
            seen.add(ln)
            lines.append(ln)
    return " ".join(lines)


def _ocr_frame(path: str, timeout: float) -> str:
    """OCR a single image file. Prefers offline pytesseract; falls back to Ollama vision.

    Returns the extracted text, or "" if no OCR backend is available / it fails. Never raises.
    """
    # Preferred: pytesseract (offline, no server, deterministic).
    try:
        import pytesseract          # type: ignore
        from PIL import Image       # type: ignore
        return (pytesseract.image_to_string(Image.open(path)) or "").strip()
    except Exception:
        pass
    # Fallback: a local Ollama vision model over base64 (best-effort, guarded).
    try:
        b64 = base64.b64encode(open(path, "rb").read()).decode()
        payload = {
            "model": VISION_MODEL,
            "prompt": _OCR_PROMPT,
            "images": [b64],
            "stream": False,
            "options": {"num_predict": 220, "temperature": 0.1},
        }
        req = urllib.request.Request(
            OLLAMA_URL + "/api/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return (json.loads(r.read()).get("response") or "").strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# public fetchers
# ---------------------------------------------------------------------------
def fetch_youtube(url: str, timeout: float = 120) -> Dict[str, Any]:
    """Fetch a YouTube video's spoken transcript from yt-dlp auto-captions.

    Returns ``{"source": "youtube", "title": str, "text": str, "segments": list}``:
      * ``title``    — the video title (best-effort; "" if unavailable).
      * ``text``     — the flattened caption transcript ("" if the video has none).
      * ``segments`` — the transcript split into sentence-ish chunks (a light convenience
                       for downstream chunking; empty when there is no text).

    Fail-safe: on any failure returns the same shape with ``"text": ""`` and an
    ``"error"`` key describing what went wrong. Never raises.
    """
    out: Dict[str, Any] = {"source": "youtube", "title": "", "text": "", "segments": []}
    if not shutil.which(YTDLP):
        out["error"] = "yt-dlp not found on PATH"
        return out
    try:
        meta = _run([YTDLP, "--skip-download", "--no-warnings",
                     "--print", "%(title)s", url], timeout=min(60, timeout))
        stdout = (meta.stdout or "").strip()
        out["title"] = stdout.splitlines()[0] if stdout else ""
    except Exception:
        pass
    try:
        with tempfile.TemporaryDirectory() as d:
            _run([YTDLP, "--skip-download", "--write-auto-subs", "--sub-lang", "en.*",
                  "--sub-format", "vtt", "--no-warnings",
                  "-o", os.path.join(d, "s.%(ext)s"), url], timeout=timeout)
            vtt = next((os.path.join(d, f) for f in os.listdir(d) if f.endswith(".vtt")), None)
            text = _vtt_to_text(vtt) if vtt else ""
        out["text"] = text
        if text:
            out["segments"] = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        else:
            out["error"] = "no captions available"
    except Exception as e:   # pragma: no cover - defensive
        out["error"] = "%s: %s" % (type(e).__name__, e)
    return out


def fetch_tiktok(url: str, max_frames: int = 8, timeout: float = 120) -> Dict[str, Any]:
    """Fetch a TikTok's ON-SCREEN text by sampling frames and OCR-ing each.

    Pipeline: yt-dlp downloads the video -> ffmpeg samples ~``max_frames`` frames
    (one every few seconds, downscaled) -> each frame is OCR'd (pytesseract, else a
    local Ollama vision model) -> the de-duplicated reads are concatenated.

    Returns ``{"source": "tiktok", "text": str, "frames_ocr": list}``:
      * ``text``       — all unique on-screen text joined by newlines ("" if none found).
      * ``frames_ocr`` — the per-frame OCR strings (kept for inspection / debugging).

    Fail-safe: missing yt-dlp / ffmpeg, a failed download, or no OCR backend all yield
    the same shape with ``"text": ""`` (and an ``"error"`` for hard failures). Never raises.
    """
    out: Dict[str, Any] = {"source": "tiktok", "text": "", "frames_ocr": []}
    if not shutil.which(YTDLP):
        out["error"] = "yt-dlp not found on PATH"
        return out
    if not shutil.which(FFMPEG):
        out["error"] = "ffmpeg not found on PATH"
        return out
    try:
        with tempfile.TemporaryDirectory() as d:
            _run([YTDLP, "-f", "b/best", "--no-warnings",
                  "-o", os.path.join(d, "v.%(ext)s"), url], timeout=timeout)
            vf = next((os.path.join(d, f) for f in os.listdir(d)
                       if f.lower().endswith((".mp4", ".webm", ".mkv", ".mov"))), None)
            if not vf:
                out["error"] = "video download failed"
                return out
            # one frame every 4s, capped at max_frames, downscaled to 640px wide
            _run([FFMPEG, "-i", vf, "-vf", "fps=1/4,scale=640:-1",
                  "-frames:v", str(max_frames), os.path.join(d, "f%02d.jpg"), "-y"],
                 timeout=timeout)
            frames = sorted(f for f in os.listdir(d)
                            if f.startswith("f") and f.endswith(".jpg"))[:max_frames]
            reads: List[str] = []
            for f in frames:
                t = _ocr_frame(os.path.join(d, f), timeout=min(120, timeout))
                if t:
                    reads.append(t)
        out["frames_ocr"] = reads
        # de-duplicate near-identical reads (on-screen text often persists across frames)
        seen: Set[str] = set()
        unique: List[str] = []
        for r in reads:
            key = re.sub(r"\s+", " ", r).strip()[:80].lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(r.strip())
        out["text"] = "\n".join(unique)
    except Exception as e:   # pragma: no cover - defensive
        out["error"] = "%s: %s" % (type(e).__name__, e)
    return out


def fetch(url: str) -> Dict[str, Any]:
    """Dispatch a URL to the right fetcher by host.

    ``youtube.com`` / ``youtu.be`` -> :func:`fetch_youtube`;
    ``tiktok.com``                 -> :func:`fetch_tiktok`.
    Any other host returns ``{"source": "unknown", "text": "", "error": ...}``. Never raises.
    """
    u = (url or "").lower()
    if "youtube.com" in u or "youtu.be" in u:
        return fetch_youtube(url)
    if "tiktok.com" in u:
        return fetch_tiktok(url)
    return {"source": "unknown", "text": "",
            "error": "unsupported url (expected youtube or tiktok): %r" % url}
