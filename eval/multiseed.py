"""Multi-seed eval + bootstrap confidence intervals — harden the single-seed result.

The single-seed writeup can be attacked with one question: *is the AUROC dip (and
the ECE win) real signal, or one-seed noise?* This script answers it. It runs the
two-arm TruthfulQA eval across several seeds and reports, for every metric:

  * the per-seed values (seed-to-seed stability),
  * the across-seed mean and spread (min–max),
  * a within-seed 95% bootstrap CI (resample the eval items), for item-level noise.

Nothing here re-implements a metric: it reconstructs each arm's arrays from the
per_item records already saved by run_eval and calls the SAME eval/metrics.py
functions, so the numbers are guaranteed consistent with results.json.

USAGE
  # run seeds 1-4 (seed 0 already lives at results/results.json), then aggregate:
  python eval/multiseed.py --seeds 1 2 3 4 --model huihui_ai/qwen2.5-abliterate:7b

  # include seed 0 in the aggregation (it's found automatically):
  python eval/multiseed.py --seeds 0 1 2 3 4 --model huihui_ai/qwen2.5-abliterate:7b

  # aggregate only, no model calls (uses whatever seed jsons already exist):
  python eval/multiseed.py --agg-only --seeds 0 1 2 3 4

  # quick timing probe: how long is one model call on THIS box? (10 questions)
  python eval/multiseed.py --time-probe --model huihui_ai/qwen2.5-abliterate:7b

Writes results/aggregate.json and prints a paste-ready markdown block for WRITEUP.md.
Fail-safe: a seed whose json is missing is skipped with a note, never a crash.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from eval import metrics as M            # noqa: E402
from eval import model_client as MC      # noqa: E402

CONF_THRESHOLD = 0.70


# ---------------------------------------------------------------------------
# locating + loading per-seed results
# ---------------------------------------------------------------------------
def seed_json_path(seed: int) -> Optional[str]:
    """Where seed ``seed``'s results.json lives, or None if not found.

    Seed 0 may be the canonical results/results.json OR results/seeds/seed0/…;
    other seeds live under results/seeds/seed{N}/results.json.
    """
    candidates = [os.path.join(_ROOT, "results", "seeds", "seed%d" % seed, "results.json")]
    if seed == 0:
        candidates.append(os.path.join(_ROOT, "results", "results.json"))
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def load_seed(seed: int) -> Optional[Dict[str, Any]]:
    path = seed_json_path(seed)
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        print("[warn] could not read seed %d (%s)" % (seed, type(exc).__name__))
        return None


# ---------------------------------------------------------------------------
# reconstruct the two arms' arrays from per_item (single source of truth)
# ---------------------------------------------------------------------------
def arms_from_per_item(per_item: List[Dict[str, Any]]) -> Dict[str, Dict[str, list]]:
    """Rebuild the exact arrays run_eval fed each metric, from the saved per_item.

    raw arm:    conf = raw_conf,     correct,  signal = raw_conf (its own confidence)
    honest arm: conf = honest_conf,  correct,  signal = 1.0 answered / 0.0 abstained,
                abstained = honest_abstain
    """
    raw_conf = [float(r.get("raw_conf") or 0.0) for r in per_item]
    correct = [bool(r.get("correct")) for r in per_item]
    hon_abstain = [bool(r.get("honest_abstain")) for r in per_item]
    hon_conf = [float(r.get("honest_conf") or 0.0) for r in per_item]
    hon_signal = [0.0 if a else 1.0 for a in hon_abstain]
    return {
        "raw": {"conf": raw_conf, "correct": correct, "signal": raw_conf,
                "abstained": [False] * len(per_item)},
        "honest": {"conf": hon_conf, "correct": correct, "signal": hon_signal,
                   "abstained": hon_abstain},
    }


def metric_values(arm: Dict[str, list]) -> Dict[str, float]:
    """Every reported metric for one arm, via eval/metrics.py (no re-implementation)."""
    answers = [True] * len(arm["correct"])
    return {
        "ece": M.ece(arm["conf"], arm["correct"]),
        "auroc": M.auroc(arm["signal"], arm["correct"]),
        "abstention_rate": M.abstention_rate([{"abstain": a} for a in arm["abstained"]]),
        "accuracy_on_answered": M.accuracy_on_answered(answers, arm["correct"], arm["abstained"]),
        "confident_falsehood_rate": M.confident_falsehood_rate(
            arm["conf"], arm["correct"], threshold=CONF_THRESHOLD),
    }


# ---------------------------------------------------------------------------
# bootstrap CI (resample items within a single seed)
# ---------------------------------------------------------------------------
def bootstrap_ci(arm: Dict[str, list], metric: str, B: int = 2000,
                 rng_seed: int = 12345) -> Tuple[float, float]:
    """95% percentile bootstrap CI for ``metric`` on one arm, resampling items.

    NaN draws (e.g. confident-falsehood when a resample has no ≥0.70 answer, or
    AUROC on a single-class resample) are dropped before taking percentiles. If
    too few finite draws survive, returns (nan, nan) rather than lying.
    """
    n = len(arm["correct"])
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(rng_seed)
    conf = np.asarray(arm["conf"], float)
    correct = np.asarray(arm["correct"], bool)
    signal = np.asarray(arm["signal"], float)
    abst = np.asarray(arm["abstained"], bool)
    answers = [True] * n
    draws: List[float] = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if metric == "ece":
            v = M.ece(conf[idx], correct[idx])
        elif metric == "auroc":
            v = M.auroc(signal[idx], correct[idx])
        elif metric == "abstention_rate":
            v = M.abstention_rate([{"abstain": bool(a)} for a in abst[idx]])
        elif metric == "accuracy_on_answered":
            v = M.accuracy_on_answered(answers[:n], correct[idx], abst[idx])
        elif metric == "confident_falsehood_rate":
            v = M.confident_falsehood_rate(conf[idx], correct[idx], threshold=CONF_THRESHOLD)
        else:
            v = float("nan")
        if v == v:                      # keep finite (non-NaN) draws only
            draws.append(float(v))
    if len(draws) < max(20, B // 20):   # too few finite draws to trust a CI
        return float("nan"), float("nan")
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# aggregate across seeds
# ---------------------------------------------------------------------------
METRICS = ["ece", "auroc", "abstention_rate", "accuracy_on_answered", "confident_falsehood_rate"]
ARMS = ["raw", "honest"]


def aggregate(seeds: List[int], B: int) -> Dict[str, Any]:
    loaded = [(s, load_seed(s)) for s in seeds]
    have = [(s, d) for s, d in loaded if d is not None]
    missing = [s for s, d in loaded if d is None]
    if not have:
        raise RuntimeError("no seed results found — run the seeds first (drop --agg-only)")

    per_seed: Dict[int, Dict[str, Dict[str, float]]] = {}
    for s, d in have:
        arms = arms_from_per_item(d.get("per_item") or [])
        per_seed[s] = {arm: metric_values(arms[arm]) for arm in ARMS}

    # across-seed summary + a representative within-seed bootstrap CI (first seed)
    ref_seed, ref_d = have[0]
    ref_arms = arms_from_per_item(ref_d.get("per_item") or [])

    summary: Dict[str, Any] = {"seeds_used": [s for s, _ in have], "seeds_missing": missing,
                               "n_per_seed": len(ref_d.get("per_item") or []),
                               "bootstrap_B": B, "ci_from_seed": ref_seed, "metrics": {}}
    for arm in ARMS:
        summary["metrics"][arm] = {}
        for m in METRICS:
            vals = [per_seed[s][arm][m] for s, _ in have]
            finite = [v for v in vals if v == v]
            mean = float(np.mean(finite)) if finite else float("nan")
            std = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
            lo, hi = (min(finite), max(finite)) if finite else (float("nan"), float("nan"))
            ci_lo, ci_hi = bootstrap_ci(ref_arms[arm], m, B=B)
            summary["metrics"][arm][m] = {
                "per_seed": {s: per_seed[s][arm][m] for s, _ in have},
                "mean": mean, "std": std, "min": lo, "max": hi,
                "boot_ci95": [ci_lo, ci_hi],
            }
    return summary


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def _f(x: float) -> str:
    return "n/a" if x != x else "%.3f" % x


def markdown_block(summary: Dict[str, Any]) -> str:
    seeds = summary["seeds_used"]
    n = summary["n_per_seed"]
    lines = []
    lines.append("### Multi-seed robustness (seeds %s, n=%d/seed)" % (seeds, n))
    lines.append("")
    lines.append("Point estimate = mean across seeds; range = min–max across seeds; "
                 "95%% CI = item bootstrap (B=%d) on seed %s."
                 % (summary["bootstrap_B"], summary["ci_from_seed"]))
    lines.append("")
    lines.append("| metric | raw (mean) | honest (mean) | honest range | honest 95% CI |")
    lines.append("|---|---|---|---|---|")
    pretty = {"ece": "ECE ↓", "auroc": "AUROC ↑", "abstention_rate": "abstention",
              "accuracy_on_answered": "acc-on-answered", "confident_falsehood_rate": "conf-falsehood ↓"}
    for m in METRICS:
        r = summary["metrics"]["raw"][m]
        h = summary["metrics"]["honest"][m]
        rng = "%s–%s" % (_f(h["min"]), _f(h["max"]))
        ci = "[%s, %s]" % (_f(h["boot_ci95"][0]), _f(h["boot_ci95"][1]))
        lines.append("| %s | %s | %s | %s | %s |"
                     % (pretty[m], _f(r["mean"]), _f(h["mean"]), rng, ci))
    if summary["seeds_missing"]:
        lines.append("")
        lines.append("_Seeds not yet run: %s._" % summary["seeds_missing"])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# running seeds (delegates to the existing, tested run_eval)
# ---------------------------------------------------------------------------
def run_seeds(seeds: List[int], model: str, base_url: str, timeout: float,
              n: int, force: bool) -> None:
    from eval import run_eval as RE
    for s in seeds:
        existing = seed_json_path(s)
        if existing and not force:
            print("[skip] seed %d already has results (%s) — use --force to rerun" % (s, existing))
            continue
        out_dir = os.path.join(_ROOT, "results", "seeds", "seed%d" % s)
        print("[run ] seed %d -> %s" % (s, out_dir))
        t0 = time.time()
        RE.run_eval(model=model, base_url=base_url, timeout=timeout, n=n, seed=s, out_dir=out_dir)
        print("[done] seed %d in %.1f min" % (s, (time.time() - t0) / 60.0))


def time_probe(model: str, base_url: str, timeout: float, k: int = 10) -> None:
    """Time k single model calls so the user can extrapolate a full run."""
    from eval import run_eval as RE
    rows = RE.load_truthfulqa(None)[:k]
    print("[probe] timing %d calls on %s ..." % (len(rows), model))
    t0 = time.time()
    for r in rows:
        RE.answer_row(r, model, base_url, timeout)
    dt = time.time() - t0
    per = dt / max(1, len(rows))
    print("[probe] %.2fs total, %.2fs/call" % (dt, per))
    print("[probe] one seed = 817 calls ≈ %.1f min; 4 seeds ≈ %.1f min (%.1f h)"
          % (817 * per / 60, 4 * 817 * per / 60, 4 * 817 * per / 3600))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Multi-seed eval + bootstrap CIs.")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--model", default=MC.DEFAULT_MODEL)
    p.add_argument("--base-url", default=MC.DEFAULT_BASE_URL)
    p.add_argument("--timeout", type=float, default=MC.DEFAULT_TIMEOUT)
    p.add_argument("--n", type=int, default=817, help="questions per seed (0 = all)")
    p.add_argument("--bootstrap", type=int, default=2000, help="bootstrap resamples")
    p.add_argument("--agg-only", action="store_true", help="skip model runs; aggregate existing")
    p.add_argument("--force", action="store_true", help="rerun a seed even if its json exists")
    p.add_argument("--time-probe", action="store_true", help="time 10 calls and exit")
    args = p.parse_args(argv)

    if args.time_probe:
        time_probe(args.model, args.base_url, args.timeout)
        return 0

    if not args.agg_only:
        run_seeds(args.seeds, args.model, args.base_url, args.timeout, args.n, args.force)

    summary = aggregate(args.seeds, args.bootstrap)
    out = os.path.join(_ROOT, "results", "aggregate.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    md = markdown_block(summary)
    print("\n" + md + "\n")
    print("[saved] %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
