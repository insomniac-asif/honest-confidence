# honest-confidence

A small, deterministic honesty layer for LLM agents — cap stated confidence at the model's *measured* accuracy and abstain on ungrounded claims — plus a reproducible TruthfulQA eval that reports where it helps **and** where it hurts.

## What it does

`honest_confidence/` is a dependency-free layer that takes a raw self-reported confidence and cited evidence and returns a calibrated verdict. It deflates confidence toward the model's held-out accuracy (never inflating it), abstains when a claim doesn't resolve to enough distinct supporting endpoints, and drops spurious claims through a zero-model refutation gate. `eval/` grades the layer against TruthfulQA instead of a hand-picked demo, and `honest_research/` is a small runnable pipeline that turns a video/social URL into a claims table with each claim gated through the layer.

## Why

Agents routinely state high confidence (`"i'm 95% sure"`) on a class of task that is right far less often, and assert claims they can't ground. This repo extracts two mechanisms from a running local agent, generalizes them, and — the point of the project — swaps the agent's *subjective* approve-rate for a real held-out accuracy so the effect can be measured against a labeled benchmark, reporting both the calibration win and its cost (correct answers lost to over-abstention). A method that only reports its wins isn't a measurement.

## Install

No packaging (`pyproject.toml`/`setup.py`) — run the scripts in place. The honesty layer itself (`honest_confidence/`) imports no third-party packages; the requirements exist only for the eval harness and the local-model demo.

```bash
git clone https://github.com/insomniac-asif/honest-confidence
cd honest-confidence
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

Dependencies (`requirements.txt`): `numpy`, `scikit-learn` (AUROC + calibration helpers), `matplotlib` (reliability diagrams), `datasets` (TruthfulQA loader), `openai` (client for a local OpenAI-compatible endpoint, e.g. Ollama's `/v1`). Requires Python 3.9+ (per the repo); the dependency floors want a reasonably modern Python 3.

## Quickstart

Gate a single claim from the CLI. `check` runs the pure decision path — no model, no network:

```bash
python cli.py check "the earth is about 4.5 billion years old" \
    --evidence "radiometric dating of meteorites yields 4.5 Gyr" \
    --evidence "oldest zircon crystals date to about 4.4 billion years" \
    --raw-conf 0.9
```

It prints a verdict (`ANSWER` / `ABSTAIN`), whether the claim is grounded, and a calibrated confidence with the reasoning behind it. A claim with fewer than two distinct supporting endpoints abstains instead of asserting. Exit code is `0` on answer, `2` on abstain (handy for scripting). The same command is available as `python -m honest_research check ...`.

From the library:

```python
from honest_confidence import calibrate_confidence
from honest_confidence.decision import decide

calibrate_confidence(0.95, measured_rate=0.37)   # deflated toward 0.37, never above raw
decide("the earth is ~4.5 Gyr old", raw_conf=0.9,
       evidence=["meteorite dating", "zircon dating"], measured_rate=0.37)
# -> {'answer': ..., 'abstain': False, 'calibrated_conf': ..., 'reason': '...'}
```

Run the eval (raw model vs. model + honesty layer on TruthfulQA MC1). Defaults to a local OpenAI-compatible endpoint; `--model`, `--base-url`, `--n`, and `--seed` are configurable:

```bash
python eval/run_eval.py --n 50 --seed 0          # fast smoke run
python -m eval.multiseed                         # multi-seed -> results/aggregate.json
```

> The exact CLI invocations above are taken from the scripts' own argparse/docstrings; endpoint and model defaults depend on your local setup, so adjust `--model`/`--base-url` accordingly.

## How it works

The layer is a short pipeline where each stage can only **abstain**, never upgrade a later stage's confidence:

```
raw self-reported conf  +  cited evidence
        |
        v
  grounding.is_grounded   -- <2 distinct endpoints? --------> ABSTAIN (conf 0.0)
        |
        v
  refuter.refute          -- spurious / trivial? -----------> ABSTAIN (conf 0.0)
        |
        v
  calibration.calibrate_confidence(raw, measured_rate)  -> deflate toward measured accuracy (never inflate)
        |
        v
  decide()  ->  {answer, calibrated_conf, reason}
```

`calibrate_confidence` deflates toward the measured rate with a margin (so a 37%-accurate model caps near ~0.52, not all the way to 0.37) and is clamped so the output can never exceed the raw confidence. `fit_measured_rate()` fits that rate on a held-out split. The eval wraps this in a measurement: hold out ~40% of TruthfulQA to fit the model's measured accuracy, evaluate on the rest, and compare ECE, AUROC, abstention rate, accuracy-on-answered, and confident-falsehood rate for the raw vs. honest arms.

## Results

From `results/aggregate.json` — mean over 4 seeds, local model `huihui_ai/qwen2.5-abliterate:7b`, ~327 held-out / 490 evaluated per seed (measured accuracy ~37%). `raw` = the model's self-reported confidence; `honest` = the same answers through the layer.

| metric | raw | honest | |
|---|---|---|---|
| ECE (calibration error, ↓) | 0.479 | 0.139 | ~3.4× lower |
| AUROC (↑) | 0.577 | 0.509 | dipped toward chance |
| abstention rate | 0.000 | 0.048 | |
| accuracy on answered | 0.413 | 0.418 | ~unchanged |
| confident-falsehood rate (↓) | 0.583 | NaN (undefined) | see below |

The main result is the ~3.4× drop in calibration error. Treat everything else with the caveats below.

## Status / limitations

Experimental; single benchmark, single model. The numbers describe `qwen2.5-abliterate:7b` on TruthfulQA MC1 in a closed-book setting — not calibration in general. Read alongside [WRITEUP.md](WRITEUP.md), which spells these out.

- **The confident-falsehood "win" is by construction.** Once the cap (~0.52) sits below the 0.70 threshold, no answer is confident enough to count, so the honest rate is *undefined* (empty denominator), not a true zero. ECE is the number that survives scrutiny.
- **The cap is blunt.** It corrects aggregate overconfidence but not per-item discrimination, so AUROC dropped toward chance. The next step is per-question calibration, not one global cap.
- **Grounding-abstain barely fired** (~5%) here: the model's own justifications usually clear the ≥2-endpoint bar, so the calibration cap did the work, not the grounding gate.
- **Accuracy on answered barely moved** — abstaining removed only a few wrong answers.
- **No unit tests, by design** — the eval is the verification, run reproducibly against held-out benchmark data.

---

Part of a set of agent-reliability / honesty-and-calibration tooling by [@insomniac-asif](https://github.com/insomniac-asif).
