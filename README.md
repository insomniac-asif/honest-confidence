# honest-confidence

A small, deterministic **honesty layer** for LLM agents — and a reproducible eval that asks whether it actually works.

An agent should not say *"I'm 95% sure"* when the class of thing it's doing is only right ~62% of the time, and it should **abstain** rather than confidently assert a claim it can't ground. This repo extracts two such mechanisms from a running local agent, generalizes them, and — crucially — **measures them against an external labeled benchmark** to see if they improve calibration and reduce confident falsehoods, *including where they hurt*.

## The one question

> Does capping an LLM agent's stated confidence at its **measured accuracy**, plus dropping **ungrounded** claims, actually improve calibration and cut confident wrong answers on a labeled benchmark — and at what cost?

## Why this is honest (and not a demo that only shows wins)

The mechanisms come from a personal agent where the "measured rate" was the *owner's subjective approve-rate* — useful in place, but **not ground truth**. The honest move, and the whole point of this repo, is to **swap that subjective signal for a real held-out accuracy** and grade against a benchmark with known answers. We report the calibration improvement **and** the cost (correct answers lost to over-abstention). A method that only reports its wins isn't a measurement.

## What's inside

| module | what it does |
|---|---|
| `honest_confidence/calibration.py` | `calibrate_confidence(raw, measured_rate)` → deflates toward measured accuracy; **never inflates**. `fit_measured_rate()` fits the rate on a held-out split. |
| `honest_confidence/grounding.py` | abstain unless a claim resolves to **≥2 distinct supporting endpoints** (pluggable resolver). *(in progress)* |
| `honest_confidence/refuter.py` | a zero-model gate: default-drop on spurious / trivial / analogy-quarantined claims. *(in progress)* |
| `honest_confidence/decision.py` | the glue: `decide(question, raw_conf, evidence) → answer \| ABSTAIN, calibrated_conf, reason`. *(in progress)* |
| `eval/run_eval.py` | one reproducible entry point: raw model vs. model+honesty-layer on **TruthfulQA**, reporting ECE, AUROC, abstention rate, accuracy-on-answered, and confident-falsehood rate. *(in progress)* |

## The eval, briefly

- **Benchmark:** TruthfulQA (817 questions, purpose-built for *confident, imitative falsehoods*).
- **Arms:** a local model answering with self-reported confidence (**raw**) vs. the same output passed through the honesty layer (**honest**).
- **Split:** hold out 40% to fit the calibration target (measured accuracy), evaluate on the other 60%. Seeded and reported.
- **Metrics:** ECE (calibration error), AUROC (does the abstain signal separate right from wrong?), abstention rate, accuracy-on-answered, and confident-falsehood rate — raw vs. honest.

## Status

Early / in progress. Calibration module and the framing are in; grounding, refuter, decision glue, and the eval harness are landing next. Results and the writeup will follow here.

## License

MIT — see [LICENSE](LICENSE).
