# Capping an agent's confidence at its measured accuracy: what it fixes, and what it doesn't

*A deterministic honesty layer for LLM agents, evaluated two-arm on TruthfulQA MC1.*

**TL;DR.** I built a small, deterministic layer that sits between an LLM agent and its output and enforces one rule: never state more confidence than the model has *measured* accuracy to back up, and abstain on claims it can't ground. On TruthfulQA MC1 with a weak 7B local model, it cut calibration error from ECE 0.472 to 0.097 (~5×) — and left no answer confident enough (≥0.70) to count as a confident falsehood, which, as §5 argues, is mostly the cap silencing the model rather than a separate win. It also *hurt* per-item discrimination (AUROC 0.582 → 0.510). This post is mostly about those last two clauses.

---

## 1. The question

I run a local-first personal AI agent. Like every LLM system I've built, its most dangerous failure isn't being wrong — it's being wrong **confidently**: stating "done" for something it didn't verify, asserting a fact with the same 95%-certain tone whether it's right or hallucinating. So I wanted a number on one specific intervention:

> **Does capping an LLM agent's stated confidence at its *measured* accuracy — plus dropping claims it can't ground — actually improve calibration and cut confident wrong answers on a labeled benchmark, and at what cost?**

"At what cost" is load-bearing. An honesty mechanism that only reports its wins isn't a measurement; it's a demo.

## 2. Where this came from (and the one honest change that made it a study)

The two mechanisms aren't invented for a paper. I extracted them from a running agent where they already earn their keep:

- a **confidence cap** that deflates a self-reported confidence toward a rate the system has actually observed, and
- a **grounding gate** that refuses to assert a connection unless it resolves to at least two distinct, real supporting endpoints.

In the source agent, the "rate" the cap deflated toward was *my own subjective approve-rate* — useful in situ, but **not ground truth**. It's a self-referential signal: the system grading its own homework.

The single change that turns this from a personal heuristic into something measurable: **swap the subjective approve-rate for a real held-out accuracy.** Fit the cap's target on a validation split with known answers, then measure raw-vs-capped calibration on a disjoint eval split. That swap is the whole experiment. It's also the honest move — it exposes the mechanism to a benchmark that can tell it it's wrong.

## 3. The method

The layer is three deterministic stages composed into one gate, `decide()`. Each stage can only *abstain* — it never upgrades a later stage's doubt, and it never invents an answer. The caller owns the answer text; the gate only decides whether it may be asserted, and at what confidence.

**Stage 1 — grounding (`is_grounded`).** A claim is grounded iff it cites ≥ `min_endpoints` (default 2) references that each resolve to something real *and* are distinct (one source cited twice counts once). "Does this reference point at something real?" is a single injectable `resolver(ref) -> bool`. The default resolver is deliberately dumb and transparent (a ref is real iff it appears in the supplied evidence) so the rule is reproducible with no hidden database; a caller with a retrieval index or a code repo supplies their own. It **fails closed**: any error or ambiguity → not grounded → abstain.

**Stage 2 — refutation (`refute`).** A default-drop skeptic. A claim survives *only* on an explicit not-spurious verdict; anything missing, unsure, or errored is treated as spurious — the bias is toward silence over a confident-but-wrong assertion. One piece is worth calling out: a **zero-model analogy quarantine**. A metaphor like "X mirrors Y" states a resemblance, not a shared fact, so a model skeptic cannot refute it and keeps letting it through. If a claim links two different domains by resemblance language, it's dropped deterministically, with a legible reason, and *no model call*. The model judge is injectable; with none supplied the gate runs deterministic-only, so any model you add is measured against that floor rather than assumed to help.

**Stage 3 — calibration (`calibrate_confidence`).** For a claim that cleared both gates, deflate the stated confidence toward the model's measured accuracy:

```
calibrated = min(raw, measured_rate + margin)
```

with a hard invariant — it **never inflates** (`calibrated ≤ raw`, always) — plus a cold-start rule: below 8 graded items the rate is treated as thin history and capped at a cautious prior, so a tiny sample can't oversell. `measured_rate` is fit by `fit_measured_rate()` on the validation split: the fraction of validation answers that were actually correct.

The eval harness (`eval/run_eval.py`) is the one entry point. It loads TruthfulQA MC1, does a seeded 40/60 split, fits `measured_rate` on the 40% validation slice, and scores the 60% eval slice under two arms:

- **raw** — the model's own answer + its self-reported confidence, ungated.
- **honest** — that same output passed through `decide()`, using the model's own justifications as the grounding evidence and refuter endpoints.

It reports five metrics for both arms — ECE, AUROC, abstention rate, accuracy-on-answered, confident-falsehood rate — and every one is defined to show a *regression* (higher ECE, lost correct answers, over-abstention) as readily as a win.

## 4. Results

TruthfulQA MC1, seed 0, local model `huihui_ai/qwen2.5-abliterate:7b`. 40/60 split → 327 validation questions used to fit the measured accuracy (**0.37**), 490 eval questions scored. `margin = 0.15`, so the cap is `0.37 + 0.15 = 0.52`. "Confident" = confidence ≥ 0.70.

| metric | raw | honest | note |
|---|---|---|---|
| **ECE** (calibration error, ↓) | 0.472 | **0.097** | ~5× better calibrated |
| **confident-falsehood rate** (↓) | 0.571 | **undefined** | nothing clears 0.70 → rate undefined (see §5) |
| abstention rate | 0.000 | 0.045 | 22 of 490 items |
| accuracy on answered | 0.429 | 0.434 | ~unchanged |
| **AUROC** (↑) | 0.582 | 0.510 | dipped — see §5 |

![reliability diagram: raw vs honest](results/reliability.png)

On the answers it stated at ≥0.70 confidence, the raw model was wrong **57%** of the time (confident-falsehood rate 0.571); its overall accuracy on answered items was 43%. The honest arm's stated confidence never rises above the cap (0.52), so confidence and accuracy finally live in the same neighborhood: ECE drops from 0.472 to 0.097, and the reliability curve pulls off the ceiling toward the diagonal.

One rigor caveat up front: all of these are **point estimates on n=490 with no confidence intervals**, and the AUROC comparison in particular rests on just 22 abstention events, so its noise is large. Raw numbers: `results/results.json`.

## 5. What this does *not* show (the actual point)

If you take one thing from this post, take this section.

**1. "Confident falsehoods eliminated" is partly a tautology — and ECE is the honest number.** The confident-falsehood rate is only defined over answers with confidence ≥ 0.70. Once every answer is capped at 0.52, *nothing* clears 0.70, so that rate isn't "zero," it's **undefined** — the layer didn't stop being confidently wrong, it stopped being *confident at all*. That's defensible behavior for a model this weak (the cap's target, 0.37, was fit on the validation split; the eval slice itself runs ~43% accurate, so the 0.52 ceiling sits deliberately close to the model's real reliability). But it would be dishonest to sell it as the headline. The number that actually survives scrutiny is **ECE**, because it measures the confidence-accuracy gap across *every* bin, not just the top one. The cap earns the ECE improvement; it gets the confident-falsehood "win" for free by definition.

**2. The cap is blunt: it buys calibration at the price of discrimination.** AUROC went *down*, 0.582 → 0.510 — from "slightly better than chance" to "chance." Two things matter here, and the second is subtle. First, the arms feed AUROC different signals: the raw arm is scored on the model's own graded confidence, which carries a little ranking signal (its 0.9s were marginally more often right than its 0.8s). The honest arm is scored on the layer's answer-vs-abstain decision (1 = answer, 0 = abstain) — and because it abstains on only 22 of 490 items, and those abstentions aren't preferentially wrong, that near-constant gate can't separate correct from incorrect: ~0.5 is what a flat signal earns. Second, scoring the honest arm's confidences instead wouldn't rescue it — 468 of 490 land at exactly 0.52 and the rest at 0, so there is almost no variation left to rank with. Either way you score it, **the layer replaces a weakly-informative graded signal with an almost-flat one.** One global cap fixes aggregate overconfidence but cannot tell a good answer from a bad one.

**3. The grounding gate barely fired, so it's mostly untested here.** The grounding gate abstained on 21 of 490 items (4.3%) and the analogy quarantine on 1 more (total abstention 4.5%). In closed-book TruthfulQA the only "evidence" available is the model's own justifications, which almost always clear the ≥2-endpoints bar — a confabulating model is happy to produce two confident-sounding reasons. So the calibration cap did essentially all the work. The gate is built to lean on a *real* resolver — a retrieval index, a code repo, a KB — and this eval doesn't give it one. Its value is unproven until it's tested against external endpoints, not self-justifications.

**4. External validity is thin.** One weak 7B model, one seed, one benchmark, MC1 only, no confidence intervals. The direction of the effect should be robust (a hard cap will always deflate an overconfident model's ECE), but the magnitudes are specific to a model that happens to be very overconfident and ~40% accurate. A stronger model has less room for the cap to help and more discrimination for it to destroy — larger models are already reasonably calibrated on well-formatted multiple-choice (Kadavath et al., 2022), so this intervention is for the regime where they aren't: smaller local models, agentic settings, distribution shift.

## 6. Where the same gap shows up (RAG, and yes, NotebookLM)

The layer splits into two halves: **grounding/abstain** (cite real support or stay silent) and **calibration** (never state confidence above measured accuracy). Production retrieval-augmented systems have shipped the *first* half — they cite sources so you can verify — but not the second. A grounded assistant hands you a cited claim with no signal for whether the underlying source is a strong result or a weak one, and no measured reliability behind the certainty of its phrasing. That is the same asymmetry this eval exposes in miniature: grounding is the easy, visible half; honest *calibration* of the grounded claim is the half nobody measures. Closing it is the point of the directions below.

## 7. Future work

- **Per-question calibration.** Replace the single global cap with a calibrator that conditions on the item, so the layer can *rank* as well as deflate. The bar is concrete: recover AUROC *above* the raw 0.582 while holding ECE near 0.10 — deflate without flattening. The missing ingredient is a per-item signal that actually varies with correctness — token logprobs, agreement across self-consistency samples, or a grounding score backed by real retrieval instead of self-citation — then a monotone map (Platt or isotonic, fit on the same held-out validation split) from that signal to a probability. Temperature scaling (Guo et al., 2017) is the reference point — one scalar on the logits, rank-preserving, fixing ECE without touching AUROC, the exact mirror of what my cap did — but it can't be applied here directly: this pipeline has only the model's *verbalized* confidence, not logits. This bullet directly targets the AUROC regression, the most important negative result here.
- **A retrieval-grounded arm.** Give the grounding gate real endpoints (a retrieval index over a corpus) instead of the model's own justifications, on an open-book benchmark, so abstain-on-ungrounded is actually exercised and can be measured.
- **Scale the sweep.** Bootstrap confidence intervals, ≥5 seeds, multiple models across the calibration spectrum, and MC2/generative variants — to separate "the cap always deflates ECE" (trivially true) from "the layer helps a model you'd actually deploy."

## 8. Relation to prior work

This sits downstream of three lines of work: **TruthfulQA** (Lin, Hilton & Evans, 2021) as the benchmark purpose-built for confident imitative falsehoods; the **calibration** literature — Guo et al. (2017) for ECE, reliability diagrams, and temperature scaling, which is exactly the per-item direction §7 points at; and the **model self-knowledge** line — Kadavath et al. (2022) on whether models know what they know, which frames the question my grounding gate crudely approximates from the outside. My contribution is narrow and empirical: not a new method, but an honest measurement of a simple, deployable intervention — *and* a clear statement of where it fails, which is the part a method paper usually omits.

**References**

- Lin, S., Hilton, J., & Evans, O. (2021). *TruthfulQA: Measuring How Models Mimic Human Falsehoods.* ACL 2022. [arXiv:2109.07958](https://arxiv.org/abs/2109.07958)
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks.* ICML 2017. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)
- Kadavath, S., et al. (2022). *Language Models (Mostly) Know What They Know.* [arXiv:2207.05221](https://arxiv.org/abs/2207.05221)

## 9. Reproducing

```bash
pip install -r requirements.txt
python eval/run_eval.py --n 817 --seed 0 --model huihui_ai/qwen2.5-abliterate:7b
# writes results/results.json + results/reliability.png
```

(TruthfulQA MC1 has 817 questions, so `--n 817` selects all of them; `round(817 × 0.40) = 327` gives the 327/490 split. `--n 0` is equivalent.)

Code: https://github.com/insomniac-asif/honest-confidence (MIT). The layer is `honest_confidence/` (four stdlib-only modules); the eval is `eval/`; `honest_research/` is a runnable end-to-end demo (`check` a single claim, or `research` a video/social URL into per-claim calibrated confidence).
