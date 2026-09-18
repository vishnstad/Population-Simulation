# Panel demo script (checklist 6.6)

Twelve minutes. Two scenarios, one of which the system **refuses** to simulate.
The refusal is the demo, not a caveat on it.

```
streamlit run app/app.py
```

Pick the arm in the sidebar. Everything below is the app's own flow.

---

## 1 · The thing the survey already answers (40 s)

Paste the verbatim wording of `natroad`:

> *(… are we spending too much, too little, or about the right amount on)
> Highways and bridges*

The router returns **observed** at cosine 0.91, and the app says so:

> *observed — this is `natroad`. Answered from the weighted cross-tab; no model
> was asked.*

**The line to say:** the objection to every project like this is *you built an
expensive way to reproduce a cross-tab you already had*. The answer is
architectural, not rhetorical: the model is never asked a question the data
answers. That is a routing decision made before any call is made.

## 2 · The thing it cannot (5 min)

Load `scenarios/us_bridge_levy.yaml` — a $4 bridge levy on vehicle registration.
Close to `natroad` in subject, and nothing in the GSS settles it.

The router returns **simulate**, and *before any numbers appear* the honesty box
says what the system's error was on the nearest questions it was actually scored
on, with the baseline and the noise floor beside it.

**The line to say:** this number has no ground truth and never will. The only
honest thing that can be said about it is how the same system did on the measured
questions most like it — so that sentence is rendered above the distribution, not
under it.

Then the population rollup (raked to ACS 2024, not to the survey sample), the
90 % intervals, and the segments furthest from the population answer.

**Point at the intervals.** Built from elicitation spread alone they covered 15.7 %
of true values at a nominal 90 %. With the out-of-fold anchor residual added they
cover 89.6 %. Three draws of one prompt agreeing with each other says nothing
about how far they all are from the truth.

## 3 · Why anyone should believe the honesty box (4 min)

Switch to `runs/report/b17_results.html`.

- **The claim.** 39 held-out GSS items, 56 clusters. Achieved W1 against the
  national-marginal baseline and the truth's own noise floor — three numbers,
  never one.
- **`s = 0` *is* B0a.** The baseline is an ablation of the method, so the
  comparison is only about whether subgroup structure was added.
- **The arms table.** Uncalibrated elicitation is worse than copying the national
  marginal; calibrated it is a fifth of that error. Nearest-anchor is worse still,
  so this is not item similarity in disguise.
- **The capability curve.** 3B → 8B → 14B, monotone, and each arm's strength was
  predictable from its *anchor* fit before a single target was scored.

## 4 · What we do not claim (90 s)

Read the last section of the report out loud. Not "simulates society", not
"predicts policy outcomes", not "replaces surveys" — the system is built on them
and dies without them. The held-out-item score is the honest proxy for a scenario
and the honesty box is where that is said to the reader.

---

### If a panel member asks…

**"Did it just memorise the GSS?"** The probe names GSS, names the waves, names
the subgroup and asks for the published cross-tab. Two items of 39 clear it, one
of them `polviews` — the most cross-tabbed item in social science. Removing both
moves the headline by 0.0001.

**"Does it only work because the anchors are on the same topics?"** Refit the
calibrator with every same-topic anchor removed and the result moves by 0.0003.

**"What does the stat card actually contribute?"** Demographics alone get −12.0 %
against the baseline; adding the anchor answers takes it to −18.3 % at a matched
ensemble. The anchors are real and they are the smaller half. That is on the
slide, not in the footnotes.

**"Where does it fail?"** It places the subgroups in the right order and too close
together — predicted between-cluster spread is about half the truth. Ranking
segments is well served; the size of the gap between two segments is understated,
and the interval is the thing to read there.
