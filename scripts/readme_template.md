# DriftGuard

**Reliable network anomaly detection under distribution shift.**

Most intrusion-detection benchmarks shuffle the data, train on 80% of it, and test
on the other 20%. That measures how well a detector separates attacks from
background traffic *in the same period it was trained on*. It says nothing about
what happens six months later, when the network has changed underneath it.

This project asks the question that shuffling cannot answer, and then does
something about the answer:

> **Does a machine-learning network anomaly detector trained on historical traffic
> continue to detect attacks reliably when evaluated on later, previously unseen
> traffic?**

> **Which methods for detecting and adapting to distribution shift recover
> detection performance most effectively while respecting a fixed false-positive
> budget?**

---

## Why I Chose This Question

Benchmark results age badly, and they age in a specific way. A published F1 of
0.95 means the model separated two classes captured within days of each other,
over the same links, with the same mix of traffic. It does not mean the model
will score 0.95 on traffic collected next year. Yet that is exactly how the
number gets used: as a statement about a detector, rather than about a
model-and-dataset pair frozen at one moment.

The second reason is methodological. Distribution shift is the standard
explanation offered when a deployed detector starts missing attacks. But "the
data changed" is a hypothesis, not a diagnosis. You can only test it by
measuring the change and measuring the damage separately, then asking whether
detecting the first would have predicted the second. This project does exactly
that: it detects drift four different ways, measures what each shift actually
did to the distribution, and reports the cases where the two disagree.

I also wanted to resist the leaderboard framing. The interesting result is not
which model wins. It is whether the answer to "does it still work" depends on the
model, and whether the obvious repair — retrain on recent data — is reliably the
right one.

## Research motivation

The direction is established research, not something this project invented.
Concept drift in intrusion detection has been surveyed extensively, and the
UGR'16 dataset exists precisely because the field recognised that older captures
could not answer questions about current networks. The specific research question
above, and the experiment design that answers it, were formulated for this
project.

**Not specified by the public source. This is a research assumption.** The
premise that a detector's published accuracy decays on later traffic is an
assumption this project set out to measure rather than take on trust. Where the
literature motivates a problem, that is different from the literature proving the
answer.

## Threat model

The adversary in this project is not the attacker in the dataset. It is the
passage of time.

**Trusted:** the historical capture, treated as a faithful record of the network;
and the benchmark's labels, treated as correct. Neither is verified.

**Untrusted:** everything after the training period. Nothing about the forward
period — not its features, its class prevalence, its duration, nor its labels —
may influence any fitted object before it is scored.

**Explicitly out of scope:** an attacker who knows the feature set and adapts to
evade it, a compromised capture point, and label poisoning. None are tested, and
no claim is made about them. There is **no adversarial-robustness claim** in this
work.

The privacy claim is narrow and worth stating precisely: features are metadata
only. No payload bytes, no IP addresses, no ports become predictive inputs. That
reduces what an observer can learn from the model. It is *not* a formal privacy
guarantee against a traffic analyst, and it is not differential privacy.

---

## Datasets

| Dataset | What it is | What it can support |
|---|---|---|
| **UNSW-NB15** | Labelled flow records with attack categories, captured over three sessions in January and February 2015. | A **relative** temporal comparison: does a detector keep working on later captured traffic. |
| **UGR'16** | Netflow records from a Greek ISP network, published as 23 weekly archives totalling roughly 215 GB. | The **stronger** temporal experiment, on a real network rather than a lab. |

These are used for different jobs on purpose.

UNSW-NB15 has excellent labels and a well-known attack taxonomy, but its time
axis is three discrete capture sessions across 27 days, and **one of those
sessions contains no attacks at all**. It therefore supports asking whether
performance falls on later captured traffic, and nothing stronger. This
repository does not claim deployment stability, because a dataset spanning 27
days cannot support it.

UGR'16 is the reverse: weaker per-flow labelling, but months of traffic from a
real ISP link with genuine attack traffic. It carries the temporal argument.

### The UGR'16 subset, exactly

The full UGR'16 release is about 215 GB across 23 weekly archives. This project
uses **six weeks**:

- **Calibration weeks (background only, never used for training):** `march_week3`, `may_week1`
- **Weeks carrying the temporal experiment:** `july_week5`, `august_week1`, `august_week2`, `august_week3`

That is 11,775,183 flows spanning 2016-03-18 to 2016-08-15. August weeks 1–3 are
three consecutive weeks with both normal and attack traffic, which is what makes
the forward test genuinely later data rather than a reshuffle.

**This is a subset, not the full dataset.** No claim is made about the weeks that
were not downloaded, and the report does not present UGR'16 as a full-dataset
result. UGR'16 attacks are produced by replaying malware and tools against a real
link, so attack realism is bounded by that setup, and the netflow export carries
no forward/backward direction split, which removes a feature group UNSW-NB15
provides.

---

## Temporal evaluation

Every experiment splits the data in time, never at random:

```
|------------------------ train ------------------------|-- val --|-- backtest --|------ forward ------|
        earliest 50%                next 15%        next 15%     latest 20%
                                                    ^                      ^
                              threshold chosen here    scored once, after the fact
```

- Periods are **contiguous and ordered**. An identical timestamp is never split
  across a boundary, because UNSW-NB15 stamps many rows with the same second.
- The **backtest** is held out inside the historical distribution. It measures how
  much the model decays *before* the forward period is touched at all.
- The **forward period** is scored exactly once, by a model that has not seen it.

The split-mode ablation runs the random split as an explicit control, because that
is the protocol every shuffled benchmark implicitly uses. It is labelled a
leakage-oriented control, not a valid production protocol: it fits its
preprocessor without a temporal cutoff, which is precisely the leak this project
exists to avoid.

## Leakage prevention

Five guards enforce that nothing from the future reaches the model, each covered
by a test that fails if the guard is removed:

1. Contiguous, ordered periods; no timestamp is split across a boundary.
2. The scaler and encoder are fitted on the **training period only**. Fitting on
   all data and then splitting leaks the forward period's mean and variance into
   training — the most common way a "temporal" split quietly becomes a random one.
3. The operating threshold is chosen on the **validation** split, never on the
   backtest or forward periods.
4. The backtest is used for exactly one thing, and the forward period is scored
   once.
5. **Adaptation is given only the window immediately before the forward period.**
   Rolling adaptation re-fits on a cadence, and at each step the fit may use only
   rows earlier than the step it is about to score.

There is also a guard against silent failure, which turned out to matter more than
any of the above: if drift is enabled but produces an empty table, the run
**raises** rather than writing `{}`. An empty result is indistinguishable from
"no drift", and a benchmark that reports the first while meaning the second is
worse than no benchmark.

---

## Architecture

```
src/
  privacy_preserving_nad/   V1: the original metadata-only pipeline, preserved intact
  driftguard/
    data/        dataset adapters (UNSW-NB15, UGR'16, synthetic), common schema
    temporal.py  chronological splitting, quantile boundaries, leakage guards
    features/    the common feature set and the train-only preprocessor
    models/      majority, logistic regression, random forest, gradient boosting
    drift/       KS, scaled Wasserstein, PSI, block-mean CUSUM; windowing
    adaptation/  five strategies, all constrained to pre-forward data
    shift/       eight controlled shift families, with realized-shift measurement
    evaluation/  metrics at a fixed FPR, calibration, failure analysis
    experiments/ immutable experiment tracking, ablations
    reporting/   figure and PDF generation from recorded output
    pipeline.py  the run that ties it together
```

**V1 is preserved, not replaced.** The original
`Privacy-Preserving-Network-Anomaly-Detection` implementation is intact under
`src/privacy_preserving_nad/`, with its own tests, config and results, and its
commit history is preserved. DriftGuard is the evolution of that project.

### Feature policy

Features are metadata only, and that is enforced rather than aspirational: the
preprocessor refuses to fit on a field the policy disallows. V1's five
payload-derived features were **removed** after finding that the metadata-only
claim was not strictly true as written originally. The feature-policy ablation
now reports "not applicable" for that reason — there is no richer arm to compare
against, and inventing one would contradict this project's threat model.

---

## Models

| Model | Role |
|---|---|
| Majority baseline | The floor. A detector that cannot beat it is not a detector. |
| Logistic Regression | The interpretable linear reference, and the cheapest model to adapt. |
| Random Forest | The workhorse nonlinear baseline. |
| Gradient Boosting | The strongest model here, and the most expensive to refit. |

All four run on the headline benchmark. Ablations use the two cheap models,
because the question there is about an effect, not about which model is best.

## Drift detection

| Method | What it measures |
|---|---|
| **Kolmogorov–Smirnov** | Maximum distance between the two empirical CDFs. |
| **Wasserstein** | Distance the data has to be moved to match, scaled by reference spread. |
| **PSI** | Population Stability Index over binned distributions. |
| **CUSUM** | Sequential detection over **block means**, with a null-calibrated threshold. |

Two decisions here are load-bearing, and both were made after measuring:

- **An alert requires an effect size and a significance threshold together.** With
  hundreds of comparisons, a nominal p-value threshold fires almost everywhere and
  measures the number of comparisons rather than the traffic.
- **CUSUM runs over block means, not raw rows.** CUSUM is a sequential test, so
  its power depends on the length of the sequence: over 100k raw flows it
  accumulates noise proportional to the row count, and alerts on everything. Its
  threshold is calibrated by simulating the null *at the same sequence length*,
  which is what makes the alert rate mean something.

Drift detection and model failure are reported as separate events. Neither
"drift detected" nor "no drift" is treated as a proxy for the other.

## Adaptation

| Strategy | What it does | Cost |
|---|---|---|
| `none` | Nothing. The baseline every other row is compared to. | — |
| `threshold_recalibration` | Re-picks the operating point on permitted historical data. Model is untouched. | Seconds |
| `recent_window_retrain` | Refits on the single window immediately before the forward period. | One fit |
| `rolling_window_retrain` | Refits on a cadence through the forward period; each step sees only earlier rows. | Many fits |
| `historical_plus_recent_retrain` | Refits on history plus the recent window, at the cost of re-seeing old data. | One fit |

There is no overall "winner" here. The interesting result is where each strategy
helps and where it fails, which is not the same thing.

---

## Main results

Every number below is generated from recorded experiment directories by
`scripts/render_results.py`. None of it is typed by hand. A stage that has not
been run is reported as not run.

<!-- BEGIN GENERATED RESULTS -->
<!-- END GENERATED RESULTS -->

---

## What I got wrong along the way

The first "completed" UNSW-NB15 benchmark looked healthy and was measuring almost
nothing. Each of these produced a clean-looking result with no error logged:

1. **The drift stage silently did nothing.** The configured window was longer
   than the forward period, so zero windows fit and the summary was written as
   `{}` — readable as "no drift detected".
2. **All three retraining strategies failed on all four models, every time.** The
   code cloned a model and immediately discarded the clone, and the clone call
   raises on the majority baseline because it is not a scikit-learn estimator. Only
   2 of 5 required strategies ever ran, and the run reported success.
3. **The UGR'16 reader silently lost one row per file**, treating the leading
   blank line as data.
4. **"Rolling" adaptation never rolled.** It performed a single fit before the
   forward period, making it byte-identical to recent-window retraining — while
   being labelled as two distinct strategies in the results table.
5. **The class-prevalence controlled shift did nothing.** It returned its input
   unchanged for any target above the current rate, and recorded the *requested*
   magnitude rather than measuring what happened.

What caught these was opening the output files and finding them suspiciously
empty or suspiciously complete — not reading the log.

The fixes are now guarded: an empty drift table fails the run, a strategy that
raises aborts with every failure listed, and an ablation that does not apply
reports why instead of printing two identical numbers. Two of those guards caught
real bugs of mine within minutes, including a config typo I introduced while
fixing an earlier one. The superseded run is preserved under `results/superseded/`
rather than deleted, because the contrast between it and the corrected run is
itself evidence that the fixes mattered.

---

## Controlled shifts

Eight shift families are applied to real data: packet size, duration, packet
rate, inter-arrival time, byte rate, protocol mixture, class prevalence, and
telemetry reduction.

Every row records **both** the requested shift and the realized one, with a
median ratio, an absolute median delta, Cohen's d, the label rate, and a
`realized_verified` flag. A perturbation that fails to move the distribution is
reported as such. One such case is already visible and is left in the results
rather than tuned away: a `byte_rate` shift of 1.5 achieves Cohen's d ≈ 0.044,
which is a very weak change by any measure.

## Ablations

Six ablations, each answering one methodological question rather than filling a
grid: temporal against random splitting, the metadata feature policy, drift
thresholds, adaptation window size, telemetry feature families, and the
false-positive budget. Their results are in the generated block above.

## Failure analysis

Failure analysis is computed from the prediction-level tables of each run, not
written by hand: false-positive and false-negative concentrations, their score
distributions, the windows where drift alerted but performance held, and the
windows where performance fell without a strong drift signal. The last two are
the interesting ones, because they bound how much a drift monitor can be trusted
as a proxy for model health.

---

## Reproduction

```bash
# 1. Install (Python 3.10+)
python -m pip install -e .
python -m pip install pytest tabulate

# 2. Point the pipeline at a raw-data directory that is not inside the repo.
#    The 2.5M-row UNSW parquet and the UGR'16 weekly archives are large, so
#    they are deliberately kept out of version control.
export DRIFTGUARD_RAW_DIR=/path/to/raw   # Windows: set DRIFTGUARD_RAW_DIR

# 3. Fetch UNSW-NB15 (preserving its timestamps)
python -m driftguard data fetch --dataset unsw_nb15

# 4. Run the experiments
python -m driftguard benchmark --config configs/benchmark.yaml       # UNSW-NB15
python -m driftguard benchmark --config configs/ugr16_subset.yaml    # UGR'16 subset
python -m driftguard shift     --config configs/shifts.yaml         # controlled shifts
python -m driftguard ablate    --config configs/ablations.yaml      # ablations

# 5. Regenerate every derived artifact from the recorded runs
python -m driftguard report --output docs/technical_report.pdf
python scripts/render_results.py
```

A fast sanity check that needs no dataset at all:

```bash
python -m driftguard benchmark --config configs/smoke.yaml
```

### Tests

```bash
python -m pytest -q
```

The suite covers the temporal leakage guards, the drift-method registry, the
adaptation strategies, the dataset adapters (including UGR'16 against real
archive bytes), controlled shifts, figure generation, configuration validation,
and the silent-no-op regressions described above.

### Docker

```bash
docker build -t driftguard .
docker run --rm driftguard python -m pytest tests/ -q
docker run --rm driftguard python -m driftguard benchmark --config configs/smoke.yaml
```

### CI

`.github/workflows/ci.yml` runs the suite on every push, with configuration
validation as a separate job. The full multi-model historical benchmark is
deliberately **not** in ordinary CI: it takes hours and needs the raw data. CI
runs the synthetic smoke configuration instead, which exercises the same code path
in seconds.

### Technical report

`docs/technical_report.pdf` is generated from the recorded experiment
directories by `python -m driftguard report`. Like the README results block, it
contains no hand-typed numbers.

---

## Limitations

- **UNSW-NB15's time axis is three sessions across 27 days, one of which has no
  attacks.** It supports a relative temporal comparison and nothing stronger. It
  cannot speak to long-term deployment stability, because it spans weeks.
- **UGR'16 results are a six-week subset.** Periods outside the subset are
  unmeasured. The subset bounds the time axis as well as the traffic variety.
- **The two datasets are not directly comparable.** Different label definitions,
  attack taxonomies and collection tools. Any cross-dataset claim is about the
  method, not about the numbers.
- **The forward period is later traffic from the same capture environment.** The
  shift it contains is whatever the environment itself did, not a designed stress
  test. Controlled shifts attribute degradation to named properties, but they
  perturb a dataset whose attack labels are already synthetic.
- **The operating threshold comes from one validation slice at a fixed
  false-positive budget.** A different budget gives different numbers, and the
  target-FPR ablation shows how sharply recall depends on that choice.
- **Calibration error is estimated on slices as small as the attack class.** Those
  numbers are point estimates without confidence intervals, so small differences
  between models should not be over-read.
- **No adversarial-robustness claim** and **no production-readiness claim.** These
  are offline experiments on public datasets.
- **Drift detection is not model monitoring.** A drift alert and a performance
  drop are separate events, and this project measures how often they disagree.

## Future work

- A continuous-evaluation harness that watches a live stream, rather than
  evaluating two fixed windows after the fact.
- Paired significance testing across time windows, so "this period was harder" can
  be separated from "this model got worse".
- Connecting drift alerts to operational cost: a drift signal is only useful if
  acting on it is cheaper than the false alarms it prevents.
- Feature-level attribution of degradation, so the answer is "the packet-size
  distribution moved" rather than "drift was detected".

## License

MIT. See `LICENSE`.

## Citation

If this work is useful, please cite the datasets you actually used. The UNSW-NB15
and UGR'16 papers are the primary references; the methodology here is mine.
