# Session notes: ButterflAI 2.0, Week 2 (`02_rate_1d.ipynb`)

## 0. Error found in the notebook

**The mesh is not "finer than the one-day quantization."** The quadrature mesh has 4001 nodes (4000 intervals) over a 17-year window:

> 17 yr / 4000 × 365.25 d/yr ≈ **1.55 days** per step

That is *coarser* than one day, not finer. The wrong comparison appears in:

- the setup cell's printed message ("finer than the catalog's 1-day quantization")
- the Section 2 text ("a spacing of 1.55 days — finer than the one-day quantization of the catalog itself")
- the rescaled-gaps task note, which says the mesh is "finer than a day, so linear interpolation between nodes costs nothing"

Getting under one day would take about 6,211 nodes (the 6,401-node row in Section 3's refinement table, at ~0.97 days, is the first one that does).

**Why it doesn't break anything.** The quadrature integrates the model's smooth rate λ(s), not the data, so the catalog's day-level rounding of event times is irrelevant to it. What justifies the mesh is Section 3's refinement test: halve the spacing and check that the integral changes by a shrinking amount (a factor of ~4 per halving for the second-order trapezoid rule).

*Smaller note:* the "√600 ≈ 25 events of Poisson scatter" remark is loose. That is the fluctuation in the *total* count. Per-bin scatter is smaller (a bin expecting ~40 events has √40 ≈ 6).

---

## 1. Earlier in the session

**Quadrature** has several meanings. In solar physics: (1) *geometric*, two viewpoints 90° apart (e.g. STEREO spacecraft seeing a CME side-on); (2) *phase*, two oscillations 90° out of phase (polar field vs sunspots in the dynamo); (3) *numerical*, evaluating an integral as a weighted sum (angular integrals in radiative transfer); (4) *signal processing*, I/Q channels. This notebook uses sense 3.

**PyTorch:** open-source ML framework built on tensors, automatic differentiation (`.backward()`), and dynamic computation graphs. Not needed until Week 5; Phase A is numpy and scipy.

---

## 2. The data split

| Term | Meaning |
|---|---|
| **Development set** | Cycles 12, 13, 14, 15, 17, 18, 19, 21, 23: 18 hemicycles, 16,611 events |
| **Withheld cycles** | 16, 20, 22, sealed for one final reveal; never fitted, tuned or selected on |
| **Hemicycle** | One hemisphere of one cycle (e.g. `19N`); the unit of replication |
| **Representative split** | Withheld cycles span the middle of the amplitude range: built to catch overfitting to validation folds, *not* to test extrapolation |

- **Why withhold whole cycles:** later models condition on the opposite hemisphere, so holding out 20N while training on 20S would leak the held-out unit.
- **Why keep 19 (strongest) and 14 (weakest) in training:** they anchor both ends of the amplitude range, so no scored cycle requires extrapolating in amplitude. They are also high-leverage points for any amplitude dependence. Cost: a good score shows interpolation, not that the model could forecast a record-breaking cycle. Caveat: each extreme has only one cycle, and its two hemispheres are not independent.
- **Overfitting the validation set:** each variant you select using the same folds leaks a little information into your choices; the sealed cycles give one unbiased check.

---

## 3. Poisson processes and noise

- **Poisson process:** events arrive independently at some rate. Counts in a bin are Poisson with mean μ.
- **Mean = variance:** for Poisson counts, Var = μ, so typical scatter is √μ. This follows from many rare, independent chances adding up (each has variance ≈ its small probability p).
- **Poisson scatter:** the irreducible counting noise. Even a perfectly smooth λ(s) gives a ragged histogram. The ± values are typical deviations in either direction, not extra events.
- **Fano factor** = variance / mean. It equals 1 for a Poisson process. Week 1 found ~70 (**overdispersion**), mainly because the rate varies by an order of magnitude across a wing, so pooling quiet and busy stretches inflates the variance. This is a varying rate, not necessarily true clustering.
- **Underdispersed** means more regular than random (variance < mean).

---

## 4. The likelihood on event times

**Rate / intensity function λ(s):** expected events per year at time s. A model with varying λ is an **inhomogeneous Poisson process**.

**Log-likelihood:**

> log L = Σᵢ log λ(sᵢ) − ∫ λ(s) ds

- First term: rewards high rate *where events happened*.
- Integral term: the expected total count; penalizes high rate *everywhere*, including where nothing happened. It is not optional (without it, log L = n log c just runs away as c grows; Figure 2).
- **Observation window:** [−6, +11] yr in `s`, fixed program-wide. Quoting a likelihood without its window is meaningless.
- **Units:** nats per hemicycle-year (this week) rather than per event (last week). A per-time score can reward getting the *number* of events right.
- **Constant-rate check:** for rate c, log L = n log c − cT exactly, and the maximizer is ĉ = n/T. This is the baseline every model must beat.
- **Task 10 guards:** return `-inf` if any rate at an event is ≤ 0 or non-finite; also if the *total* is non-finite. The sentinel must be **−inf**, never +inf, which would look like a perfect score.
- **Implementation notes:** call `rate_fn` once and reuse it; return a plain `float`; `event_s` = data, `rate_fn` = model (a function of an array), `s_mesh` = integration grid (a parameter so Section 6 can pass a wrong window on purpose).
- **Quadrature / trapezoid rule:** approximate the integral by evaluating on a grid and summing trapezoids. Verified by **refinement** (no closed form needed).
- **Verified result:** numeric and analytic log L agreed to ~1e-12; at c = n/T the value was largest (5625.86 vs 5370.12 at c = 50 and 5024.32 at c = 200 for 19N, n ≈ 1,590).

### Python idioms met along the way

- `lambda s: np.full_like(np.asarray(s, float), c)`: an inline function returning the constant `c` with the same shape as `s`, so it works at both the event times and the mesh nodes.
- Passing functions as arguments (`rate_fn`) is why "only λ changes" from week to week.
- `lambda s, c=c_hat: ...` freezes `c` at definition time; without the default, the lambda looks up `c` when called, which bites when lambdas are built in a loop.

---

## 5. The gamma-shaped rate

> λ(s) = a · (s − s₀)^(k−1) · exp(−(s − s₀)/θ) for s > s₀, else 0

| Parameter | Role | Constraint |
|---|---|---|
| `a` | amplitude (pure scale) | a > 0 |
| `θ` | decay timescale, years | θ > 0 |
| `k` | sharpness of the rise | k > 1 |
| `s₀` | onset: when emergence becomes possible | S_A < s₀ < s_first |

- **Peak** sits at s₀ + (k − 1)·θ.
- **Gamma probability density:** a one-humped, right-skewed family on x > 0 with x^(k−1)·e^(−x/θ) shape and a normalizing constant. The notebook's rate has the same *form* but is **not** a density: `a` replaces the normalizer, so ∫λ = expected number of events.
- **Unconstrained parameterization:** optimize over four free reals q:
  - a = e^q₀, θ = e^q₁, k = 1 + e^q₂, s₀ = S_A + (s_first − S_A)·σ(q₃)
- **Logistic** σ(x) = 1/(1 + e^−x), mapping ℝ → (0, 1). **Logit** is its inverse: logit(p) = log(p/(1−p)), used to turn a starting fraction into q₃ (0.5 → 0).
- **Task 11 bug I hit:** multiplying by (1 + e^−q₃) instead of *dividing* by it. This gave an onset of −2.0 at q₃ = 0 instead of the −5.0 the "halfway" check predicts.
- **Sanity checks:** q₃ = 0 should give the midpoint between `S_A` and `s_first`; large positive q₃ → just below the first event; large negative → just above the window start.

---

## 6. Fitting (Task 12)

- **Maximum likelihood estimate (MLE):** the parameters maximizing the log-likelihood of the observed event times.
- **Objective:** a function of `q` alone that the optimizer minimizes: minus the Task 10 log-likelihood, with data, mesh and `s_first` fixed from the surrounding code. Return a large **finite** number (not +inf) for failures so Nelder–Mead can still rank bad points; something like 1e12 is safer than `np.finfo(float).max`.
- **Nelder–Mead:** derivative-free simplex method; suits a cheap likelihood with no hand-written derivatives, and is robust to flat regions.
- **Multiple starting points:** the surface may have several local maxima; run several starts and keep the best. For each start:
  1. choose starting `k` (> 1) and a starting onset fraction (strictly in (0, 1))
  2. θ = (target peak − s₀)/(k − 1), with the target peak near s = 0 (Figure 1 puts the maximum within a year or two of 0). Recompute per start so only the *shape* varies, not the peak location
  3. `a` = n / ∫(rate with a = 1), so the starting expected count equals the observed count n = `len(event_s)`
  4. convert to q (log, log(k − 1), logit of the fraction) and run `minimize` twice, the second time from the first's answer `.x`
  5. keep the result with the lowest `.fun`; return `(q_hat, s_first)`
- **Free-scale identity:** d(log L)/d(log a) = n − ∫λ, so at the MLE the expected count equals the observed count. Useful as a check on any fit.
- **Bugs fixed along the way:** overwriting `event_s` with its first element; `np.zip`/`np.random.default_rng.random` misuse; missing initial `q_hats = []`; passing a result object where a vector (`.x`) was needed; `log(k + 1)` instead of `log(k − 1)`; a hard-coded q₃ that didn't match the s₀ used for θ and `a`; the third `gamma_rate` argument being `event_s` instead of `s_first`; `min` over result objects without `key=lambda r: r.fun`.
- **Reproducibility:** `np.random.default_rng()` with no seed ignores the notebook's seed; seed it, or use a small fixed grid of starts.

---

## 7. Identifiability (previewed; Task 18 comes later)

**Identifiability:** whether the data can distinguish different values of a parameter. It belongs to the model *and* the data, not the optimizer, and a tidy parameterization does nothing to fix it. A parameter is determined when:

1. different values make different predictions,
2. enough events fall where it matters (the rising edge has very few),
3. there are enough events relative to Poisson noise,
4. it can't be offset by moving other parameters (**degeneracy**, where parameters trade off along a **ridge**; combinations can be sharp even when individual parameters aren't).

**Profile likelihood:** fix one parameter, re-optimize all others, record the best likelihood, and repeat over a range. A sharp minimum means determined; a flat curve means not. A fitted value sitting exactly on a bound is a warning sign.

---

## 8. Quick reference

| Symbol / term | Meaning |
|---|---|
| `s` | years since the wing's mean latitude crossed 15° |
| `S_A`, `S_B` | window start/end, −6 and +11 yr |
| `S_MESH` | 4001-node quadrature grid |
| `EVENTS[h]` | sorted event times for hemicycle `h` |
| `n`, `T` | event count, window length (17 yr) |
| nat | unit of log-likelihood (natural log) |
