# io_min, eleven zones: expectations for the two worked MTUs (written before running them, 2026-11-21)

Zones: DE_LU, FR, NL, BE, AT, CZ, PL, DK_1, DK_2, SE_4, NO_2. Time unit: the auction's 15-minute MTU.

Network sources:
- Core flow-based domain from JAO: presolved CNECs with PTDFs and RAM, external constraints, and allocation constraints.
- Core hubs omitted from the model (HR, HU, RO, SI, SK) held at their published net positions.
- Links outside the Core domain at the time (DE–DK2, DE–SE4, DE–NO2, NL–NO2, PL–SE4, and DE–DK1 when its virtual hub is
  inactive) are directional limits.
- Flows among the Nordic zones and between them and the rest of Europe are held at their scheduled values. Results
  are conditional on those flows.

Worked MTUs:
- Congested: 2026-02-11 17:00 UTC, which has several binding CNECs and a DE–FR spread of 94 EUR/MWh.
- Coupled: the first MTU found in early February 2026 with no binding Core constraint and equal Core prices.

Expectations, with thresholds fixed before running:

- W1, domain form. On every presolved CNEC, the published net positions satisfy PTDF·NP ≤ RAM with a maximum
  violation of 5 MW. Binding CNECs (shadow price > 0) have slack of 5 MW or less. If not, look for LTA inclusion or
  the alpha factor before going on.
- W2, signs. The published shadow prices times the PTDF differences reproduce the observed Core zonal price spreads
  within 0.05 EUR/MWh, with one sign convention for every pair.
- W3, accounting. For each zone, the ENTSO-E net position equals the Core net position, plus the ALEGrO and link flows,
  plus the schedules with the rest of Europe, within 10 MW.
- W4, Jacobian tests:
  - for one price area, det(Pᵀ diag(S′) P) = ΣS′ up to a constant fixed by P;
  - for full separation, it equals ΠS′;
  - rank(P) + free net-position dimensions = 11 in both worked MTUs.
- W5, re-clear. With shocks recovered at any θ, re-clearing the eleven zones on the published domain reproduces the
  observed prices within 0.5 EUR/MWh and the zonal net positions within 20 MW. A miss shows a missing piece of the
  domain, not a wrong θ.

## External slope check, eleven-zone (written 2026-11-24, before the run)

Estimand. The model's dp_DE/dD_DE: re-clear with German demand +1 GW, holding neighbour demand, rest-of-Europe
exchanges and inferred rows at their recovered or observed values, on the replay domain. It is evaluated on the random
fit-sample MTUs, which represent the period, not the outage sample. It is weighted the same way as the two-zone check
(`cf.py`: squared residual German demand of the hour) and compared with the same empirical net slope: 2.38
[2.17, 2.60] overall, by daily-price tercile low 2.61, middle 2.10, high 2.37. Scarcity MTUs, those with a
re-cleared price above 4000, are masked and counted. The check does not feed back into the model.

Expectations and thresholds:
- E1, direction: the model is still steeper than the empirical slope. Rest-of-Europe exchanges and inferred rows are
  held, so imports can respond less than they did in the data. Met if empirical / model < 1.
- E2, improvement on the benchmark: the ratio is closer to 1 than the two-zone 0.67, because ten neighbour curves
  and the flow-based domain now carry part of the 1 GW. Met if the ratio is above 0.75. Material agreement (in
  [0.8, 1.25]) is not expected.
- E3, pattern: the gap is largest in the high-price tercile, as in the two-zone check, and larger in AHC and generic
  MTUs, where more flows are held.
- E4, sanity: on the outage MTUs, the demand-shock ΔpDE is within 25% of the outage ΔpDE at the median (a 1 GW
  outage and 1 GW of extra demand differ only through the gas floor cut and the fuel split). A larger gap points to a
  bug.

## Slope check with the cold-start interior-point solver, current curves (written 2026-11-24, before the run)

Same estimand, MTUs and draws as the first slope check. The only change is the solver: `solve11.solve_ip` with a 5 MW
balance tolerance, in place of the warm-start active set. That solver converged in 92.6% of MTUs, against 99.8% for
the interior-point solver on a 458-MTU test.
- S1: convergence of at least 98% of MTUs.
- S2: on MTUs where both solvers converged, the slopes agree within 0.05 EUR/MWh per GW at the 99th percentile.
- S3, direction: the solves that newly converge are mostly steep cases (curve ends, generic constraints). The
  weighted ratio therefore falls below 0.68. The size is not predicted.
- S4: baseline replay error in DE below 0.1 EUR/MWh in 99% of MTUs.

This is the reference against which the curve-end change is judged; the curve change is not tuned to it.

Result (2026-11-24): S1 met (98.3%); S2 met (99th percentile difference 0.016, 0.45% of MTUs above 0.05); S3 met (the
weighted ratio falls from 0.68 to 0.59, and the 1,035 newly converged draw-MTUs have a mean slope of 12.3 and a median
of 3.3); S4 met (99th-percentile replay error 0.027). By band: low 0.61, middle 0.69, high 0.48; generic MTUs 0.42.
The reference for the curve-end change is therefore 0.59, not 0.68.

## Curve-end recovery screen (written 2026-11-24, before the run)

Data: 600 training-window MTUs simulated by `ends_sim.py`, with the old-curve posterior medians for everything else.
Two truths:
- **Case A (ends present, off the prior centres).** m_lo = −35, s_lo = 4, m_hi = 150, s_hi = 100,
  ρ = (0.15, 0.8, 0.2), k_lo,DE = 8, k_hi,DE = 4, k_lo,nb = 0.5, k_hi,nb = 0.35. Each is 1.3–1.8 prior sd from its
  prior centre.
- **Case N (null).** No end segments: all k = 0 and ρ = 1, so nothing sheds.

MAP fits (`map11.py`) start from two dispersed points: the prior centres, and a far point (m_lo = −60, s_lo = 25,
m_hi = 350, s_hi = 20, ρ = 0.9, k_lo,DE = 0.5, k_hi,DE = 0.2, k_lo,nb = 0.02, k_hi,nb = 0.02).

Thresholds:
- R1 (case A). From both starts, the MAP end parameters that are identified at the simulated price range are
  recovered: k_lo,DE and k_hi,DE within ±35%, m_lo and m_hi within ±15 EUR/MWh.
- R2 (case N, no invented slope). MAP k_lo,DE < 1.5 GW, k_hi,DE < 1.0 GW, k_lo,nb < 0.1 and k_hi,nb < 0.1. In other
  words, the Jacobian term does not create end segments that are not there.
- R3 (stability). The two starts agree on case A within the R1 tolerances.

If R2 fails, the specification is not used as is: the end segments would come from the likelihood's reward for slope,
not from the data. If R1 or R3 fails, a NUTS recovery is requested before any real fit.

Result (2026-11-24, MAP screen):

| Case, start | k_lo,DE | k_hi,DE | m_lo | m_hi | k_lo,nb | k_hi,nb | ρ (L, C, G) |
|---|---|---|---|---|---|---|---|
| A truth | 8 | 4 | −35 | 150 | 0.5 | 0.35 | 0.15, 0.8, 0.2 |
| A, centre | 7.88 | 3.44 | −36.3 | 234 | 0.50 | 0.06 | 0.40, 0.52, 0.41 |
| A, far | 8.02 | 0.72 | −36.3 | 313 | 0.50 | 0.12 | 0.41, 0.52, 0.41 |
| N truth | 0 | 0 | — | — | 0 | 0 | 1, 1, 1 |
| N, centre | 3.79 | 1.87 | −6.0 | 268 | 0.02 | 0.11 | 0.48, 0.61, 0.75 |
| N, far | 4.02 | 2.21 | −19.3 | 314 | 0.02 | 0.12 | 0.49, 0.58, 0.75 |

Verdicts:
- **R1: partly met.** The low end is recovered (k_lo,DE, m_lo, k_lo,nb). The scarcity segment is not (m_hi off by
  84–163, k_hi,nb off by a factor of 3–6). The floor-shedding shares ρ are not recovered (not in R1, reported).
- **R2: not met.** On null data the fit invents about 4 GW of German curtailment and about 2 GW of scarcity supply,
  from both starts. The cause is open. Candidates:
  - the simulator drops MTUs whose solve fails or whose price leaves [−500, 3000] (24 in the null case), which
    removes extreme prices the fit may then explain with end slope;
  - the Beta(2, 2) prior on ρ excludes "no shedding";
  - the truth used in the comparison is not an exact parameter vector, and that comparison returned −inf at both
    points.
  These are to be checked before any conclusion about the likelihood.
- **R3: not met for scarcity** (k_hi,DE 3.44 vs 0.72 across starts). Met for the low end.

Caveats: the MAP losses were still falling by about 2 per 500 steps, and the neighbour truth in the fit
parameterisation is approximate (medians of hyperparameters and offsets, not of the curves).

By the rule written above, this specification is not used as is.

Diagnostics on the R2 failure (2026-11-24):
- **Dropped MTUs.** 23 of the 24 dropped in the null are failed solves, not prices out of range. With bounded supply,
  some heavy-tailed shocks have no equilibrium; a real market would clear at the price cap. The simulation therefore
  truncates the extreme shocks the t-distribution generates.
- **Regime reading.** In 0.5–1% of simulated MTUs the prices are not in the V read from the solver's multipliers
  (residual up to 590). The fit now excludes MTUs with a residual above 0.05, as build11 does for real data.
  Excluding them did not remove the invented ends: null MAP from an exact posterior draw gave k_lo,DE 3.0 and
  k_hi,DE 1.7, with the loss converged.
- **Likelihood profile at the exact truth** (draw 100, all other parameters held). Adding German curtailment raises
  the price-system likelihood by at most +4.0 nats (k_lo,DE 3, m_lo 3.5, s_lo 7.6). That is at the edge of chance
  for three free parameters, and it comes from 10 simulated MTUs at p ≤ 0. Scarcity gains nothing (+0.02). The null
  screen therefore had little power; it does not show a biased likelihood.
- **Predictive misfit of the old curves.** Simulated from the old-curve posterior with its own shocks, the share of
  MTUs at p ≤ 0 is 1.7–2.9% and above €200 1.5–1.9%. The real training sample has 6.2% and 3.0%. The old curves
  under-produce negative and scarcity prices: a posterior-predictive failure where the curve-end change is aimed.
- **Gain grows with the data, so it is systematic.** On 1,780 null MTUs (52 at p ≤ 0) the MAP again invents
  k_lo,DE 3.2 and k_hi,DE 2.9. At the truth, the likelihood gain is +10.5 nats for the low end (+4.0 on 578 MTUs) and
  +2.8 for scarcity (+0.02).
- **Truncation explains part of it.** With shocks scaled to 0.6, fewer solves fail. At 3 GW the low-end gain then
  turns strongly negative (−11.6), but a gain of +4.8 remains at 1 GW, and scarcity gains +1.2. The remaining failed
  solves (25 of 1,839) sit at ordinary prices (€74–91), so they are solver failures, not missing equilibria.
- **Status.** The cause is partly the simulation's dropped draws and partly unexplained. The specification is still
  not used.

## Slope check on the training window, current curves (written 2026-11-24, before the run)

Model: the training-window refit of the current curves (`post_train_old.npz`), 8 draws evenly spaced. MTUs: the
training fit-sample records. Empirical slope: the cf.py recipe on training hours only. Solver: interior point.
This is the reference for the curve-end specification, so no threshold applies. Expected: a weighted ratio near the
full-period 0.59 (0.5–0.7), with the high-price band lowest.

Result (2026-11-24). Training refit `post_train_old.npz`: 800 MTUs, 2 chains, 400+400, 41 minutes, no divergences.
The German block agrees across chains. Neighbour curves reach r-hat ≤ 1.18 with close chain medians (FR, SE4, NO2
slowest); the raw offsets reach 1.92, but those are not separately identified.

Slope check on the training window: empirical 2.15 [1.90, 2.39]; model weighted 3.15 (draws 2.99–3.33); ratio
**0.69**. By band: low 0.75, middle 0.70, high 0.60. Hybrid coupling 0.77; generic 0.33. Converged 97.2%. Met the
expectation (0.5–0.7, high band lowest). This is the reference the curve-end specification must move into
[0.8, 1.25].
- **Dropped draws are structural.** 22 of the 25 failed solves at shock scale 0.6 have no conic solution at all. The
  same 25 MTUs fail at scale 0.4, so their domains cannot be met for any shock. This is selection on MTUs, not on
  shocks.
- **The invented slope depends on shock size.** Likelihood gain at the truth, by shock scale:

  | Shock scale | k_lo,DE = 3 GW | k_lo,DE = 1 GW | k_hi,DE = 1 GW |
  |---|---|---|---|
  | 1.0 | +10.5 | +6.9 | +1.8 |
  | 0.6 | −5.0 to −11.6 | +4.8 | +1.2 |
  | 0.4 | −29 to −42 | −0.8 to +2.0 | +1.1 |

  The bias appears with realistic heavy-tailed shocks. Its cause is not found.

## Option A: floor shedding only (written 2026-11-24, before the screen)

Specification (fit11 ENDS=2): each German fuel's floor sheds a share shed_f ∈ [0, 1] at negative prices, around a
shared centre m_lo and scale s_lo. The prior is Beta(1, 3), whose mode is 0, so "no shedding" is allowed. There are
no free curtailment or scarcity terms. Shedding is observed directly in fuel outputs at low prices.

The screen uses MAP fits from two starts (centre: shed 0.2, m_lo −10; far: shed 0.7, m_lo −40, s_lo 20), shock scale
1, an exact posterior draw (100), extremes kept, 1,800 training-window MTUs. Two truths:
- **A2 (shedding present).** shed = (0.85, 0.20, 0.80), m_lo = −35, s_lo = 4.
- **N2 (null).** shed = 0.

Thresholds:
- R1′ (A2): each shed_f within ±0.2 of truth and m_lo within ±15, from both starts.
- R2′ (N2): each shed_f < 0.15 from both starts (no invented shedding).
- R3′: the two starts agree within the R1′ tolerances.

If R2′ fails, option A is not used either.

Result (MAP screen, option A):

| Case, start | shed L | shed C | shed G | m_lo |
|---|---|---|---|---|
| A2 truth | 0.85 | 0.20 | 0.80 | −35 |
| A2, centre / far | 0.79 / 0.79 | 0.17 / 0.17 | 0.80 / 0.80 | −34.3 / −34.3 |
| N2 truth | 0 | 0 | 0 | — |
| N2, centre / far | 0.62 / 0.62 | 0.00 / 0.00 | 0.32 / 0.32 | +3.1 / +2.7 |

Verdicts:
- **R1′ met** and **R3′ met**: shedding is recovered when present, identically from both starts.
- **R2′ not met**: on null data the fit invents lignite and gas shedding centred near +3 EUR/MWh. By the rule above,
  option A is not used as is.

Decomposition at the exact truth: the invented shedding gains +8.4 nats in the Jacobian term and loses −5.3 in the
shock density, net +3.1 (+0.5 from fuel outputs). The gain is spread over the 37 MTUs priced in (−20, 30], with no
outliers. A net gain of about 3.6 for three parameters is the size chance produces, so the screen lacks power at 37
informative MTUs.

### Follow-up diagnostic (written before running; a new test, not a rerun of R2′)

Null data simulated on 1,000 training MTUs selected **by inputs**: half from hours whose observed price was at most
€20 (high renewables), half at random. Selection is on the MTU and its inputs; outcomes are re-simulated, so this is
valid. Same MAP screen and starts.
- D1: each shed_f < 0.15 from both starts.
- D2 (case A2 on the same MTUs): R1′ tolerances.

If D1 is met, the R2′ failure was a power problem, and option A goes to a training fit (with approval). If D1 fails,
option A is dropped and the curve ends stay as they are.

Result (follow-up diagnostic):

| Case, start | shed L | shed C | shed G | m_lo |
|---|---|---|---|---|
| D1 null, centre / far | 0.65 / 0.62 | 0.00 / 0.01 | 0.05 / 0.08 | +8.9 / +4.5 |
| D2 truth | 0.85 | 0.20 | 0.80 | −35 |
| D2, centre / far | 0.59 / 0.59 | 0.00 / 0.01 | 0.75 / 0.75 | −36.8 / −36.8 |

Verdicts:
- **D1 not met for lignite**; gas and coal stay near zero.
- **D2 not met for lignite** (0.59 vs 0.85); gas and coal within tolerance.

Implied supply: on null data, the MAP moves along the lignite ridge (F_L 1.2 → 2.8–3.0 GW, σ_L 0.44 → 0.30) and sheds
62–65% of the larger floor. Most of this cancels, but German supply ends up 0.5–0.8 GW below truth at p ≤ 0 and up to
0.2 GW above it at 20–40 EUR/MWh. Between −20 and +20 EUR/MWh it rises about 1 GW more than the truth, roughly 70%
more slope where there is none. That would lower dp/dD exactly where the slope check is weakest, a bias that would
pass for an improvement.

**Decision (by the rule above): option A is dropped. The curve ends stay as they are.** The lignite floor/dispersion
ridge is the underlying problem: any low-price flexibility added to German supply is absorbed into it.

## Fuel-cost loading fix (written 2026-11-25, before any fit)

Specification (fit11 BPOS=1): each neighbour's fuel-cost loading b_z = exp(m_lbb + t_lbb z_z), positive by
construction and centred at 1 (full pass-through). Everything else as in the current curves. Chains start with every
b_z = 1 and each zone's threshold a + b·mean(L) unchanged.

Inputs-only diagnostic on the current training fit (`extrap11.py`): between the training-median (108) and the
holdout-median (182) fuel-cost index, every neighbour's supply at 150 EUR/MWh already falls, by 74 GW in total
(France −16, NO2 −25), because the curves' spread scales with the index. The holdout misfit (France's recovered shock
+15.5 GW against +2.8 in training) is therefore one of degree, not direction. France's negative loading (−0.37) makes
its threshold fall as costs rise, whereas a positive loading makes it rise.

Expectations:
- F1 (inputs only): France's loading is positive, and its supply change at 150 EUR/MWh between the two index medians
  is more negative than −16 GW.
- F2 (training slope check): the weighted ratio stays within 0.6–0.8. The fix is not aimed at the slope.
- F3 (second holdout scoring, labelled as such; the first score stays the reference): the unconditional eleven-zone
  forecast's evening-peak bias shrinks in magnitude from −43. The previous-day-shock forecast's price MAE is no
  worse than 20.5 by more than its bootstrap interval.
- F4: the holdout's lagged recovered French shock moves towards its training mean (+15.5 → closer to +2.8).

F3 and F4 read holdout outcomes and cannot be clean tests: the specification was chosen after seeing the first
holdout score. F1 and F2 can be.

Pre-run check (2026-11-25, training data only): the refit was not run.
- France's modelled supply is the same at the holdout's typical (price, index) as at training's (62.7 vs 63.0 GW).
- France's recovered shock is seasonal within training: +8.2 GW in October 2025 at a low index, −9.1 in February,
  +18.4 in August.
- The evidence points to French need or availability, not fuel costs. Forcing the loading positive could cut the
  holdout error for the wrong reason.
- The ENTSO-E French unavailability notices (bt_unavail_gen_FR) cannot be used as is: summed overlapping notices
  exceed the fleet (up to 238 GW), and none are counted before 2026.
