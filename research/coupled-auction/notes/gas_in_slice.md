# The learned gas curve in the vertical slice (GASCURVE=1)

Result: the learned curve helps clearly in winter and mixes in summer. February (2026-02-10, 14 days): inside the 90%
band 0.83 against 0.61, median error +1 against -8 EUR/MWh, scarcity share 0.001 against 0.018. August (2026-08-12, the
curve's out-of-sample window): the band covers more (0.71 against 0.54), but the median error grows from +24 to +33
EUR/MWh, and the band is twice as wide (median 107 against 53 EUR/MWh), so the coverage gain in August is not evidence
of a better fit. In August the slice's cleared gas is 0.7-1.7 GW below observed gas output off-peak and 0.5-0.6 GW above
it in the evening; 37% of quarters clear at or below the curve's lowest slice and 16% at or above its backstop. GASCURVE=1
also makes the coal (zone A) and gas (zone B) peak blocks divisible, which the solver needs (addendum at the end). A
control run with only that change moves August by -0.04 in coverage and +1.5 in median error; net of it the curve gives
+0.21 coverage and +7.8 EUR/MWh median error in August, +0.22 and a median error of +1.0 instead of -7.1 in February. Default behaviour is unchanged (old seeds max diff 0.0, verify.py passes, February baseline JSON byte-identical).

| window | run | price_inside_90 | median band width (p95 - p05) | median_error | scarcity_share | median_error_by_4h (00-04, 04-08, 08-12, 12-16, 16-20, 20-24) |
|---|---|---|---|---|---|---|
| Aug | baseline | 0.539 | 52.8 | +23.9 | 0.000 | 14.8, 21.9, 28.0, 1.2, 32.1, 39.3 |
| Aug | divisible peak blocks only (PEAKDIV=1) | 0.499 | 51.3 | +25.4 | 0.000 | 14.8, 22.0, 31.7, 8.4, 32.1, 39.3 |
| Aug | GASCURVE=1 | 0.713 | 106.9 | +33.2 | 0.000 | 35.2, 31.9, 39.1, 14.8, 33.0, 39.2 |
| Feb | baseline | 0.607 | 85.0 | -8.2 | 0.018 | -13.2, -10.3, -1.1, -9.3, 1.8, -7.7 |
| Feb | divisible peak blocks only (PEAKDIV=1) | 0.609 | 54.4 | -7.1 | 0.014 | -13.0, -10.1, -0.1, -8.0, 4.6, -7.7 |
| Feb | GASCURVE=1 | 0.830 | 55.2 | +1.0 | 0.001 | -5.0, 1.0, 6.4, -1.3, 6.5, -2.4 |

All six runs: COND=1 COSTS=1 FLEETMODE=output, data fleet from de_fleet.py (fleet_aug.json built for this note,
fleet_feb.json as before), 40 draws x 14 days, failed_draws 0. Outputs in the scratchpad's hist/ directory, file tags
_datafleet_costs_cond_{aug,feb}_output{,_peakdiv,_gascurve}; figure hist_window_datafleet_costs_cond_aug_output_gascurve.png.

## Expectations: met or not

- E1 (tractability): not met by the design as first written; met after the addendum. With fill-or-kill peak blocks next
  to the curve's divisible span blocks, 35 of 560 August draws ran past 15 s (cut loops, see the addendum). With divisible
  peak blocks: 0 of 560 failed in each window, every clear in 1 master round with no cuts, 0.45 s per draw (two clears,
  so about 0.23 s per clear); a 14-day run took 1.2 min against 2.5-3 min for the baseline.
- E2 (the slice reproduces the curve, mean absolute deviation at most 0.5 GW and smaller than with lambda = 0): met in
  August (0.40 GW; 0.86 GW with lambda = 0; 4 days x 5 draws; 0.41 against 0.95 on 7 days x 10 draws), borderline in
  February (0.56 GW against 0.71 on 4 days x 5 draws, not met; 0.50 against 0.71 on 7 days x 10 draws, at the threshold). Of
  the deviation about 0.2 GW is the 8-slice discretisation (the price often sits at a slice, accepted anywhere within
  it); the rest is that a mixture of span responses is not exactly a sigmoid of the windowed margin. The windowed term is
  clearly there in both windows.
- E3 (August, |median_error| at most baseline + 2 and price_inside_90 at least baseline - 0.03): not met. Coverage rose
  (0.54 to 0.71) but only because the band doubled in width; |median_error| rose from 23.9 to 33.2.
- E4 (August, within-day sd of the slice median up by at least 20% and closer to the observed): met on the letter, but
  it overshoots: mean within-day sd 40.2 to 65.4 EUR/MWh against 53.1 observed (distance 12.9 to 12.3).
- E5 (August, worst 4-hour block improves by at least 5): not met. 20-24 h: 39.3 to 39.2.
- E6 (February: scarcity at most 0.038, |median_error| at most 10.2, price_inside_90 at least 0.577): met (0.001, 1.0,
  0.830). The divisible-peak-block control gives 0.014, 7.1, 0.609, so the gain is the curve's.
- E7 (default unchanged): met. sbi_sim.py with SEED_OFFSET=0 on 4 datasets against out/sims.npz x[:4]: max |diff| 0.0.
  verify.py: "all checks passed". The February baseline rerun after the last vslice.py edit gives a JSON byte-identical
  to the one from before any edit.

## Where August's gas sits on the curve (measured) and a reading of the figure

Measured with gas_slice_check.py on 7 August days x 10 draws (and the same for February), slice gas against observed
gas output (energy-charts, hourly) by 4-hour block, GW:

| | 00-04 | 04-08 | 08-12 | 12-16 | 16-20 | 20-24 |
|---|---|---|---|---|---|---|
| Aug slice | 3.29 | 4.88 | 2.68 | 1.25 | 4.71 | 6.43 |
| Aug observed | 4.98 | 4.68 | 3.40 | 2.23 | 4.23 | 5.80 |
| Feb slice | 11.40 | 12.50 | 13.35 | 13.07 | 13.96 | 12.59 |
| Feb observed | 10.59 | 11.33 | 12.01 | 11.29 | 12.34 | 11.04 |

In August 37% of quarters clear at or below the price of the curve's lowest slice and 16% at or above the backstop price
(February: 16% and 9%). So in August the slice asks too little of gas at night and midday and too much in the evening:
its residual demand for gas swings more than real gas output, and the price runs off both ends of the curve. In the
figure this shows as flat off-peak plateaus of about 100-120 EUR/MWh (observed 150-170) and evening plateaus of 250-290
(observed about 200). That the cause is the rest of zone A's book - lignite, coal and "other" sized at their 99th output
percentile and offered near cost, which do not follow the margin the way real coal and lignite do - is a hypothesis, not
tested here. In February the slice runs 0.8-1.8 GW more gas than observed all day but stays mostly inside the curve
(floor 9.2 GW, F + A 17.4 GW), and the price errors are small.

## Caveats

- Peak blocks divisible under GASCURVE: a change to coal's (and zone B's gas) representation made only for the solver.
  The control run isolates it: price_inside_90 -0.04 and 12-16 h median error 1.2 to 8.4 in August; in February +0.002,
  but the band narrows from 85 to 54 EUR/MWh (fewer scarcity tails).
- Curve fits: gas_curve_fit.py, dynamic model by hour block, 12 months before each window, data cut before m_win is
  built. August: lambda 0.93 (0.89-0.96), A 8.0 GW, F 0.96 GW, no divergences, max r-hat 1.007 (matches gas_curve2.py's
  dynamic_blocks). February: lambda 0.96, A 8.15 GW, F 9.2 GW, mu_b (18.7, 15.4, 17.1, 20.0, 26.5), s_b (15.1, 15.7, 15.2,
  18.2, 23.3), no divergences, max r-hat 1.008. The floor is the window month's F from its first 9 (Feb) or 11 (Aug) days,
  before the window; not carried forward from another month. The by-block model's lambda (0.93-0.96) is higher than the
  pooled model's 0.84.
- The curve is in realised output (energy-charts gas), offered with its own reference cost 1.85 gas + 0.37 carbon; the
  slice's markup, offered share and outages no longer touch gas. Unit capacity beyond F + A is a backstop at the 99.5%
  curve price, a choice made before the runs and not tuned after.
- Span blocks see only the delivery day: the curve's window at 0-3 h and 21-24 h reaches into neighbouring days.
- February with the curve still has p95 at or above 500 EUR/MWh in 13% of quarters (baseline 36%): the upper tail of
  the band reaches scarcity, though the median does almost never (0.1%).
- Fourteen days per window and one summer window: the August result could look different in other summer weeks.

## Expectations, written before any run

Baseline throughout = the same command without GASCURVE (COND=1 COSTS=1 FLEETMODE=output, data fleet), same seeds, so
theta and the day shocks are paired draw by draw. Known baseline for February (earlier run): price_inside_90 0.607,
median_error -8.2, scarcity_share 0.018. No August baseline existed before this work.

E1. Tractability. With divisible span blocks (minimum acceptance ratio 0) every clear finishes in one master round, no
    draw fails (failed_draws = 0 in both GASCURVE runs), and a clear takes at most 2 s on average (at most twice the
    baseline's time).

E2. The slice reproduces the curve it was given. On check days, the slice's cleared gas output (awards plus what the
    portfolio nets) against the curve evaluated at the slice's own cleared prices with the same posterior draw: mean
    absolute deviation at most 0.5 GW, and smaller with the curve's lambda than with lambda = 0 (the windowed response
    is really there, not just the static one).

E3. August, out of sample for the curve: the curve does not make the price fit worse. |median_error| with the curve at
    most |median_error| without it plus 2 EUR/MWh, and price_inside_90 at least the baseline's minus 0.03.

E4. August: the curve is far more price-elastic than 22 GW at cost +-7%, so the slice's median price moves more within
    the day. The mean within-day standard deviation of the slice's median price rises by at least 20% against the
    baseline, and gets closer to the observed one.

E5. August evening: the 4-hour block with the largest |median_error| in the baseline improves by at least 5 EUR/MWh.

E6. February (winter floor about 8 GW offered price-taking, curve total F + A about 15-16 GW, unit capacity beyond that
    offered at the curve's 99.5% quantile price): scarcity_share at most the baseline's plus 0.02 (at most 0.038),
    |median_error| at most 8.2 + 2, price_inside_90 at least 0.607 - 0.03.

E7. Default path unchanged: the February baseline rerun after the edits gives a JSON identical to the one before them;
    the old seeds of sbi_sim.py give max |x - x_old| = 0.0; verify.py prints "all checks passed".

## Representation (chosen before the runs)

The curve says realised German gas output = F + A sigmoid((m_eff - mu_b)/s_b), with m = price - (1.85 gas + 0.37 carbon),
m_eff = m + lambda (m_win - m) and m_win the centred +-3 h mean margin. In the slice, zone A's "gas" offers are replaced by:

- the floor F as a sell order at -20 EUR/MWh in every quarter (price-taking in practice, like must-run lignite; not at
  -500, which hist_window would treat as the fixed import and drop from scheduled generation);
- A cut into 8 equal quantile slices at q = 1/16, 3/16, ..., 15/16, each priced at 1.85 gas + 0.37 carbon + mu_b(t) +
  s_b(t) logit(q). This is the curve's inverse, so a quarter that clears at margin m takes A sigmoid((m - mu)/s) of
  output, up to the discretisation. The curve's reference cost is used, not the slice's cost() plus markup: mu already
  holds other costs, markup and the start-cost premium as they show in the data, so markup, the offered share pi, forced
  outages and published outages are not applied again to the curve's MW (it was fitted on realised output);
- a share 1 - lambda of each slice offered as period steps (responds to the quarter's own price) and a share lambda as
  span blocks of 5 hours over five staggered tilings of the day (spans start at offsets 0, 1, 2, 3, 4 h; spans are cut at
  the day's edges), a fifth of the lambda share in each tiling. A block is accepted only if the mean price over its span
  reaches its limit (the span mean of the slice's price), so the lambda share responds to a multi-hour mean margin - the
  windowed term of m_eff. Over the five tilings a quarter's margin is averaged with a triangular kernel of half-width
  5 h, whose variance (4 h^2) equals that of the curve's centred +-3 h box. The block price is where start costs enter: a
  unit that starts must recover its start cost over the hours it runs, which is why output follows the window and not the
  hour. The start cost is not identified separately from mu in the fitted curve, so it is not added on top; the block's
  limit is the curve's price for that slice, averaged over the span. (Changed before the runs: a first version with two
  tilings aligned to the curve's hour blocks missed the curve by 0.68 GW on a test day, mostly at the morning and evening
  ramps; the five 5-hour tilings gave 0.55 GW on the same day. 16 slices instead of 8 did not help: 0.70 GW.)
- blocks are divisible (minimum acceptance ratio 0). The aggregate of about 50 market units responds smoothly (that is
  what the sigmoid fits), fill-or-kill blocks per slice would add lumpiness the curve does not show, and 8 x 28 = 224 fill-or-kill
  blocks would risk many no-good-cut rounds in the solver;
- unit capacity beyond F + A (the data fleet less outages, as before) is offered as period steps at the curve's 99.5%
  quantile price (about mu + 5.3 s, 110-150 EUR/MWh over the gas-fired cost): the curve never saw output above F + A at
  the margins it saw, so this capacity is a backstop against artificial scarcity, not part of the fitted response.

Parameters: one posterior draw of the "dynamic_blocks" model per prior draw (lambda, mu_b, s_b, A and the floor F of the
last fitted month), drawn with a separate generator so the theta and day shocks stay those of the baseline. The curve is
refitted per window by gas_curve_fit.py on the 12 months before the window's first day (August: data before 2026-08-12;
February: data before 2026-02-10). The floor is the fitted F of the window's own month, estimated from its first 11 (or 9)
days, which lie before the window: no data from the window, but also not carried forward from another month. Unlike
gas_curve2.py, the data are cut at the window before m_win is built, so no window price leaks into the fit.

## Addendum (2026-11-21, after the first August run): divisible peak blocks

The first GASCURVE August run did not finish in 25 minutes. A scan with a 15 s limit per draw found 35 of 560 draws over
the limit. Cause: zone A's coal peak block is fill-or-kill; in those draws the master accepts it paradoxically, and the
solver's no-good cut excludes only the whole on/off vector of all blocks. With 224 divisible span blocks in the book the
master escapes every cut by switching one span block (with minimum acceptance ratio 0 its on/off variable is free when it
is not accepted; with 1e-4 it moves one block to a tiny acceptance), so the loop runs to its 200-round limit. The fix
kept inside the opt-in: a new optional Info field peak_mar (None = 1, as before), set to 0 for both zones whenever
GASCURVE=1, and its own switch PEAKDIV=1 for the control run. With every block divisible the master problem is a linear
program and valid prices exist in the first round. The proper fix, which is not made here because it changes the
solver, would be a cut over the fill-or-kill blocks' decisions only.
