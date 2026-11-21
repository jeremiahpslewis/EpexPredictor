# A year-round gas floor and a recent lignite level in the slice (FLOOR2, LIGLEVEL)

## Result

Sizing lignite at its recent in-the-money level (LIGLEVEL) is the half of the diagnosis that the slice confirms. In August it cuts the median error from +33.2 to +22.3 EUR/MWh and raises price_inside_90 from 0.71 to 0.78. Gas output moves to the observed level: the mean is 4.41 against 4.47 GW observed, where it was 3.43 before. The gain comes with a band a third wider (143 against 107 EUR/MWh) and an evening block that now overshoots (20-24 h: from +39 to -33). The year-round gas floor (FLOOR2) adds about 0.3 GW of gas in August, as intended, but makes the price slightly worse (+36.1). It is price-taking supply, so it can only push prices down, and the August nights were already too low. With both switches August lands at +24.7, between the two. In February both switches move the slice away from a price level that was right (+1.0): -5.8 with FLOOR2, -3.9 with LIGLEVEL and -11.9 with both. LIGLEVEL also brings back false scarcity, which the medians hide. scarcity_share is 0.017 with LIGLEVEL and 0.016 with both, nearly the 0.018 that GASCURVE had removed. The mean absolute error of the slice median rises from 17.7 to 80.7 and 82.7 EUR/MWh, the slice median averages 157 and 168 against 93 observed, and one draw failed with both switches. Each switch, however, brings its own technology onto the observed level. With FLOOR2 February gas is 10.49 against 10.76 GW, where it was 12.19. With LIGLEVEL February lignite is 9.66 against 9.74 GW, where it was 11.11. One reading of the technology table, an inference rather than a test, is that February's earlier price fit rested in part on offsetting errors: too much gas and lignite, too little coal. The best August variant is LIGLEVEL alone (figure hist_window_aug_liglevel.png). The criterion is |median_error| and price_inside_90, where it is best (+22.3, 0.777). On mean absolute error of the median, both switches together are marginally lower (41.2 against 41.3; current 42.3), and their 20-24 h block is milder (-11 against -33). Neither is a recommended default: the evening overshoots, the band is wider, and February suffers.

| window | variant | price_inside_90 | median band width (p95 - p05) | median_error | scarcity_share | median_error by 4 h (00-04, 04-08, 08-12, 12-16, 16-20, 20-24) |
|---|---|---|---|---|---|---|
| Aug | current | 0.713 | 106.9 | +33.2 | 0.000 | 35.2, 31.9, 39.1, 14.8, 33.0, 39.2 |
| Aug | +FLOOR2 | 0.690 | 103.0 | +36.1 | 0.000 | 36.9, 33.0, 42.4, 14.7, 36.7, 44.9 |
| Aug | +LIGLEVEL | 0.777 | 142.6 | +22.3 | 0.000 | 26.9, 22.2, 32.4, 8.9, 20.3, -33.3 |
| Aug | +both | 0.759 | 141.6 | +24.7 | 0.000 | 29.6, 24.0, 34.9, 13.6, 24.0, -11.3 |
| Feb | current | 0.830 | 55.2 | +1.0 | 0.001 | -5.0, 1.0, 6.4, -1.3, 6.5, -2.4 |
| Feb | +FLOOR2 | 0.818 | 81.1 | -5.8 | 0.001 | -7.1, -2.5, -3.9, -6.1, -4.1, -8.6 |
| Feb | +LIGLEVEL | 0.798 | 79.2 | -3.9 | 0.017 | -7.9, -2.5, 0.7, -4.7, -0.5, -7.3 |
| Feb | +both | 0.748 | 86.0 | -11.9 | 0.016 | -10.8, -8.8, -15.1, -10.5, -13.1, -13.6 |

All eight runs use 40 draws for each of 14 days. There were no failed draws, except one in February with both switches. The current rows reproduce notes/gas_in_slice.md exactly. Band width is the median over quarters of p95 - p05.

The slice's output against observed output (energy-charts, hourly) is shown below, in GW, as the mean over the window of the per-quarter median over draws, with awards plus netted MW, by 4-hour block (Europe/Berlin), for all 14 days and 40 draws (TECHOUT=1). gas_slice_check.py, used in notes/gas_in_slice.md, averages over draws instead, on 7 days and 10 draws, so its numbers differ. For August gas, run on all 14 days and 4 draws, the mean over draws is higher than the median at night. For the current variant it is 3.36, 4.42, 2.62, 1.23, 4.26 and 6.22 GW by block. For LIGLEVEL it is 4.60, 5.70, 3.24, 1.41, 5.55 and 7.68 GW, about 4.70 on average against 4.47 observed, so on means LIGLEVEL slightly overshoots the gas total rather than matching it.

| | 00-04 | 04-08 | 08-12 | 12-16 | 16-20 | 20-24 | mean |
|---|---|---|---|---|---|---|---|
| Aug gas observed | 5.16 | 5.02 | 3.69 | 2.65 | 4.45 | 5.85 | 4.47 |
| Aug gas current | 2.63 | 4.18 | 2.48 | 1.17 | 4.27 | 5.86 | 3.43 |
| Aug gas +FLOOR2 | 2.92 | 4.39 | 2.83 | 1.61 | 4.42 | 6.06 | 3.71 |
| Aug gas +LIGLEVEL | 3.85 | 5.39 | 3.08 | 1.30 | 5.58 | 7.25 | 4.41 |
| Aug gas +both | 4.00 | 5.59 | 3.40 | 1.73 | 5.73 | 7.40 | 4.64 |
| Aug lignite observed | 6.91 | 6.93 | 5.21 | 4.16 | 6.43 | 7.18 | 6.14 |
| Aug lignite current | 10.13 | 10.16 | 7.74 | 6.49 | 8.82 | 10.50 | 8.97 |
| Aug lignite +FLOOR2 | 10.04 | 10.08 | 7.65 | 6.40 | 8.76 | 10.46 | 8.90 |
| Aug lignite +LIGLEVEL | 8.35 | 8.35 | 6.44 | 5.45 | 7.28 | 8.53 | 7.40 |
| Aug lignite +both | 8.32 | 8.31 | 6.38 | 5.38 | 7.22 | 8.52 | 7.36 |
| Feb gas observed | 9.81 | 10.47 | 11.16 | 10.62 | 11.69 | 10.79 | 10.76 |
| Feb gas current | 11.07 | 11.90 | 12.56 | 12.34 | 13.18 | 12.10 | 12.19 |
| Feb gas +FLOOR2 | 9.14 | 10.08 | 11.06 | 10.59 | 11.66 | 10.39 | 10.49 |
| Feb gas +LIGLEVEL | 11.37 | 12.39 | 13.21 | 12.94 | 13.89 | 12.72 | 12.75 |
| Feb gas +both | 9.53 | 10.73 | 11.96 | 11.37 | 12.65 | 11.10 | 11.22 |
| Feb lignite observed | 9.08 | 9.56 | 9.96 | 9.72 | 10.40 | 9.74 | 9.74 |
| Feb lignite current | 10.40 | 10.74 | 11.41 | 10.98 | 11.94 | 11.21 | 11.11 |
| Feb lignite +FLOOR2 | 10.78 | 11.12 | 11.83 | 11.28 | 12.39 | 11.74 | 11.52 |
| Feb lignite +LIGLEVEL | 9.08 | 9.37 | 9.89 | 9.52 | 10.33 | 9.74 | 9.66 |
| Feb lignite +both | 9.43 | 9.65 | 10.20 | 9.72 | 10.68 | 10.18 | 9.98 |

Coal, for reference: the slice runs less coal than observed in every variant. In August it is 0.5-4.0 GW against 2.2-4.3 GW observed, and in February 0.8-3.4 GW against 3.7-4.7 GW. Cutting lignite moves it about 0.5 GW closer. This is the next gap: in February it is most of what the two switches expose.

In August, LIGLEVEL brings the gas total to about the observed level (median 4.41, mean about 4.70, against 4.47) but not its shape. On medians, night gas is still 1.2-1.3 GW short (00-04), midday 1.3 GW short, and the evening 1.1-1.5 GW over (16-24 h); on means the night gap is 0.6 GW and the evening excess 1.1-1.8 GW. The slice's residual demand for gas still swings more within the day than real gas output does. In the figure, the evening plateaus of 250-290 EUR/MWh against about 200 observed show this, and so do the nights at about 100-130 against about 150. The wider band is measured in the 16-20 block. The median p95 there rises from 183 to 276 EUR/MWh, about the backstop price, while the other blocks change by at most 12. The share of quarters at or above the backstop rises from 0.14 to 0.20 (gas_slice_check.py, 14 days by 4 draws).

## Expectations: met or not

E1 is met. FLOOR2 alone gives a median error of +36.1, inside +30 to +38, and price_inside_90 of 0.690, within 0.04 of 0.713. Gas in the 00-04 block rises by 0.29 GW.

E2 is not met. Five of its seven parts hold: the median error is +22.3, price_inside_90 is 0.777, scarcity is 0.000, lignite falls by 1.57 GW and gas in the 00-04 block rises by 1.22 GW. Two parts fail. The 00-04 block's median error fell by 8.3 EUR/MWh, short of 10, and the band widened to 142.6, above 107.

E3 is met. With both switches the median error is +24.7, at most +27, and the floor gives back 2.4 EUR/MWh of LIGLEVEL's gain, at most 3.

E4 is not met. With LIGLEVEL the slice's lignite exceeds observed lignite by 1.26 GW (7.40 against 6.14), and by 1.22 GW with both switches, not by the 1.5 GW predicted. It is still too high in every block, by 1.0-1.4 GW off-peak.

E5 is not met. February gets worse with every switch, but less than predicted. FLOOR2 gives -5.8 and 0.818, against at most -8 and at most 0.78; its scarcity of 0.001 holds. LIGLEVEL gives -3.9, against at most -8. Both switches give -11.9 and 0.748, against at most -15 and at most 0.70; their scarcity of 0.016, above 0.01, holds.

E6 is met. After the last code edit, `SEED_OFFSET=0 OUTNAME=r.npz sbi_sim.py <tmp> 4` matches out/sims.npz x[:4] with max diff 0.0, and verify.py prints "all checks passed". The August GASCURVE run without the new switches gives a JSON byte-identical to the one saved before any edit (cmp). The current-variant runs with TECHOUT=1 also give JSONs byte-identical to the earlier ones, in both windows.

## Caveats

FLOOR2 swaps the floor under a curve whose A, μ_b and s_b were fitted jointly with a different, monthly floor. In February the floor drops by about 2.3 GW at the same A, so F + A falls from 17.4 to about 15 GW, and more gas is priced at the backstop. A curve refitted with the B + f1 HDD floor inside it would be the consistent version. The floor is a point value per day, so the posterior spread of F is lost. HDD_d comes from archived short-lead weather, not a forecast issued the day before. February's B comes from summer 2025 and misses the 2026 rise. The straight HDD line is known to overshoot in shoulder months (notes/gas_floor.md).

The lignite statistic targets the median in-the-money output. The slice then offers it at near cost in every quarter, so lignite in the slice still does not follow the margin within the day as real lignite does: that would need the lignite curve of notes/lig_coal_curves.md in the book, as was done for gas. The August drop in lignite inside the window is not visible to any estimator that uses only earlier data, so 1.2 GW of over-supply remains.

These are two 14-day windows with one summer window, and the comparisons are paired by seed. The disclosure above applies: some in-window statistics were seen before the expectations were written.

Files: hist_window.py (FLOOR2, LIGLEVEL and TECHOUT, all opt-in), vslice.py (new optional Info field tech_orders) and hist_window_aug_liglevel.png. Run outputs are in the scratchpad's fl/ directory: runs/, ident/, runs.sh, summarise.py and lig_backtest.py.

## Expectations (written 2026-11-21, after the estimators were fixed and before any run of the slice)

All runs use COND=1 COSTS=1 FLEETMODE=output GASCURVE=1 with the window's FLEET and GASCURVE_DRAWS, 40 draws by 14 days and the same seeds, so the variants are paired draw by draw. The "current" values are those of notes/gas_in_slice.md: in August, price_inside_90 0.713, median band width 106.9, median_error +33.2 and scarcity 0.000; in February, 0.830, 55.2, +1.0 and 0.001. Median error is observed minus the slice's median, so a positive value means the slice is too low.

The two estimators, chosen before the runs, give these numbers. FLOOR2 sets the floor to 1.42 GW on every August day except one, where it is 1.69 GW, against the curve's own floor of 0.96 GW. In February it sets 4.5 to 8.5 GW, 6.9 on average, against 9.2 GW. LIGLEVEL sizes lignite at 10.19 GW in August, against 12.62, and 13.38 GW in February, against 15.86, so the expected lignite offer falls by about 2.0 GW in August and 2.1 GW in February.

The mechanisms follow from the book. The floor is price-taking supply at -20 EUR/MWh. Raising it adds gas MW in quarters where the price sits below the curve's lowest slice and pushes out coal or lignite there. It cannot raise a price, so on its own it cannot cure August's too-low nights. Cutting lignite leaves more residual demand for coal and gas, which should lift the night plateau of about 110 EUR/MWh, set by coal and lignite near cost, towards the gas curve. In February both switches take supply out of a book whose price was already about right: the floor drops by about 2.3 GW at the same A, μ and s, and lignite by 2.1 GW. Both should make February's prices too high.

E1. August, FLOOR2 alone does not fix the price. The median error stays between +30 and +38, price_inside_90 stays within 0.04 of 0.713, and the slice's gas rises by at least 0.2 GW in the 00-04 block.

E2. August, LIGLEVEL alone moves the price the right way. The median error falls to +25 or less, the 00-04 block's median error falls by at least 10 EUR/MWh from +35.2, price_inside_90 is at least 0.68, the band is no wider than 107, and scarcity_share stays at most 0.01. The slice's mean lignite output falls by at least 1.5 GW, and its 00-04 gas rises by at least 0.5 GW.

E3. August, with both switches the median error is at most +27. The floor gives back a little of LIGLEVEL's gain, but not more than 3 EUR/MWh.

E4. August lignite is still too high with LIGLEVEL. The slice's mean lignite output exceeds the observed mean (6.1 GW over the window) by at least 1.5 GW, because the drop below the curves came inside the window, and no estimator using only earlier data sees it.

E5. February gets worse with each switch. With FLOOR2 the median error is -8 or lower, price_inside_90 is 0.78 or lower, and scarcity_share is at most 0.03. With LIGLEVEL the median error is -8 or lower. With both it is -15 or lower, price_inside_90 is 0.70 or lower, and scarcity_share is above 0.01.

E6. The default is unchanged. Run after the last edit, `sbi_sim.py` with SEED_OFFSET=0 matches out/sims.npz x[:4] with max diff 0.0, verify.py prints "all checks passed", and the August GASCURVE run without the new switches reproduces its earlier JSON byte for byte.

Disclosure: while choosing the estimators, I printed some in-window statistics, which are August lignite and gas at negative prices and February lignite. The estimators were not tuned to them. B follows one rule in both windows, and the lignite statistic was chosen by a backtest at 35 origins before 10 February 2026 (fl/lig_backtest.py in the scratchpad). Against the next 14 days' median lignite in hours above cost + 40, the 28-day median scored MAE 1.05 GW with bias -0.13, against 1.86 and +1.77 for the current 3-month p99. The 28-day 90th percentile scored 1.30 and +1.09. The 14-day median scored 0.85 but had no value at 2 origins.

## Estimators

FLOOR2: F_d = B + f1 HDD_d, which replaces the GASCURVE floor of each posterior draw. A, λ, μ_b and s_b are kept. HDD_d is max(0, 15 - the daily mean of the six German land points of the weather store), the same as in gas_floor.py. f1 = 0.417 GW per HDD comes from gas_floor.py's into-winter fit, whose data end on 31 October 2025. That is the only HDD fit that ends before both windows, and it is used in both. In August HDD is about 0, so f1 hardly matters there. B is the median of gas output less f1 HDD over the last 100 negative-price hours, before Berlin midnight of the window's first day, on days without heating (HDD at most 0.5). This measures the year-round price-insensitive floor of process heat, hot water and small CHP, in hours where neither the price nor heating calls gas. A 60-day window would leave February with 4 negative-price hours. Taking the last 100 negative hours without the heating condition reaches into autumn, where the straight HDD line overshoots, and gives B = -0.6 GW. With the heating condition, B is 1.42 GW for August, from 2 July to 11 August 2026, and 1.10 GW for February, from 4 July to 21 September 2025, so February's B misses the 2026 rise of the summer floor.

LIGLEVEL: lignite capacity is the median of hourly lignite output over the 28 days before the window, in hours priced above lignite cost + 40 EUR/MWh, divided by 0.84. Lignite cost is 1.1 times the daily EUA plus 8. The divisor 0.84 is the one FLEETMODE=output uses: the prior's mean offered share times one minus the mean forced-outage share. The slice's expected lignite offer therefore equals lignite's typical output in hours when it is clearly in the money. A high quantile was considered and rejected. Over the last 28 days the 90th to 95th percentile is about the same as the 3-month 99th (10.5 against 10.6 GW in August). It would set the offer at the top of recent output, the same over-sizing as now, and it scored worse in the backtest.
