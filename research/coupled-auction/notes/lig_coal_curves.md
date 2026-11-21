# Lignite and hard coal supply curves with a drifting location

## Result

Letting the location μ drift does not improve out-of-sample prediction of lignite or hard coal output, either at the onset of the 2026 departure or inside it. What helps is letting the level of the curve move. Four variants were scored on each 14-day test window, all using the same commitment window and noise model. The fixed curve, in which nothing walks, was added as a baseline after the tests had been run. The F walk lets the floor move by 4-week block (the registered static model). The A walk keeps the floor constant and lets the top of the curve move. The A plus μ walk adds the drifting location to the A walk. Test MAE in GW, in that order:

- lignite, 12–25 August 2026: 2.51, 1.98, 1.77, 1.80;
- lignite, 15–28 October 2026: 1.08, 1.09, 0.96, 1.02;
- hard coal, August: 0.54, 0.68, 0.75, 0.67;
- hard coal, October: 1.16, 0.66, 0.58, 0.61.

The registered drifting model (F plus μ walk) scored 2.03, 1.22, 0.67 and 0.70. Adding the μ walk to the A walk changed MAE by +2%, +6%, −11% and +5%. Against the fixed curve, the A plus μ walk is 28% better for lignite in August, 6% better for lignite in October and 47% better for hard coal in October, but 24% worse for hard coal in August. The A walk alone does as well or better in three of the four cases, so the gain comes from the moving level, not the moving location. In October every model with a moving level over-predicts, by about 0.6 GW for lignite and 0.5 GW for coal. In August the lignite models over-predict by 1.5 GW (2.4 GW for the fixed curve) and cover only 54–58% of hours. That miss is abrupt: lignite output fell 2–3 GW below every forecast from about 13 August, which fits the August onset in the earlier notes.

The location shift is real for lignite. It is part of a gradual climb, steepening from August 2026, rather than a step. In the all-data fit (descriptive, and using data after the test windows), lignite μ ranges between −21 and −4 EUR/MWh from late April 2024 to early March 2026 and between +2 and +4 from April to June 2026. It is −6 in the block starting 20 July, +4 from 17 August, and +10 and +9 in the blocks starting 14 September and 12 October 2026 (90% bands about ±2.5). Year on year, block for block, μ was already 5–12 EUR/MWh higher through summer 2025 than in summer 2024. In 2026 it is 6–15 higher than 2025 from March to June, −2 in late July, +13 from mid-August, and +20 in September and October. Part of the March–April 2026 rise is seasonal, since μ also rose in spring 2025 as A fell.

μ and A are only partly separable on the upper shoulder of the curve. The posterior correlation of last A with last μ in the drifting test fits is 0.29–0.36 for lignite and 0.32–0.55 for coal, and 66–81% of test hours sit above 0.8 on the fitted sigmoid. There, a 15–20 EUR/MWh shift in μ with s ≈ 20 moves output by only a few tenths of a GW.

For hard coal, the μ walk in the fit ending 15 October runs between −31 and −8 EUR/MWh through winter 2025/26, between −14 and 0 in spring, and between +6 and +10 from late July 2026. The all-data coal fit did not converge: r-hat ran to the thousands, with chains in different modes. The converged 12-month fits cannot separate a 2026 shift from a summer seasonal pattern. The coal μ also carries 2.5 times any change in the coal price, which is fixed at 12 EUR/MWh thermal here.

In every fit the commitment weight λ sits at its upper bound (median 0.99–1.00). Output follows the centred ±3 h mean margin, not the hourly one, so the true horizon may be longer than ±3 h.

## Expectations (written 2026-11-21, before any fit was run)

The models are the gas curve of gas_curve2.py applied to each fuel: output = F_k + A·sigmoid((m_eff − μ)/s), with m_eff = m + λ(m_win − m), m_win the centred ±3 h mean margin, F_k a floor that follows a random walk over 4-week blocks counted back from the end of the fit (so the last block, which is carried forward, is complete), Student-t noise. Model (a) has one μ; model (b) lets μ follow a random walk over the same 4-week blocks, with A, s and λ shared, so the only difference between (a) and (b) is the μ walk. For hard coal the coal price is fixed at 12 EUR/MWh thermal and the drifting location absorbs any departure from it, since a constant latent coal price is collinear with μ; the coal μ path is therefore behaviour plus 2.5 times the coal price change, and the two cannot be separated. Lignite margin: p − (1.1·carbon + 8); coal margin: p − (2.5·12 + 0.85·carbon + 10).

E1. In the October fit (2025-10-15 to 2026-10-14), the lignite μ walk is flat to within ±10 EUR/MWh from October 2025 to July 2026 and then rises by at least 20 EUR/MWh between the July 2026 block and the last block (late September to mid-October 2026), with the onset in August or September 2026.

E2. The hard coal μ walk shows the same pattern with a rise of at least 15 EUR/MWh, onset in August or September 2026.

E3. In the August window (2026-08-12 to 08-25), whose fit ends at or before the onset, the drifting model's MAE is within 10% of the static model's MAE for each fuel: neither has seen the shift.

E4. In the October window (2026-10-15 to 10-28), the drifting model's MAE is at least 20% lower than the static model's for each fuel, and the static model over-predicts output there (signed mean error of prediction minus actual above +0.3 GW).

E5. The 90% predictive interval of the drifting model covers between 80% and 97% of hourly output in both windows for both fuels; the static model's coverage falls below 80% in the October window.

E6. Sampling is clean: fewer than 1% divergent transitions and maximum r-hat below 1.05 for every fit. If not, the model is simplified (fewer drifting parts, coarser blocks) rather than forced.

E7. The commitment weight λ is above 0.3 for lignite (lignite units are slower to commit than gas).

## Registered specification (v1) results, recorded as run

The eight window fits of the registered specification ran cleanly (at most 1 divergence in 2400 draws, maximum r-hat 1.022). Test scores, static then drifting, MAE / signed mean error (prediction minus actual) / 90% coverage, GW: lignite August 1.98 / +1.69 / 0.52 against 2.03 / +1.75 / 0.49; lignite October 1.09 / +0.68 / 0.95 against 1.22 / +0.84 / 0.89; coal August 0.68 / −0.49 / 0.96 against 0.67 / −0.48 / 0.95; coal October 0.66 / +0.55 / 0.89 against 0.70 / +0.57 / 0.87. The drifting model did not beat the static one anywhere. In-sample checks showed why the v1 walk was not measuring location: in the lignite October fit the μ step scale went to 25 EUR/MWh per block against a prior scale of 10, μ swung −35 → +50 → +20 over the last three blocks while the floor F fell to 0.7 GW and rose again (posterior correlation of last F with last μ 0.65), and in the static fit the saturated-region residuals ran +1.1 GW in October–December and −0.4 to −0.6 GW in April–July, i.e. the seasonal movement is in the top of the curve, which the F walk was imitating. The model was therefore respecified once (v2: F constant, the walk moved to A, μ step prior halved to 5) and, for coal, a variant with one scale walk on F + A (v3); the choice between v1, v2 and v3 is made on validation windows ending 14 days before each test window (fits ending 2026-07-29 and 2026-10-01, scored on the following 14 days) before the test windows are rerun once.

## Respecified model (v2), validation and tests

On the validation windows, v2 had the lowest or near-lowest MAE for every fuel and window. The one exception was the lignite fit ending 1 October: v1 drifting scored 1.03 GW against 1.14 GW for v2 drifting, and v1 static failed to converge there (r-hat 217). The coal scale-walk variant v3 was no better than v2: 0.49 against 0.48 GW in late July and early August, and 0.60 against 0.59 GW in early October. Its static fit also failed to converge on one window (r-hat 1200). v2 was therefore used for both fuels.

v2 removes the F–μ trade-off, since F is constant, but replaces it with a weaker A–μ trade-off. The posterior correlation of last A with last μ is 0.29–0.55, against 0.57–0.90 for last F with last μ under v1. On validation, the drifting v2 model beat the static v2 model in all four cases, by 2–17%:
- lignite: 0.99 against 1.01 GW, and 1.14 against 1.38 GW;
- coal: 0.48 against 0.51 GW, and 0.59 against 0.64 GW.

On the test windows, which were run once with v2, it did not.

The test windows were scored as median prediction against actual over 336 hours, with the last block's μ, A and F carried forward. Signed error is prediction minus actual; coverage is of the 90% interval.

| Window | Static: MAE, signed error, coverage | Drifting: MAE, signed error, coverage |
|---|---|---|
| Lignite August | 1.77, +1.50, 0.58 | 1.80, +1.52, 0.54 |
| Lignite October | 0.96, +0.63, 0.93 | 1.02, +0.65, 0.89 |
| Coal August | 0.75, −0.50, 0.94 | 0.67, −0.46, 0.96 |
| Coal October | 0.58, +0.52, 0.92 | 0.61, +0.52, 0.91 |

The fixed-curve baseline was added after the tests, so it is a reference point, not a selection. Its scores:

| Window | MAE | Signed error | Coverage |
|---|---|---|---|
| Lignite August | 2.51 | +2.44 | 0.56 |
| Lignite October | 1.08 | −0.24 | 0.99 |
| Coal August | 0.54 | +0.32 | 0.98 |
| Coal October | 1.16 | +1.16 | 0.92 |

Sampling for all twelve test-window fits gave at most 2 divergent transitions in 2400 draws and a maximum r-hat of 1.019.

The μ step scale settled at 6.8–12 EUR/MWh per block, above its prior scale of 5, so the walk still absorbs some movement beyond location. This is clearest in the last three blocks of the lignite October fit: μ moves −10, +24, +7 EUR/MWh while A moves 5.1, 7.2, 8.5 GW. In that fit, lignite μ runs:
- −19 to −2 from October 2025 to early March 2026;
- −1 to +4 from April to June;
- −10 in the block starting 23 July;
- +7 at the end (90% band +5 to +10).

The fitted width s is 18–23 EUR/MWh for both fuels, against 10–14 for the fixed curve.

## Expectations: verdicts

E1 is not met. Under v2 the lignite μ was not flat through July 2026: it rose about 12 EUR/MWh at the start of April, partly seasonally. The rise from the July 2026 block (−10) to the last block (+7) is 17, short of 20. Under v1 the path could not be interpreted (−35, +50, +20).

E2 is not met. Coal μ rises about 20 EUR/MWh from winter to late summer 2026, but gradually from May, and the rise cannot be told apart from a seasonal pattern.

E3 is not met. For lignite the drifting model is within the band, 2% worse. For coal it is 11% better, outside the 10% band.

E4 is not met. The drifting model is 6% worse for lignite and 5% worse for coal in October. The static model does over-predict there (+0.63 and +0.52 GW, above +0.3), but so does the drifting one.

E5 is not met. Drifting coverage is 0.89, 0.96 and 0.91 in three cases but 0.54 for lignite in August. Static coverage in October is 0.93 and 0.92, not below 0.80.

E6 is not met. All twelve test-window fits are clean. Three other fits failed to converge and were dropped rather than forced: v1 static lignite on the validation window ending 1 October, v3 static coal on the validation window ending 29 July, and the all-data v2 coal fit.

E7 is met. λ is 0.99–1.00 for lignite and 0.96–1.00 for coal, at the boundary of its prior.

## Caveats

Two 14-day windows of strongly autocorrelated hourly output are thin evidence.

The specification was changed once, after the registered model's test scores had been seen. The change was justified from in-sample diagnostics and chosen on validation windows, but it was not blind. The fixed-curve baseline was also added after the tests.

A shared width s cannot represent the slower rise of the 2026 curve. October 2026 reaches the October 2025 top only above +100 EUR/MWh, so part of the departure is split between μ and A.

The A walk also absorbs seasonal availability, such as planned outages in summer.

The 4-week block carried forward is the main source of error. The August lignite miss is a drop inside the test fortnight that no curve fitted to earlier data anticipates.

Files: research/coupled-auction/lig_coal_curves.py (fits and figure; run outputs are kept outside the repo, in the scratchpad's lcc folder) and research/coupled-auction/lig_coal_curves.png.
