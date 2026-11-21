# Is the multi-hour margin response unit commitment? (per-unit test)

## Result

The overall test I set in advance fails. It required E1 plus at least three of E2–E5, and only E1 and E3 are met. E2 misses one threshold narrowly, and E4 (the evening) and E5 (hysteresis) are not met. What did hold is the central point. The multi-hour response of German gas output runs through units switching on and off, and those units run in blocks of many hours. Holding the current margin, the clock hour and the season fixed, fleet output of the 50 units rises 0.057 GW per EUR/MWh of (m_win − m) (day-bootstrap 90% interval 0.051–0.072). 80% of that slope (77–83%) comes from the capacity that is on, and 12% from the load factor of units already running. This is a commitment response, with one qualification: the response to the current margin itself is also mostly on/off (75%), so the window channel is only somewhat more on/off than the static one. Runs are long. 7% of runs with an observed start last under 3 h, the Kaplan–Meier median run is 12 h and the median down-time 8 h. Starts sit on the rising edge of the margin. The median margin is −31 EUR/MWh three hours before a start, 7 EUR/MWh at the start hour and 22 EUR/MWh over the run.

Two parts of the commitment story are not supported. The unit data give no wider late-evening curve: the evening-to-midday width ratio is 0.78 or 1.28 depending on the fit, and system gasg gives the same two numbers on the same hours, so the 1.6× is a property of the earlier fit's specification, not of the plants. There is also no hysteresis at equal current margin once the clock hour is held fixed. A direct lead-versus-lag test shows that the window response is symmetric. Capacity on responds as strongly to the mean margin of the next three hours as to that of the previous three (0.028 and 0.027 GW per EUR/MWh). This supports a centred window rather than a backward-looking carry-over or a purely forward-looking one.

## Data and definitions

The data are ENTSO-E per-unit actual generation for gas units of at least 100 MW in the 50Hertz, Amprion and TenneT areas, from 2025-08-12 to 2026-08-11. The files hold alternate days only. That gives 187 local days in 179 contiguous stretches: 172 of 24 h and 7 of 48 h, where two consecutive days are kept at a month boundary. The same days are present in all three TSO areas. All states, runs and down-times are computed inside contiguous observed stretches. A missing value (0.4% of unit-hours, mostly one TenneT unit absent for whole days) is treated as unobserved, never as off.

A unit is on when its output exceeds 20% of its robust maximum, the 99th percentile of its hourly output over the 12 months. Units whose percentile is below 50 MW are dropped, which leaves 50 units with 15.2 GW. On the observed hours these units sum to a median 72% of system gas output (gasg).

The margin is m = p − (1.85·gas + 0.37·EUA), and m_win is the centred ±3 h mean. Both are computed on the continuous system grid and then joined to the unit hours. Only 2 units (0.3 GW) are on in at least 90% of observed hours, so nearly all units count as cyclers, including CHP units that cycle with the seasons.

Because each stretch is a single day, censoring dominates. Of the cycler on-runs, 38% cover the whole stretch, 21% are cut at the start, 21% are cut at the end, and only 20% are complete. Lengths are therefore taken by Kaplan–Meier from runs with an observed start.

## Expectation by expectation

E1, the fleet decomposition, is met. Within cells of 5-EUR margin bin × clock hour × season, the capacity-on channel accounts for 80% of the window slope (threshold 60%, 90% interval 77–83%). The result holds with other cells: 79% within 10-EUR bin × hour × month, and 84% within 5-EUR bin × month × hour block. In unit counts the slope is 0.16 units switched on per EUR/MWh of window gap. For comparison, the response to the current margin itself (within hour × month) is 75% capacity on. So both the static and the window response are mainly commitment, and the window part slightly more so.

E2, starts and the margin, is not met, because one threshold is missed. The share of run-hours at negative current margin is 24.2% against a threshold of 25%. The other parts hold. The median run-average margin exceeds the start-hour margin by 16.0 EUR/MWh (threshold 10). 40% of starts happen at negative margin, 82% of runs have a positive average margin, and 24% of runs start at a negative margin yet average positive. Winter and summer look alike: 26% and 22% of run-hours are at negative margin, and the start-to-run gap is 13 and 18 EUR/MWh. The null comparison came out as expected. Across all cycler hours, 36% of on-hours are at negative margin and 43% of off-hours at positive margin, both above 15%. The second number is a reminder that the 1.85 heat rate is a CCGT number. Older, open-cycle and heat-led units face different true margins, so a negative m does not mean the unit is losing money. The event study around starts is partly the shape of the average day, since most starts fall in the morning or late afternoon ramp.

E3, minimum run and down times, is met. Of runs with at least 3 h of observation after the start, 7.4% last under 3 h (threshold under 10%), and 26% last under 6 h. The Kaplan–Meier median run is 12 h (threshold at least 6 h). The median of complete runs alone is 5 h, which is biased short. Down-times under 3 h are 6.1% (threshold under 15%), with a median of 8 h.

E4, the evening, is not met. The carry-over itself holds. Of 1,264 cycler runs started between 16:00 and 19:59, 78% are still on at 22:00 (threshold 50%). These carried-over units average 2.1 GW of the 6.9 GW on in 20–24 h. But only 17% of their 21:00–23:59 run-hours are at negative margin (threshold 30%), because evening margins are usually still positive.

The wider evening curve is not reproduced either. I fitted a logistic in the current margin with monthly levels. For unit output the evening-to-midday width ratio is 0.78 with a free amplitude per block and 1.28 with an amplitude shared across blocks. For capacity on it is 1.09. System gasg on the same hours gives exactly the same 0.78 and 1.28. So the gap to 1.6 is not caused by the 28% of output missing from the unit files. It comes from the fit specification: the earlier hierarchical Student-t model with a shared amplitude.

Removing the carried-over units shrinks the evening width of capacity on to a third of midday. That removes exactly the units that respond to the evening margin, so it is not a clean counterfactual and I do not count it.

E5, hysteresis, is not met. The test used margins within the non-flat range of −52 to 64 EUR/MWh, bins of equal 5-EUR current margin, and the direction rule ±5 EUR/MWh over ±1 h. Within the same clock hour, capacity on is 0.03 GW lower when the margin is falling than when it is rising, with a 90% interval of −0.28 to +0.45 GW. In units the difference is −0.05 (90% −1.0 to +1.8). Pooled, or within 4–6 h blocks, the sign is the reverse of hysteresis: 0.22 and 0.52 GW less on when falling. That pattern is a time-of-day effect and disappears at clock-hour resolution. This direction test is weak for the question, because the ±1 h direction barely separates the window gap (mean m_win − m of −1.3 EUR/MWh when falling, −2.2 when rising). The lead/lag regression above is the sharper test, and it finds the response symmetric, not a backward-looking carry-over.

## What it means for modelling gas offers

Gas supply should be modelled with on/off commitment rather than independent hourly steps. The data support blocks of roughly 6–12 h, a minimum run of about 3–4 h (over 90% of runs exceed 3 h) and minimum down-times of the same order. The commitment decision weighs the margin over a window centred on the hour, about three hours each side. That is what the fitted m_win already encodes, and it is a reasonable reduced form. A separate hysteresis or lag-state term is not supported.

Start cost is only weakly identified here. The gross margin that a complete run earns, summed over its hours and scaled to the unit's capacity, is an upper bound on what a start may cost, and it has a median of 55 EUR per MW of capacity and a lower quartile of 7 EUR/MW. Meanwhile 18% of runs have a negative average margin at the 1.85 heat rate. So start costs are at most a few tens of EUR/MW, but the data do not pin them down. The late-evening width should not be attributed to commitment of large units.

## Caveats

Sampling on alternate days makes every stretch a single day. Runs crossing midnight are censored and overnight runs are invisible as runs. Run durations beyond about 12 h are identified only from runs that start in the morning. Down-times longer than about 20 h cannot be measured. Run-average margins of right-censored runs stop at midnight.

The units cover 72% of gas output. The single 1.85 heat rate misstates the margin of non-CCGT and heat-led CHP units, and the 20% threshold treats low-load CHP operation as off. The width comparison depends on the fit specification. The bootstraps resample days, with 300 draws.

## Expectations (written before any computation)

If the λ ≈ 0.84–0.93 window response is commitment, the per-unit data should show the following. Units are ENTSO-E per-unit actual generation (≥100 MW, gas, TSOs 50Hertz, Amprion, TenneT), 2025-08-12 to 2026-08-11; a unit is "on" when output exceeds 20% of its robust maximum (99th percentile of hourly output over the 12 months). Units are split into cyclers (on in less than 90% of observed hours) and near-always-on units (heat-led CHP / must-run), and the start/run tests are run on cyclers only.

E1 (fleet decomposition, the discriminating test). At fixed current margin m, the residual of fleet output against (m_win − m) is carried mainly by the number (capacity) of units on, not by output per on unit: the capacity-on channel accounts for at least 60% of the slope. Against commitment: output per on unit carries more than half.

E2 (starts and margin). For runs of cyclers with an observed start, the run-average margin exceeds the start-hour margin by at least 10 EUR/MWh in the median, and at least 25% of run-hours occur at negative current margin while the median run-average margin is positive. Against: the start-hour margin is about as high as the run average and fewer than 10% of run-hours are at negative margin. Null comparison: the share of cycler off-hours at positive margin should be well below the share of run-hours at negative margin only if commitment is not symmetric; I expect both to be substantial (≥15%), i.e. units both stay on through dips and stay off through short spikes.

E3 (minimum run and down times). Among cycler runs with an observed start at least 3 h before the observed stretch ends, fewer than 10% last under 3 h; the median completed run is at least 6 h. Among observed down-times (with the same 3 h rule), fewer than 15% are under 3 h. Against: more than 25% of runs under 3 h (peaker-like hour-by-hour dispatch).

E4 (evening). Of cycler runs started between 16:00 and 19:59 local, at least 50% are still on at 22:00, and at least 30% of their 21:00–23:59 hours are at negative margin. The empirical units-on-versus-margin curve is wider in 20–24 h than in 10–16 h by a factor of at least 1.3, and removing the carried-over evening units narrows the evening width by at least half of the excess over midday. Against: evening-started runs mostly stop by 22:00, or the evening width is no wider than midday.

E5 (hysteresis). At equal current margin (bins within the non-flat region of the curve), more capacity is on when the margin is falling (dm < −5 EUR/MWh over ±1 h) than when it is rising (dm > +5), by at least 1 GW of capacity on, and this survives comparison within the same hour block (day-bootstrap 90% interval excludes zero). Against: the difference vanishes within hour blocks.

Overall: if E1 and at least three of E2–E5 are met, the λ window response is largely commitment.
