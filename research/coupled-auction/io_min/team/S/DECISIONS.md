# S decisions log

- **2026-11-26 16:00 UTC. Interval convention.** Paired circular 7-day block bootstrap, two-sided 90%, as in
  `faithful/RESULTS_MATCHED.md`. "Excludes 0" in the promotion rule means this interval.
- **2026-11-26 16:00. Combination lever closed.** In-sample bounds ≤ 2.6% on the 61 days and every causal weighting or
  bias scheme is flat or worse (TRIALS S001-S012). C is moved to Timestral covariates, member disagreement as a
  feature, and a less correlated fourth member.
- **2026-11-26 16:00. Negative-price regime parked** (8% of error on the 61 days, none in November).
- **2026-11-26 16:00. Live forecast for D = 2026-11-28 (provisional rule, evidence available today).**
  E issues before 10:15 UTC on 11-27: the deployed app's forecast, the frozen champion configuration, and `matched`.
  The team's forecast is `matched`: it is the only variant with a paired interval excluding 0 on stored days (61 days
  and pooled 79), its models are trained up to each day's cutoff only, and its sign is negative in all three months.
  It is not promoted to champion (no walk-forward months yet), so it is labelled "candidate" in the file.
  Fallback: the frozen champion configuration, if the `matched` run fails an assertion or is not done by 10:05 UTC.
  `published` / `matchedpub` stay shadows until the probe log shows D's TSO load before the gate on more days.
- **2026-11-26 16:00. Walk-forward sample must fit one ≤ 20-min wave** (≤ 36 days at measured runtimes); no chained
  waves. If that is too thin for the 9-of-12-months rule, the question goes to the user.
- **2026-11-26 ~16:15. Live forecast for D = 2026-11-28: decision changed. Team forecast = frozen champion
  configuration on live inputs; `matched` is a labelled shadow.** Review of E's `FAITHFUL_LIVE` mode showed that live
  weather is the latest forecast on every slot (about 24-44 h lead on D), while `matched` still switches to boosters
  trained on 2-day-old weather on most of D. The stored evidence for `matched` compares both model sets on 2-day-old
  weather; the live pairing (d2-trained boosters on younger weather) has no backtest. Order in the team file:
  champion, else app. `matched` and its residual-load components are logged so the TSO figures of 11-28 show which
  residual load was closer on live weather.
  Consequence for the benchmark: the strict-vintage backtest is pessimistic about weather age compared with live
  operation (strict tree RL 2.56 GW; the live-logged series scored 2.11 GW on the same 61 days). The live log measures
  the gap from 11-27.
- **2026-11-26 ~16:20. Fourth member "similar day" (C009) dropped**: error correlation 0.4-0.57 with the members, but
  MAE 2.9-4.2, and it worsens the blend on the 61 days. Stacking and Timestral time-of-day bias correction closed (C007,
  C008). C continues with Timestral variants on replayed inputs.
- **2026-11-26 ~16:40. Live run is manual.** E's dry run for D = 2026-11-27 passed every assertion (app 290 s,
  champion 255 s, matched 393 s under load; fetch 6 min). A background waiter for 09:30 UTC was refused by the
  session's permission system and was not worked around. Someone has to start `team/E/live_forecast.py --day
  2026-11-28` at about 09:30 UTC on 11-27 (command in `E/LIVE.md`). The weather prefetch at 06:05 UTC is armed if the
  container stays up; the run falls back to stored weather and labels it.
- **2026-11-26 ~16:50. Walk-forward champion cache on hold; question for the user.** Measured cost is about 240 s per
  day and process, so one 20-minute wave holds 16 days, not 36. Sixteen days cannot support "9 of 12 months", and
  three waves would be chaining. In addition, strict-vintage grid weather (d1/d3) exists only from 2026-08-25; 24 of
  the 36 sampled days need about 1-2.5 thousand Open-Meteo calls first. Options: (A) a one-off exemption for the cache
  (36 days in 3 waves, about 40 min, and again per finalist); (B) one wave of 16 days and a rule by quarters. Until
  then the reference stays the 61 + 18 stored days plus the forward test.
