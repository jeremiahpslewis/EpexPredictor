# consist confirmation: written before the run (2026-11-26 ~23:45 UTC)

The spec is frozen in CONSIST_SPEC.md, with no retuning. consist was screened on 6 fresh days (11-10, 13, 16, 19, 22,
25). They are excluded here.

**Days**
- **Part A, independent of the screen:** all 61 window days 09-08..11-07. The log covers 0-60 of about 85 context days.
- **Part B:** the 12 remaining fresh days: 11-09, 11-11, 11-12, 11-14, 11-15, 11-17, 11-18, 11-20, 11-21, 11-23, 11-24,
  11-26.

**Primary comparison.** consist96 vs repro96, paired per day, on the same rebuilt frames and horizon. Reported:
- mean diff with a 90% circular 7-day block bootstrap, for A, for B, and pooled;
- days better;
- by month (Sep, Oct, Nov);
- by log-length tercile on A;
- without the single best day.

**Secondary**
- repro96 vs the stored champion market route (construction effect).
- consist96 vs the stored champion market route.
- Blend with consist96 replacing the market route, vs the champion blend; and with repro96 replacing it, as the
  control.
- Error correlation of consist96 with the champion trees and Timestral.

**Decision rule.**
- consist is a candidate for the walk-forward / live shadow if pooled A + B is < 0 with the interval excluding 0, and
  A alone is < 0.
- A gain that grows with log length is expected if the mechanism is real.
- Two procs, each at most 18 min. Unfinished days are dropped.
