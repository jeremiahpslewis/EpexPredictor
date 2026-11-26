# fclog: written before scoring (2026-11-26 ~17:30 UTC)

**Idea.** The level miss follows the RL forecast error (anatomy: corr 0.63), and the trees train on published-RL /
archive-weather rows. Adding the stage-2 rows as they were at forecast time for past days (vintage weather, forecast
RL, out-of-sample stage-1 cross prices; labelled with the known price) should teach the trees how far to trust the
forecast RL.

A deployable version needs the app to log its stage-2 feature rows each day. This test emulates that log with the
dumped D frames.

**Variants** (`screen.py --daily2`, matched-RL frames `D`, stage 1 refit weekly, stage 2 refit daily):
- `base@d2`: control. The origin's training frame plus the forecast-time frames of the days from the origin to D-1.
- `fclog1@d2`, `fclog5@d2`: the origin's training frame, with the rows at logged times replaced by every earlier
  forecast-time frame from the start of the log, at weight 1 and at weight 5.

**Test set: fresh18** (11-09..11-26). It was not used to select fclog, but it was used for P002/P003, so it is not
pristine. Stage-1 origins are 11-09, 11-16 and 11-23 (`fresh_w7`). The log covers 09-08..11-07 (`w8` frames) plus the
fresh days before D.

**Also reported:** the 61 days (`w8`), split into the first and second 30 days, where the log is 0-29 and 30-60 days
long.

**Decision rule.**
- An fclog variant is a candidate only if its paired diff vs `base@d2` on fresh18 is < 0 with a 90% interval excluding
  0, and the 61-day second half is not worse.
- Two variants are tested, so a hit at the interval's edge counts as weak.
- A candidate then needs the daily faithful check and E's walk-forward sample.
