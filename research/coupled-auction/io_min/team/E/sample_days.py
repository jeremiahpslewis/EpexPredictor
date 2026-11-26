"""Stratified sample of target days for the long walk-forward window, 2025-12-01..2026-11-26 (team E).

Core sample (S, 2026-11-26): 3 days per month x 12 months = 36 days, one wave of <= 20 min.
  - month i (0 = 2025-12) gets weekdays (3i, 3i+1, 3i+2) mod 7, so over 12 months every weekday appears 5 or 6 times;
  - the 3 days fall in the month's thirds (days 1-10, 11-20, 21-end), one per third, so each month is spread;
  - within a third the day with the assigned weekday is drawn with a fixed seed (20261126). The weekday-to-third
    assignment is also drawn.
Extension (only if the user approves more compute): 2 per ISO week, drawn the same way from the remaining days, written
separately; the core is never changed after scoring starts.
Writes team/E/sample_core36.txt (one ISO date per line) and prints a summary.
"""
import calendar
import os
from collections import Counter
from datetime import date

import numpy as np

SEED = 20261126
FIRST, LAST = date(2025, 12, 1), date(2026, 11, 26)


def months():
    y, m = FIRST.year, FIRST.month
    while (y, m) <= (LAST.year, LAST.month):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def core(rng):
    out = []
    for i, (y, m) in enumerate(months()):
        wds = [(3 * i + j) % 7 for j in range(3)]
        rng.shuffle(wds)
        last = min(calendar.monthrange(y, m)[1], LAST.day if (y, m) == (LAST.year, LAST.month) else 31)
        thirds = [(1, 10), (11, 20), (21, last)]
        for (lo, hi), wd in zip(thirds, wds):
            cands = [date(y, m, dd) for dd in range(lo, hi + 1) if date(y, m, dd).weekday() == wd]
            out.append(cands[rng.integers(len(cands))])
    return sorted(out)


if __name__ == "__main__":
    days = core(np.random.default_rng(SEED))
    assert len(days) == 36 and len(set(days)) == 36 and FIRST <= days[0] and days[-1] <= LAST
    fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_core36.txt")
    open(fn, "w").write("".join(f"{d}\n" for d in days))
    print(fn)
    print("weekdays:", dict(sorted(Counter(d.strftime("%a") for d in days).items())))
    print("months:", dict(Counter(d.strftime("%Y-%m") for d in days)))
    print(",".join(map(str, days)))
