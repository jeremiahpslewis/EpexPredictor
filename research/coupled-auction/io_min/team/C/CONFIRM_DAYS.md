# Pre-registered confirmation days for the Timestral nbpast spec (written 2026-11-26 before the 12-day screen finished)
Rule: deterministic. 61-window days not in the 12-day screen, every 4th from the 2nd (14); fresh days (11-09..11-26) not in the
screen, spread evenly: the remaining list at positions round(i*11/6), i=0..6 (7). One wave, 3 procs x 7 days, daily stage-1, strict vintage, base RL. Not extended.
Screen days (excluded): 2026-09-10, 2026-09-20, 2026-09-30, 2026-10-10, 2026-10-20, 2026-10-30, 2026-11-10, 2026-11-13, 2026-11-16, 2026-11-19, 2026-11-22, 2026-11-25
61-window (14): 2026-09-09,2026-09-14,2026-09-18,2026-09-23,2026-09-27,2026-10-02,2026-10-06,2026-10-11,2026-10-15,2026-10-19,2026-10-24,2026-10-28,2026-11-02,2026-11-06
fresh (7, FAITHFUL_FRESH=1): 2026-11-09,2026-11-12,2026-11-15,2026-11-18,2026-11-20,2026-11-23,2026-11-26

Update 2026-11-26 (before screen results; S chose option A): one-day dump + inline nbpast replay costs ~215 s, so the wave is
4 procs x 5 days = 20 days on an empty machine. Rule: drop the last fresh day (2026-11-26). Final list:
- (superseded) proc 1: 2026-09-09,2026-09-14,2026-09-18,2026-09-23,2026-09-27
- (superseded) proc 2: 2026-10-02,2026-10-06,2026-10-11,2026-10-15,2026-10-19
- (superseded) proc 3 (FAITHFUL_FRESH=1): 2026-10-24,2026-10-28,2026-11-02,2026-11-06,2026-11-09
- (superseded) proc 4 (FAITHFUL_FRESH=1): 2026-11-12,2026-11-15,2026-11-18,2026-11-20,2026-11-23
Fresh mode reads stores re-fetched to 11-26 with the local vintages first; every day's base replay must equal the stored
champion Timestral (max abs diff 0), otherwise that day is reported as invalid, not replaced.

Final rule (2026-11-26, set by S before the wave, before any confirmation result): 4 days per process, 16 days = the first 11 of
the 61-window list and the first 5 of the fresh list above. Wall clock per process capped at 20 min; an unfinished day is
dropped and never topped up.
- proc 1: 2026-09-09,2026-09-14,2026-09-18,2026-09-23
- proc 2: 2026-09-27,2026-10-02,2026-10-06,2026-10-11
- proc 3 (FAITHFUL_FRESH=1): 2026-10-15,2026-10-19,2026-10-24,2026-11-09
- proc 4 (FAITHFUL_FRESH=1): 2026-11-12,2026-11-15,2026-11-18,2026-11-20
