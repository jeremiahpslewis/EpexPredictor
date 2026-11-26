# I (inputs): residual-load bound, decomposition, cross-border drivers (2026-11-26)

Scripts (arrays and logs in `$SP/team/I/`, nothing in the repo):
- `oracle_rl.py`: the faithful harness (`faithful_bt.py`, imported unchanged) with the true residual load. It patches
  `rl_components` (tree RL) and `TimestralLoad.residual_load` (Timestral's own RL, `ORACLE_TS`). `ORACLE_TARGETS`
  picks the components. "True" means the stored TSO day-ahead figures (load − wind − solar), the same series the RL
  boosters are trained on. They are not known at the gate.
- `score_oracle.py SP DIR`: paired with `out_strict` on the same days, per member, by window, per day.
- `size_drivers.py SP` (`MODE=combo` for combinations): offline sizing of 79 cross-border and fundamental candidates
  against the blend's day-by-window level miss.
- `entsoe_probe.py SP`: checks whether D's ENTSO-E day-ahead load (6.1.B) and wind/solar (14.1.D) documents exist at
  probe time. Network only. It must run before the gate to answer the timing question.

## 1. True-RL bound (I001): 9 of the 61 days, already looked at

Days: 09-09, 09-15, 09-20, 09-25, 10-03, 10-13, 10-19, 10-29, 11-06. They were picked by weekday spread before any
scoring. Champion blend on these days: 1.808, against 1.648 on all 61. Intervals: per-day bootstrap, 90%.

| member | champion | true RL | diff |
|---|---|---|---|
| trees | 2.005 | 1.597 | −0.408 [−0.646, −0.130] (−20%) |
| Timestral | 1.853 | 1.615 | −0.238 [−0.386, −0.085] (−13%) |
| market route | 1.999 | 1.952 | −0.047 [−0.403, +0.328] (worse on 3 of 9 days) |
| blend | 1.808 | 1.583 | −0.225 [−0.468, +0.068] (−12.5%) |
| blend, oracle trees only | | 1.580 | −0.228 [−0.326, −0.134] (8 of 9 days) |

- **By window (blend):** 00-06 −0.31, 06-10 −0.55, 10-16 +0.01, 16-21 −0.24, 21-24 −0.07.
- **Blend bias:** −0.48 → −0.09.
- **Market route:** P's offline check (`team/P/marketlab.py`) finds that its Timestral shift forecast already absorbs
  part of the RL error. Re-running the shift with true RL double-corrects on big-miss days (09-09). This is expected
  from its design and needs a consistent RL definition if RL is fixed.
- **This is not a clean ceiling.** Here Timestral's own RL is also set to the truth, and the market route then
  double-corrects. On the I003 days this configuration gives −0.062 against −0.283 for the parts without that change.
- **`matched` on the same 9 days, paired:**
  - blend −0.156, trees −0.286, Timestral −0.060, market −0.029;
  - room left (oracle − matched): blend **−0.069** (6 of 9 days), trees −0.122, Timestral −0.179, market −0.018.
  - On the I003 days the wind/solar-only oracle leaves −0.076 beyond `matched`.
- **Verdict:** the RL lever is worth about 12–17% of the blend on these days, and `matched` already has most of it.
  The room left beyond `matched` is about 0.07–0.08 ct, under S's 0.10 bar, on days that favour `matched` (on all
  61 days `matched` gets −0.061). A clean 9-day `ORACLE_TS=0` wave would settle the size; it is proposed, not run.

## 1b. Decomposition (I003): 4 of the 9 days, high RL miss (champion tree RL 4.07 GW)

Days: 09-09, 09-20, 10-03, 10-19 (every second day of the list, fixed before scoring). Timestral's own RL is left as
the model (`ORACLE_TS=0`). Each cell is the change against the champion.

| | trees | Timestral | market | blend |
|---|---|---|---|---|
| true load only (≈ `published`, but also true after D) | +0.076 | −0.060 | −0.012 | +0.005 (2 of 4 days better) |
| true wind+solar only | −0.559 | −0.194 | +0.023 | **−0.288** (4 of 4 days) |
| sum of the two | −0.484 | −0.254 | +0.011 | −0.283 |
| full oracle (`ORACLE_TS=1`), same days | −0.423 | −0.150 | +0.234 | −0.062 |
| `matched` (in hand), same days | −0.377 | −0.120 | −0.047 | −0.212 |

- **The lever is wind and solar.** No price gain from true load was detected on these 4 high-miss days (trees +0.53
  in 10-16, −0.16 in 16-21). The load part is not sized on typical days.
- **The parts add up**: −0.283 against the wind/solar-only −0.288. With Timestral's own RL also set to the truth, the
  market route double-corrects (P's finding), which is why the full oracle is smaller here.
- **Caveat:** 4 days with large RL misses, so the shares are indicative only.

## 2. Where the RL error sits (stored components, `matched`, all 61 / fresh 18 days)

| RL MAE on D (GW) | 61 | fresh 18 |
|---|---|---|
| app (champion) | 2.56 | 2.45 |
| matched | 2.29 | 2.15 |
| matched + true load (= `matchedpub`) | 2.03 | 1.71 |
| matched load + true wind/solar | 0.89 | 1.39 |
| component MAE (matched): load / wind / solar | 0.89 / 1.71 / 0.71 | 1.39 / 1.55 / 0.57 |

- **Weather age:** d1 slots (night of D, 26% of slots) score 1.41 against 2.60 on d2+ slots. This is confounded with
  the hour, since d1 covers only the night.
- **The ../../rl study** (autumn, all hours of day 1), RL MAE:
  - all-d1 weather 2.16;
  - pessimistic gate map 2.69.
- **With near-perfect (archive) weather** about half of the RL MSE remains: it is model error against the TSO's
  figure. The live system (24–44 h lead, between d1 and d2) should therefore sit near 2.0–2.2 GW on its own, and the
  rest of the gap to the oracle is a model-quality problem, not a weather-age problem.

## 3. Cross-border and fundamental drivers (I002): closed as linear corrections

- **Test:** causal walk-forward correction `e ~ window FE + b·x` of the blend's day-by-window miss on 61 days
  (09-08..11-07, 47 scored).
- **Timing assumption:** the neighbours' day-ahead load is assumed to be out before the gate; this is not verified.
- **`$SP/team/I/daywin_frame.parquet` is for analysis only.** It holds columns that are not gate-known: `de_rl_err`
  (truth), and `nb_ws_*` / `nb_rl_*` (14.1.D).
- **No gate-known candidate helps on its own** (−0.001 to +0.12). The candidates: neighbours' day-ahead load, change
  and level, 11 zones and their sum; FR nuclear unavailability (pregate); THE gas; EUA.
- **The DE RL error (truth) does:** −0.102 [−0.183, −0.021].
- **Neighbours' day-ahead wind+solar change (14.1.D, not gate-known)** helps only jointly with the DE RL error:
  −0.14 to −0.15. The partial correlation given the RL error is 0.2–0.3 for BE/FR/NL wind. It acts as a weather
  signal, not a cross-border fundamental.
- **Fresh 18 days:** FR nuclear −0.09 against the intercept, on 8 days only (the series ends 11-16); gas and EUA ≈ 0.

## 4. Publication timing (open)

- **Probe at 16:26 UTC on 11-26** (after the gate), for D = 11-27:
  - DE_LU day-ahead load (A65) is present, 96 points;
  - **DE_LU wind/solar (A69) is ABSENT**;
  - FR, NL and BE have both;
  - nothing exists yet for D = 11-28.
  - So DE's TSO day-ahead wind/solar for D was not out by D-1 16:26 UTC (17:26 CET), well after the gate. It is not
    gate-known. This agrees with E's energy-charts probe (no wind/solar for D at 15:51 UTC).
- **Manual morning check (11-27, 07:00–10:45 UTC, network only, about 5 min, never inside a heavy slot):**

  ```
  SP=/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad
  $SP/venv/bin/python -I research/coupled-auction/io_min/team/I/entsoe_probe.py $SP
  grep "D=2026-11-28" $SP/team/I/probe/entsoe_probe.log
  ```

  The key is read from `$SP/entsoe/key` and is never printed.
  - **"Yes"** for a series is a line `DE_LU A69 D=2026-11-28 | status 200 ... points N` with N > 0, at a probe time
    before 11:00 UTC.
  - **"No"** is `ACK No matching data found`.
  - The DE A65 line answers the `published` load question the same way.

## 5. Wind booster screen (I004, I005; pre-registered in WIND_PREREG.md, `wind_lab.py`, `wind_score.py`)

**Setup.** d2 history and d2 weather; wind MAE against the TSO figure on k ≥ 2 slots; 61 and fresh 18 days.

**Weekly refits (I004).**
- V0 scores 2.019 / 1.775 GW, against the stored daily `matched` 1.894 / 1.663. Weekly refits alone cost about 7%.
- Best variant: V7, the capacity-factor target plus extra aggregates, chosen post hoc: −5.0% / −3.3%.

**Daily check on 8 fixed days (I005).** The weekly winner V1 turns worse with daily refits (+4.5% / +18%), because
its trend correction only offsets staleness. So the weekly screen cannot rank variants (P found the same for the
price trees).

**Verdict.** No variant comes near −10% on both windows. Booster form is not where the wind error is. Nothing goes to
a price screen.

## 6. Grid points / second NWP

`GRID_REQUEST.md` is a request list only. It proposes a sizing pilot first: about 2,700 weighted Open-Meteo calls
of archive wind on about 90 denser northern points, scored with archive weather. Vintage history follows only on a
go.
