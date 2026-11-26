# shiftl1: frozen specification (written 2026-11-26 ~16:35 UTC, before any 61-day or daily-cadence number for it)

It was selected on fresh18 at weekly cadence, as the best of 9 variants, so fresh18 is a selection set for it. The test
sets are the 61-day weekly dump and the six 61-window days of `fixed_days.txt` at daily cadence. No retuning on these.

**What it is.** The deployed stage-2 frame is unchanged: same rows, same features, stage-1 cross prices. The label and
the output mapping change.

**Label.** For every training row (all ≤ cutoff, the app's training rows after `dropna`):

`shift = marketmodel.supply_shift(residual_load - 8.0, price * 10, gasprice, 75.0)`

- `price` is the known DE price in ct/kWh.
- `residual_load` is the frame's column: for rows ≤ cutoff, the TSO day-ahead figures as `ResidualLoad.get_data`
  serves them.
- `gasprice` is the frame's column (THE, ffilled).

Rows with |price| ≤ 0.02 ct/kWh are left out, because the shift is not identified in the zero band.

**Model.** One LightGBM booster, `objective = l1`, otherwise the app's defaults: 100 rounds, learning rate 0.1, 31
leaves, `force_col_wise`. No weights, no gas-relative twin.

**Forecast for D.** `price = marketmodel.clearing_price(residual_load_D - 8.0, gasprice_D, 75.0, shift = booster(X_D)) / 10`

- `residual_load_D` is the stage-2 frame's RL on D: the app's RL trees on vintage weather, or under `matched` the
  matched selection. It is the same series the champion trees read, gate-known.
- `gasprice_D` is the frame's gas on D: gas day D-1 ffilled, as the app reads it, known at the gate.
- Carbon price 75 and other plant 8 GW are the market route's constants.

**Leak review.**
- Labels use only prices and published TSO figures ≤ cutoff; this is asserted in `dump.py` / `assert_no_leak`.
- The clearing for D uses only the forecast RL and the gate-known gas. No published D wind, solar or load is used,
  except that under the `published` RL variants, which are not used here, the published load would enter through the RL.

Code: `screen.py` variant `shiftl1` on a `dump.py` dump; with `--train-every 1` every day is its own origin (daily member).
