# Two-zone benchmark after the repair (2026-11-21)

## Fuel-split basins

Six chains were run, two started in each of three configurations; the hard floor cap was smoothed (SMOOTH = 0.05 GW).
Script `fit_modes.py`, 600 warmup and 600 draws per chain.

- Chains started in mode 0 and in the hybrid (mode 0 lignite and coal, mode 1 gas) converge to the same basin.
- Chains started in mode 1 stay in mode 1.
- Within each basin, max r-hat ≤ 1.007 with no divergences.

| Basin | F_L | F_C | gas floor φ0 | σ_L | σ_C | σ_G | α_L / α_C / α_G | h_G |
|---|---|---|---|---|---|---|---|---|
| Mode 0 | 0.34 | 1.44 | 3.72 | 1.16 | 0.10 | 0.13 | 0.94 / 0.39 / 0.41 | 1.67 |
| Mode 1 | 5.41 | 0.13 | 0.28 | 0.10 | 0.87 | 0.46 | 0.61 / 0.70 / 0.57 | 1.50 |

Mean log density by block:

| Basin | Price system | Lignite output | Coal output | Gas output | Prior |
|---|---|---|---|---|---|
| Mode 0 | −79,502 | −872 | −645 | −984 | −117 |
| Mode 1 | −79,642 | −891 | −685 | −962 | −107 |

Mode 0 is ahead by about 167 nats overall. The fuel observations pull in opposite directions: lignite and coal favour
mode 0, gas favours mode 1. In the raw data, observed gas output at negative prices (1.6 GW) matches mode 1. Observed
lignite above 200 EUR/MWh (11.2 GW) matches mode 0. Both basins overstate the floors at low prices.

Both basins are reported: a likelihood that treats thousands of hours as nearly independent overstates a 167-nat gap.

## Outage counterfactual by basin

1 GW of the cheapest flexible gas; means over 120 draws, cap hours excluded. Script `cf.py`.

| Basin | ΔpDE (EUR/MWh) | Δ imports (GW) | Δ gas | Δ lignite | Δ coal | ΔpN | Cap hours | Average-unit ΔpDE |
|---|---|---|---|---|---|---|---|---|
| Mode 0 | +2.77 (2.68–2.85) | +0.58 | −0.69 | +0.08 | +0.04 | +1.30 | 18 | +2.14 |
| Mode 1 | +3.45 (3.37–3.55) | +0.74 | −0.85 | +0.06 | +0.05 | +1.69 | 15 | +2.07 |

Within each basin the 90% band is about ±3%. The basins differ by 25%. The fuel split is a first-order source of
uncertainty for the outage, and a single-basin posterior would not show it.

## External check

Model dp/dD from a +1 GW German demand re-clear, against the empirical net slope (21.144 recipe, same hours):

| Price level | Model | Empirical | Empirical / model |
|---|---|---|---|
| All | 3.5–3.6 | 2.38 | 0.67 |
| Low | | | 0.77–0.79 |
| Middle | | | 0.66–0.68 |
| High | | | 0.56–0.57 |

The same holds in both basins. The model is too steep by about half, so the two-zone price effects are probably
overstated.
