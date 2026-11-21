# io_min: expectations written before any run (2026-11-21)

Panel: hourly, 2025-10-01 to 2026-11-08, 9,673 hours; DE_LU against the aggregate of its ten SDAC neighbours.

## Definitions fixed before the runs

- Outage, primary definition: 1 GW of the cheapest flexible gas, i.e. a CCGT that is in merit whenever gas runs above
  its floor: S_G^cf(p) = S_G(p) - min(1, S_G(p) - F_G). The must-run floor is not touched.
- Outage, secondary definition: an average unit, i.e. available gas capacity minus 1 GW, so the flexible part shrinks
  proportionally across the cost distribution (an upper part of which is out of merit anyway).
- Every counterfactual is model counterfactual minus model baseline. Both re-clear with the same recovered shocks.

## Expectations (thresholds)

- E1 recovery (synthetic data from a fixed theta on the real exogenous series): the true value lies inside the 90% interval
  for at least 80% of structural parameters. The gas heat rate is recovered with posterior sd below 0.05, and gas
  availability with sd below 0.03.
- E2 real fit: r-hat below 1.01, divergences below 1%. Gas mean heat rate h_G between 1.8 and 2.3 MWh_th/MWh_el. Gas
  availability alpha_G between 0.70 and 0.95. Exchange slope at the centre, (I+E)/(4 s_m), between 0.2 and 0.8 GW per
  EUR/MWh (binned data suggest about 0.6).
- E3 confounded pair: by economics, gas availability alpha_G against gas heat-rate dispersion sigma_G (both set how
  much gas is on near the top of the curve, which is extrapolated in most hours). Expected posterior |corr| > 0.4.
  Runner-up: exchange scale s_m against the DE supply slope (they split a DE shock between Δprice and Δimports).
- E4 counterfactual, primary outage:
  - mean ΔpDE over all hours 1-4 EUR/MWh; 3-8 EUR/MWh in hours where gas is marginal;
  - imports +0.3 to +0.6 GW;
  - gas output -0.3 to -0.6 GW net (other gas steps up);
  - lignite + coal +0.0 to +0.2 GW;
  - ΔpN +0.3 to +1.5 EUR/MWh.

  Basis: a net slope of about 3 EUR/MWh per GW (notes 21.145).
- E5 external check: the model's dp/dD from a +1 GW re-clear, averaged over hours, is within 0.8 to 1.25 of the
  empirical net slope on the same hours by the 21.144 recipe. By price level, the model is too steep at high prices,
  as in 21.145. This is an over-identification check on the same data, not an out-of-sample test.
