# io_min, eleven zones: expectations for the two worked MTUs (written before running them, 2026-11-21)

Zones: DE_LU, FR, NL, BE, AT, CZ, PL, DK_1, DK_2, SE_4, NO_2. Time unit: the auction's 15-minute MTU.

Network sources:
- Core flow-based domain from JAO: presolved CNECs with PTDFs and RAM, external constraints, and allocation constraints.
- Core hubs omitted from the model (HR, HU, RO, SI, SK) held at their published net positions.
- Links outside the Core domain at the time (DE–DK2, DE–SE4, DE–NO2, NL–NO2, PL–SE4, and DE–DK1 when its virtual hub is
  inactive) are directional limits.
- Flows among the Nordic zones and between them and the rest of Europe are held at their scheduled values. Results
  are conditional on those flows.

Worked MTUs:
- Congested: 2026-02-11 17:00 UTC, which has several binding CNECs and a DE–FR spread of 94 EUR/MWh.
- Coupled: the first MTU found in early February 2026 with no binding Core constraint and equal Core prices.

Expectations, with thresholds fixed before running:

- W1, domain form. On every presolved CNEC, the published net positions satisfy PTDF·NP ≤ RAM with a maximum
  violation of 5 MW. Binding CNECs (shadow price > 0) have slack of 5 MW or less. If not, look for LTA inclusion or
  the alpha factor before going on.
- W2, signs. The published shadow prices times the PTDF differences reproduce the observed Core zonal price spreads
  within 0.05 EUR/MWh, with one sign convention for every pair.
- W3, accounting. For each zone, the ENTSO-E net position equals the Core net position, plus the ALEGrO and link flows,
  plus the schedules with the rest of Europe, within 10 MW.
- W4, Jacobian tests:
  - for one price area, det(Pᵀ diag(S′) P) = ΣS′ up to a constant fixed by P;
  - for full separation, it equals ΠS′;
  - rank(P) + free net-position dimensions = 11 in both worked MTUs.
- W5, re-clear. With shocks recovered at any θ, re-clearing the eleven zones on the published domain reproduces the
  observed prices within 0.5 EUR/MWh and the zonal net positions within 20 MW. A miss shows a missing piece of the
  domain, not a wrong θ.

## External slope check, eleven-zone (written 2026-11-24, before the run)

Estimand. The model's dp_DE/dD_DE: re-clear with German demand +1 GW, holding neighbour demand, rest-of-Europe
exchanges and inferred rows at their recovered or observed values, on the replay domain. It is evaluated on the random
fit-sample MTUs, which represent the period, not the outage sample. It is weighted the same way as the two-zone check
(`cf.py`: squared residual German demand of the hour) and compared with the same empirical net slope: 2.38
[2.17, 2.60] overall, by daily-price tercile low 2.61, middle 2.10, high 2.37. Scarcity MTUs, those with a
re-cleared price above 4000, are masked and counted. The check does not feed back into the model.

Expectations and thresholds:
- E1, direction: the model is still steeper than the empirical slope. Rest-of-Europe exchanges and inferred rows are
  held, so imports can respond less than they did in the data. Met if empirical / model < 1.
- E2, improvement on the benchmark: the ratio is closer to 1 than the two-zone 0.67, because ten neighbour curves
  and the flow-based domain now carry part of the 1 GW. Met if the ratio is above 0.75. Material agreement (in
  [0.8, 1.25]) is not expected.
- E3, pattern: the gap is largest in the high-price tercile, as in the two-zone check, and larger in AHC and generic
  MTUs, where more flows are held.
- E4, sanity: on the outage MTUs, the demand-shock ΔpDE is within 25% of the outage ΔpDE at the median (a 1 GW
  outage and 1 GW of extra demand differ only through the gas floor cut and the fuel split). A larger gap points to a
  bug.

## Slope check with the cold-start interior-point solver, current curves (written 2026-11-24, before the run)

Same estimand, MTUs and draws as the first slope check. The only change is the solver: `solve11.solve_ip` with a 5 MW
balance tolerance, in place of the warm-start active set. That solver converged in 92.6% of MTUs, against 99.8% for
the interior-point solver on a 458-MTU test.
- S1: convergence of at least 98% of MTUs.
- S2: on MTUs where both solvers converged, the slopes agree within 0.05 EUR/MWh per GW at the 99th percentile.
- S3, direction: the solves that newly converge are mostly steep cases (curve ends, generic constraints). The
  weighted ratio therefore falls below 0.68. The size is not predicted.
- S4: baseline replay error in DE below 0.1 EUR/MWh in 99% of MTUs.

This is the reference against which the curve-end change is judged; the curve change is not tuned to it.
