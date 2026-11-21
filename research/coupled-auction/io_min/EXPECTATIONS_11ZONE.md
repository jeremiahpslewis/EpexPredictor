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
