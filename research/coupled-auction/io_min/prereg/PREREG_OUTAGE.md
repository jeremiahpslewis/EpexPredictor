# Pre-registration: 1 GW German gas outage, eleven zones (written 2026-11-21, before any outage result)

## Estimand

The effect of a 1 GW German gas outage, announced before the auction, on:
- the DE_LU price;
- German net imports;
- each zone's supply response ΔS_z.

The effect is conditional on four things held at their published schedules:
- all exchanges with the rest of Europe;
- flows among DK1, DK2, SE4 and NO2;
- the net positions of the omitted Core hubs (HR, HU, RO, SI, SK);
- the residual load forecasts.

Every number is model outage minus model baseline. Both are re-cleared with the same recovered shocks and under the
same headroom assumption.

Replacement shares come from supply responses, not border flows. Within the system, Σ_z ΔS_z = 1 GW minus any unserved
energy, and the split of German imports by border is not determinate in coupled MTUs. DE's own response is split by
fuel: lignite, coal, and the remaining gas (the outage removes 1 GW of the cheapest flexible gas, as in EXPECTATIONS.md).

## Sample

`outage_mtus.txt`, sha256 prefix 1791b1d500006f0b, 312 MTUs, drawn with seed 20261121 by the code in the commit that
adds this file:
- 13 months, October 2025 to October 2026;
- 4 random days per month;
- 6 fixed quarter-hours per day: 01:00, 05:00, 09:00, 13:00, 17:00 and 21:00 UTC.

No MTU is dropped after results are seen. MTUs with missing inputs are listed, with the reason, before any result.

## Network margin: replay and sensitivity (not marginalisation)

Two definitions, used throughout:
- **Verified binding row:** a row with a published shadow price whose published RAM equals the flow at the published
  net positions, within 1 MW.
- **Unverified binding row:** any other row with a shadow price.

**Replay ("calibration").** This is the case of ELEVEN_ZONE_SPEC.md §5.3:
- rows with a shadow price have RAM = observed flow;
- presolved rows violated by the observed outcome have their RAM raised to the observed flow;
- every other row keeps its published RAM.

It is labelled calibration, not integration.

**Headroom scenarios, η ∈ {0.10, 0.25, 0.50, 1.00}.** The same η applies to the baseline and the outage re-clear.
- Binding rows (verified and unverified): RAM = observed flow, as an equality.
- Non-binding presolved rows: RAM = flow_obs + η · max(RAM_pub − flow_obs, 0).
- Rows the published domain violates at the observed outcome: RAM = flow_obs.
- Links outside the domain: congested by the convention below → limit = observed flow in the congested direction.
  Otherwise limit = flow_obs + η · (largest flow in that direction over ±14 days − flow_obs).

η = 1.00 is the replay case. A claim to have integrated out the margin would need a distribution over η, or over
row-level headroom, and an actual marginalisation. That is not done here.

**Flags reported per MTU:**
- the published domain excludes the observed baseline (any presolved row violated by more than 5 MW);
- the number of unverified binding rows;
- outage re-clears with unserved energy, or with a price at the cap.

## Link congestion and loss convention

A link from a to b carries flow f > 0 with loss factor ℓ. Uncongested means p_a = (1 − ℓ) p_b; congested means
|p_a − (1 − ℓ) p_b| > 0.01 EUR/MWh. ℓ per link is estimated before any outage run:
- on MTUs from October and November 2025 only;
- as the median of 1 − p_a/p_b;
- over MTUs with |p_a|, |p_b| > 20 EUR/MWh, 0 < f < 0.9 × the largest flow in that direction over the period, and
  |1 − p_a/p_b| < 0.05.

A link whose estimate is within 0.002 of zero is treated as lossless. ℓ is held fixed afterwards. Losses enter the
clearing as (1 − ℓ) on the receiving side.

## Output: the first table

Rows: headroom scenario (replay, η = 0.10, 0.25, 0.50, 1.00). Columns, each as posterior mean with a 90% interval
across draws, and the median across MTUs:
- ΔpDE;
- Δ German net import;
- ΔS_z for each of the eleven zones, with DE split by fuel;
- the share of MTUs flagged.

Also reported separately: the same table for MTUs where gas is marginal against the rest.

**Reading rule, fixed now.** The answer changes materially across headroom if the median ΔpDE moves by more than 30%,
or Germany's own share of the replacement moves by more than 0.15, between η = 0.10 and η = 1.00. If it does, that
sensitivity is reported as a structural finding.
