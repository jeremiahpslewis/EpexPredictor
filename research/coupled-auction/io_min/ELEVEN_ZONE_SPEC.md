# Eleven-zone structural model of the day-ahead auction: specification and worked inversion

Status (2026-11-21): specified, with the inversion worked in one coupled and one congested MTU. Not yet fitted; the
outage counterfactual has not been run. The two-zone model in this folder (`model.py`, `run_fit.py`) is the benchmark.

## 1. Zones, time and information

- Zones: Z = {DE_LU, FR, NL, BE, AT, CZ, PL} (Core, modelled) and {DK1, DK2, SE4, NO2} (linked to Core by HVDC).
- Time unit: the auction's 15-minute MTU. The Core flow-based domain is published hourly and applies to the four
  quarter-hours of the hour.
- Firms offer before noon D-1. They know:
  - fuel and carbon prices of D-1;
  - their own availability, including outage notices filed before the gate;
  - the published load, wind and solar forecasts.

  They do not know the zonal shocks u. A price-taker offers marginal cost whatever it expects others to do. An outage
  announced before the auction changes the outage owner's offer (capacity removed) and nobody else's.

## 2. Supply

German supply, by fuel f ∈ {lignite, coal, gas}. Price-taking; heat rates logistic across units:

    S_f(p) = F_f + (α_f K_f − F_f) · G((p − c_f) / (σ_f c_f)),      G = logistic CDF
    c_L = a_L + h_L e_L τ,    c_C = a_C + h_C e_C τ,    c_G = h_G (P_g + e_G τ) + v_G
    F_G = φ0 + φ1 HDD,        F_L, F_C constant (must-run floor, GW)

K_f is the fleet less the unavailability published before noon D-1. α_f is the offered (committed) share.
e_f are IPCC emission factors. τ is the EUA price, P_g the THE gas price.

Each neighbour z has one curve, conditional on its own costs:

    S_z(p) = A_z + B_z · G((p − a_z − b_z L_t) / (σ_z L_t)),      L_t = 1.85 P_g + 0.37 τ

The parameters (log A_z, log B_z, a_z, b_z, log σ_z) are partially pooled across the ten neighbours (hierarchical
normal).

## 3. Network and clearing

Decision vector y per MTU:
- NP_z: the AC Core net positions of the seven modelled Core zones (JAO `hub_<z>`);
- a: the ALEGrO flow BE→DE (hub_ALBE = −a, hub_ALDE = +a);
- f_ℓ: the flows out of the Core zone on seven Core–Nordic links (DE–DK1, DE–DK2, DE–SE4, DE–NO2, NL–NO2, NL–DK1,
  PL–SE4). A link whose virtual-hub PTDF is published sits inside the flow-based domain (NP_VH = −f).

Zone balance, with n_z = H_z y the net export inside the eleven-zone system:

    S_z(p_z) − D_z − u_z − X_z = n_z = H_z y

- D_z: published residual load (load − wind − solar forecasts);
- X_z: exchanges held at their schedules (all rest-of-Europe borders, and the flows among the Nordic zones).

Constraints:

    flow-based:    Σ_h PTDF_kh NP_h(y) ≤ RAM_k − Σ_{o ∈ HR,HU,RO,SI,SK} PTDF_ko NP_o^obs     (presolved CNECs k)
    Core balance:  Σ_{z ∈ Core} NP_z − Σ_{AHC links} f_ℓ = − Σ_o NP_o^obs
    ALEGrO:        |a| ≤ 1000;     links outside the domain:  −L_ℓ^in ≤ f_ℓ ≤ L_ℓ^out

The clearing maximises welfare: min Σ_z C_z(n_z) subject to the constraints, where C_z'(n_z) = p_z. It is solved in
its dual, which has a closed form for logistic supply:

    min_{p, μ≥0, λ, ν}  Σ_z [ ∫S_z(p_z) dp − (D_z + u_z + X_z) p_z ] + μ'RAM + λ e0 + ν'limits
    s.t.  H'p + A'μ + e λ + (ν_hi − ν_lo) = 0

The ∫S terms are softplus functions, solved by cvxpy with Clarabel. Prices and zonal net positions are jointly
determined, and German imports are an output.

The constraint is the shared domain: all Core net positions consume the same RAM on every CNEC, so no border has a cap
of its own. German net import is determinate. Its split by border is not determinate in coupled MTUs, and is reported
only under a stated flow rule.

Held at published values, so the outage results are conditional on them:
- the net positions of the omitted Core hubs;
- the rest-of-Europe schedules;
- the flows among DK1, DK2, SE4 and NO2.

## 4. Observation model and inversion

Observed per MTU:
- the eleven prices;
- the seven Core AC net positions, the ALEGrO flow, the seven link schedules, the omitted hubs' net positions;
- the held schedules;
- the published binding set (shadow prices with their PTDF rows);
- German fuel outputs, with their own error as in the benchmark.

- **Shocks identified.** Each zonal shock comes from its own balance and its own net position:
  u_z = S_z(p_z) − D_z − X_z − n_z. When two zones clear at the same price, the price adds no information; the
  eleven net positions carry the information that separates the shocks. No zonal shock is integrated out.
- **What u_z contains.** The split of u_z into a demand part and a supply part is not identified.
- **Error in held exchanges.** An error in X_z goes into u_z and is held fixed in counterfactuals.
- **Regime and Jacobian.** The regime comes from published data only:
  - binding rows are those with a published shadow price;
  - ALEGrO is binding at its limit;
  - a link outside the domain is binding when the prices on its two sides differ by more than 0.01 EUR/MWh.

  Let V be the price subspace that the binding rows allow, V = {p : H'p ∈ span(A_B', e)}, with orthonormal basis Q.
  The outcome (p, n) lies on an 11-dimensional piece: dim V price coordinates plus 11 − dim V free net-position
  directions, and the two are orthogonal. The likelihood of the observed outcome is

      f_u(u(p, n; θ)) · det(Q' diag(S'(p; θ)) Q) · const(data)

  This is ΣS′ for one price area and ΠS′ under full separation. Because the regime comes from data, there is no
  regime-probability term.
- **Shocks integrated out or calibrated.** The network margin actually used in the allocation is not fully
  published: see 5.3. The rule below, fixed before fitting, calibrates it in place of a latent RAM error. Equivalently,
  RAM errors on non-binding rows are set to the smallest values consistent with the observed outcome.

## 5. Worked inversion: two MTUs (`worked.py`, results in `worked_results.json`)

### 5.1 The two MTUs

| | Congested: 2026-02-11 17:00 UTC | Coupled: 2026-02-01 03:00 UTC |
|---|---|---|
| Binding rows | 3: ALEGrO BE import (9.4), CREYS–St-Vulbas 400 kV (723.1), Resita–Timisoara (32.4) | none in Core |
| Price subspace + free n | 7 + 4 = 11 | 5 + 4 Nordic links = 5 + 6 = 11 |
| Core prices | DE 131.0, FR 37.0, NL 119.1, BE 108.3, AT 140.1, CZ 136.0, PL 135.4 | all 98.85 |
| Nordic prices | DK1 131.0, DK2 160.3, SE4 160.3, NO2 127.7 | DK1 88.5, DK2 96.0, SE4 65.8, NO2 95.9 |
| DE n_z (MW) | −1,354 | +4,708 |
| Links (MW, out of the Core zone) | DE→DK1 −1,824; DE→DK2 +890; DE→SE4 +209; NL→DK1 +700; NL→NO2 +640; PL→SE4 −157; ALEGrO BE→DE +1,000 | DE→DK1 −2,500; DE→DK2 −663; DE→SE4 −216; NL→DK1 −503; PL→SE4 −300; ALEGrO −921 |

How the inversion works in each MTU:
- **Coupled MTU.** The seven Core prices are a single number, so the seven Core shocks are pinned by the seven Core
  net positions. The four Nordic prices each differ from DE across a binding link.
- **Congested MTU.** Prices separate along the directions spanned by three binding rows and the binding links.
  Each shock still comes from its own zone's balance.

### 5.2 Checks (thresholds from `EXPECTATIONS_11ZONE.md`)

| Check | Congested | Coupled | Verdict |
|---|---|---|---|
| W1: published NPs satisfy all presolved CNECs (max violation ≤ 5 MW) | 0.005 MW | 74.6 MW on PST Meeden 2 (DE–NL), no shadow price | met / not met |
| W2: shadow prices × PTDF differences reproduce Core spreads (≤ 0.05) | 0.008 EUR/MWh, sign p_z − p_DE = −Σμ(PTDF_z − PTDF_DE) | trivially 0 | met |
| W3: zone accounting n_z + X_z = sum of border schedules | within 0.4 MW in every zone | within 0.2 MW | met |
| W3′: against ENTSO-E "net position" | off by the non-SDAC borders (CH, GB, …) | same | not met as written; that series has a different scope and is not used |
| W4: dims sum to 11, orthogonal, observed p ∈ V | 7+4, 1e-13, 0.007 | 5+6, 5e-16, 1e-13 | met |
| W4: unit tests ΣS′ / ΠS′ | exact | exact | met |
| W5: re-clear with recovered u (links held) | 0.008 EUR/MWh, 0.1 MW | 0.010 EUR/MWh, 0.2 MW | met |
| W5: re-clear, links endogenous, calibrated domain | 0.048 EUR/MWh, 0.3 MW | 0.003 EUR/MWh, 0.2 MW | met |
| W5: re-clear, published domain uncalibrated | 0.058 EUR/MWh | 5.3 EUR/MWh, 275 MW (the PST row) | not met in coupled MTU |

### 5.3 What the checks found about the published domain

Over 2–8 February 2026, the published flow-based domain matched the market outcome only in part:
- Of 1,827 binding non-external records, flow equals published RAM within 1 MW in only 46–52% (by quarter-hour);
  5–95% range −260 to +55 MW. The pattern is the same in every quarter of the hour, so it is not hourly-versus-
  quarter-hour publication.
- PST Meeden 2 binds in 155 MTUs that week, yet on 1 February it is violated by up to 168 MW in 15 of 24 hours, with
  no shadow price.

The calibration rule, fixed before fitting:
- rows with a published shadow price enter with their own PTDFs and RAM = observed flow;
- presolved rows violated by the published outcome have their RAM raised to the observed flow;
- a link outside the domain whose two sides differ in price has its limit set to the observed flow;
- any other link has limit max(observed flow, largest flow in that direction over the surrounding window).

In the counterfactual, how much margin is left on non-binding rows is a modelling assumption. It is reported as a
sensitivity: published RAM against calibrated RAM.

## 6. Open before fitting

- Fit the neighbours' curves (pooled) and the German block on quarter-hours. The German block does not yet mix on real
  data: in the two-zone benchmark, max r-hat is 22 and the chains split on which fuel carries the floor. The plan is a
  smoothed floor, initialisation at the MAP, four chains and a longer warmup.
- Bulk download of JAO for 2025-10 to 2026-11: about 117 presolved rows per hour, and queries are capped at two days.
- DE→NO2 (NordLink) is scheduled at 0 in both worked MTUs, against a mean of about +150 / −290 MW over the window.
  This needs checking against outage notices.
- Counterfactual: a 1 GW German gas outage, announced, re-cleared per posterior draw on a stratified MTU subset that is
  fixed before looking.
