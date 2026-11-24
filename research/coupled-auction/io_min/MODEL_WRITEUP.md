# A minimal structural model of the coupled day-ahead auction: equations, identification, one counterfactual

Status (2026-11-24). This file consolidates the eleven-zone model as fitted and used. Detail is in:
- `ELEVEN_ZONE_SPEC.md`: specification and worked inversion;
- `ELEVEN_ZONE_FIT.md`: the pooled fit;
- `RESULTS_OUTAGE_11ZONE.md`: the outage study;
- `BENCHMARK_RESULTS.md`: the two-zone benchmark;
- `prereg/PREREG_OUTAGE.md`: the pre-registration and its three addenda.

Everything below is price-taking unless it is labelled strategic (section 10). The strategic part is analytic and not
estimated.

## 1. Market, time and information

Eleven zones are modelled: DE_LU, the six other Core zones FR, NL, BE, AT, CZ, PL, and the four Nordic zones linked to
Core by HVDC, DK1, DK2, SE4, NO2. The time unit is the 15-minute MTU of the single day-ahead coupling (SDAC). The
sample runs from 2025-10-01 to 2026-11-08.

Firms submit offers before 12:00 on D−1. When they offer, they know:
- the D−1 fuel and carbon prices: THE gas P_g and EUA τ;
- their own availability, including every unavailability notice published before the gate;
- the published load, wind and solar forecasts.

They do not know the zonal shocks u_t. A price-taker offers its marginal cost whatever it expects the shocks to be. An
outage announced before the gate therefore changes the owner's offer, by removing the capacity, and nobody else's.

## 2. Supply

**Germany, by fuel f ∈ {L (lignite), C (hard coal), G (gas)}**, with heat rates dispersed logistically across units:

    S_f(p) = F̃_f + (α_f K_f − F̃_f) · G((p − c_f) / (σ_f c_f)),       G(z) = 1 / (1 + e^(−z))
    F̃_f = −s · log(e^(−F_f / s) + e^(−α_f K_f / s))                     (smoothed min(F_f, α_f K_f), s = 0.05 GW)
    c_L = a_L + h_L e_L τ,   c_C = a_C + h_C e_C τ,   c_G = h_G (P_g + e_G τ) + v_G
    F_G = φ0 + φ1 · HDD_t

The terms are:
- K_f: installed capacity less the unavailability published before the gate (the retained operating constraint: only
  capacity known to be available can be offered);
- α_f: the offered share of that capacity;
- F_f: a must-run floor (CHP, system services, minimum-load commitments), which for gas rises with heating degree
  days;
- c_f: the marginal cost of the median unit, built from heat rate × fuel price plus emission factor × carbon price.

German supply is S_DE = S_L + S_C + S_G. Everything else in Germany (renewables, nuclear-free remainder, hydro,
biomass, storage) enters through the published residual load D_DE and the shock u_DE.

**Neighbours.** One curve per zone z, conditional on its own fuel costs:

    S_z(p) = s̄_z [α_z + β_z G((p − a_z − b_z L_t) / (σ_z L_t))],      L_t = 1.85 P_g + 0.37 τ

The terms are:
- s̄_z: the zone's mean supply need in the fit sample (a scale constant);
- L_t: a gas-and-carbon cost index;
- the parameters (α_z, log β_z, a_z, b_z, log σ_z): partially pooled across the ten neighbours, each with a normal
  hyperprior on its location and scale.

**Fixed constants (not estimated).**
- Emission factors: e_L = 0.364, e_C = 0.341, e_G = 0.202 tCO₂/MWh_th.
- Gas variable cost: v_G = 3 EUR/MWh.
- Floor smoothing: s = 0.05 GW.
- Cost index coefficients: 1.85 and 0.37 in L_t.
- Link loss factors: DE–SE4 0.025, DE–NO2 0.031, NL–NO2 0.032, SE4→PL 0.011, others 0. These were estimated
  separately from schedule and flow data and fixed in prereg addendum 1.
- Fuel-likelihood tempering: KQ = 4.

## 3. Network and joint determination of dispatch, exchange and prices

**Decision vector y ∈ R²²** per MTU, with three parts:
- the seven Core AC net positions;
- the ALEGrO flow a (BE→DE);
- for each of the seven Core–Nordic HVDC links ℓ, both directions f⁺_ℓ, f⁻_ℓ ≥ 0, as received quantities. The
  sender ships f / (1 − loss).

Under advanced hybrid coupling (from mid-June 2026), a link's virtual hub sits in the flow-based domain:

    NP_VH = −f⁺/(1 − ℓ_ab) + f⁻

and the Core-side leg is inside the Core zone's AC net position.

**Zone balance.** n = H y is the net export inside the system, and X_z holds the exchanges with the rest of Europe and
among the Nordic zones at their schedules:

    S_z(p_z) − D_z − u_z − X_z = n_z = (H y)_z,        z = 1..11

**Constraints:**

    flow-based   A y ≤ b,   b_k = RAM_k − Σ_{o ∈ HR,HU,RO,SI,SK and other non-modelled hubs} PTDF_ko · NP_o^obs
    Core balance e'y = e0       (Σ Core NPs − Σ AHC virtual-hub legs = − Σ_o NP_o^obs)
    bounds       |a| ≤ 1000 MW;  0 ≤ f^±_ℓ ≤ L^±_ℓ   (links outside the domain)

**Clearing** maximises gross surplus. Its KKT system determines everything jointly:

    S(p) − D − u − X = H y                              (balance)
    H'p + A_act' μ + e λ + E_b' ν = 0,   μ ≥ 0          (prices are the duals: spreads only where a constraint binds)
    A_act y = b_act,  e'y = e0,  bounds active          (complementary slackness)

Zonal prices p, net positions n, the flows y and German dispatch by fuel S_f(p_DE) are one fixed point. German net
imports are −n_DE. A German shock moves p_DE, which moves every coupled zone's price, supply and net position along
the directions the binding constraints leave free. The split of German imports by border is not determinate in coupled
MTUs. Only the net position is.

The equilibrium is solved exactly by active-set Newton (`net11.polish`). A conic dual (Clarabel) supplies the start
when needed. A variable held at a value is an equality with a multiplier of either sign.

## 4. Observation model

Observed per MTU:
- the eleven prices;
- the Core net positions, ALEGrO, the link quantities (JAO virtual-hub positions under hybrid coupling, ENTSO-E
  schedules otherwise), the omitted hubs' net positions and the rest-of-Europe schedules;
- the published binding set (shadow prices with their own PTDF rows);
- German generation by fuel.

**Prices and positions.** The model is exactly invertible:

    u_z = S_z(p_z; θ) − D_z − X_z − n_z

The regime (which rows bind) is fixed by the data and does not depend on θ. It comes from the published shadow
prices, plus constraints that bind but are unpublished. Those are inferred from observed prices and flows by a
targeted search (link capacities, ALEGrO, the PL and BE allocation constraints, a generic residual direction as a
last resort). The outcome lies on an 11-dimensional piece made of dim V price coordinates, where
V = {p : H'p ∈ span(A_B', e)} with orthonormal basis Q, and 11 − dim V free net-position directions. The density of
the observed outcome on that piece is the joint change of variables, with no regime-conditioning term:

    f(p, n | θ) = MVT_11(u − μ; ν, diag(sd) Ω diag(sd)) · det(Q' diag(S'(p; θ)) Q)

The determinant reduces to ΣS′ for a single price area and to ΠS′ under full separation. Normalisation on the regime
support was checked on simulated mixed-regime data (`mixed_regime.py`: the pieces integrate to 1 and match regime
frequencies; 19 of 22 parameters recovered in the eleven-zone simulate-and-recover).

**Fuel outputs (measurement equation).** Realised generation equals the curve at the German price, plus a loading on
the after-auction residual-load surprise gap_t (realised minus forecast residual load), plus Student-t error:

    q_f,t = S_f(p_DE,t) + β_q,f gap_t + ε_f,t,      ε ~ t(ν_q, 0, sd_q,f),     log-likelihood weighted by 1/KQ

The tempering accounts for strong serial dependence of the errors, which a factorised likelihood would otherwise count
as independent evidence.

## 5. Free parameters and what identifies each

German block (15):
- **α_L, α_C, α_G (offered shares).** Identified by how far fuel output saturates in high-price MTUs relative to
  available capacity K_f, which varies with published unavailability.
- **a_L, h_L (lignite cost intercept and heat rate).** The intercept is identified by the price level at which
  lignite output starts to respond. The heat rate is identified by how that threshold moves with the EUA price
  τ (e_L is fixed).
- **a_C, h_C (coal).** Identified the same way, from coal output against price and τ.
- **h_G (gas heat rate).** Identified by how the price at which gas output responds moves with P_g + e_G τ. Gas and
  carbon prices move separately over the sample.
- **σ_L, σ_C, σ_G (cost dispersion).** Identified by how gradually each fuel's output rises with price, i.e. the
  slope of q_f against p_DE around c_f.
- **F_L, F_C (must-run floors).** Identified by lignite and coal output in low and negative-price MTUs.
- **φ0, φ1 (gas floor, HDD loading).** Identified by gas output at low prices on warm against cold days.

Shocks (11 + 55 + 1 + 1):
- **μ_DE.** Identified by the mean German balance residual; the neighbour means are fixed at zero, so s̄_z α_z
  absorbs them.
- **sd_u (11).** Identified by the dispersion of each zone's balance residual.
- **Ω (LKJ correlation).** Identified by co-movement of zonal residuals across MTUs, e.g. common weather.
- **ν.** Identified by the frequency of large residuals, the tails.

Neighbours (5 × 10 zone parameters, 10 hyperparameters):
- **a_z, b_z.** Identified by how the zone's price at a given net supply moves with its cost index L_t, the level and
  the loading.
- **σ_z.** Identified by the slope of the zone's net position against its own price, holding L_t.
- **α_z, β_z.** Identified by the floor and range of the zone's supply relative to s̄_z, at low and high own prices.
- **Pooling hyperparameters.** Identified by the spread of these across zones.

Fuel measurement (3 + 3 + 1):
- **sd_q,f.** Identified by the residual spread of each fuel's output around its curve.
- **β_q,f.** Identified by co-movement of fuel output with the after-auction residual-load surprise.
- **ν_q.** Identified by the tails of those residuals.

## 6. The most confounded parameters: one lignite/coal ridge

Correlations in the pooled posterior draws (four chains, two seeded in each two-zone basin) pick them out:

| Pair | Pooled correlation | Within-chain correlation |
|---|---|---|
| σ_C – α_C | +0.95 | +0.79 |
| σ_L – F_L | −0.92 | −0.80 |
| σ_L – σ_C | −0.92 | −0.70 |

Pooled correlations mix in the seeding of chains in different basins. The within-chain correlations are the cleaner
guide, and they lead with σ_L–F_L (−0.80) and σ_C–α_C (+0.79). The two pairs belong to one ridge over (σ_L, F_L, σ_C,
α_C). The two most confounded parameters are the lignite dispersion and floor (σ_L, F_L), and their coal counterparts
move with them. Along it, the model reassigns the flexible thermal
response between lignite and coal:
- "flexible lignite, rigid coal": σ_L ≈ 0.55–0.6, σ_C ≈ 0.1, F_L ≈ 0.9;
- "rigid lignite, flexible coal": σ_L ≈ 0.1–0.2, σ_C ≈ 0.5, F_L ≈ 2.0–2.3.

The price system sees only the sum, S_L + S_C. The fuel observations pull in opposite directions (BENCHMARK_RESULTS.md).

Their posterior (chain medians):

| Start | σ_L | σ_C | F_L | F_C |
|---|---|---|---|---|
| mode 0, chain 0 | 0.55 | 0.10 | 0.91 | 0.78 |
| mode 0, chain 1 | 0.58 | 0.10 | 0.96 | 0.77 |
| mode 1, chain 0 | 0.35 | 0.48 | 1.36 | 0.34 |
| mode 1, chain 1 | 0.15 | 0.56 | 2.20 | 0.23 |

The seeded chains do not mix (r-hat 1.54 and 2.53). The pooled draws are therefore not a posterior over the ridge.
The relative weight of the two ends is not estimated, and results are reported conditional on the start.

**Counterfactual along the ridge** (each of the 12 outage-study draws, balanced panel, replay; means over MTUs):

| Draw | σ_L | F_L | σ_C | α_C | ΔpDE | imports | Δlignite | Δcoal | Δgas (other units) |
|---|---|---|---|---|---|---|---|---|---|
| mode0 d0 | 0.58 | 0.76 | 0.12 | 0.45 | 2.94 | 0.54 | 0.10 | 0.07 | 0.29 |
| mode0 d159 | 0.51 | 0.97 | 0.08 | 0.44 | 3.00 | 0.55 | 0.12 | 0.06 | 0.27 |
| mode0 d319 | 0.58 | 0.85 | 0.12 | 0.46 | 2.67 | 0.53 | 0.10 | 0.07 | 0.30 |
| mode0 d479 | 0.55 | 0.54 | 0.10 | 0.45 | 2.86 | 0.54 | 0.12 | 0.07 | 0.27 |
| mode0 d639 | 0.59 | 0.77 | 0.07 | 0.45 | 2.73 | 0.54 | 0.10 | 0.07 | 0.30 |
| mode0 d799 | 0.61 | 0.92 | 0.08 | 0.41 | 2.71 | 0.54 | 0.10 | 0.06 | 0.31 |
| mode1 d0 | 0.17 | 2.16 | 0.48 | 0.54 | 3.08 | 0.54 | 0.10 | 0.07 | 0.29 |
| mode1 d159 | 0.11 | 2.27 | 0.48 | 0.56 | 3.20 | 0.56 | 0.11 | 0.07 | 0.27 |
| mode1 d319 | 0.59 | 1.14 | 0.17 | 0.45 | 2.78 | 0.53 | 0.11 | 0.06 | 0.30 |
| mode1 d479 | 0.13 | 1.92 | 0.57 | 0.55 | 2.97 | 0.56 | 0.11 | 0.06 | 0.27 |
| mode1 d639 | 0.16 | 2.07 | 0.51 | 0.56 | 2.86 | 0.54 | 0.11 | 0.06 | 0.29 |
| mode1 d799 | 0.15 | 2.06 | 0.49 | 0.55 | 2.74 | 0.54 | 0.10 | 0.06 | 0.29 |

The shares in this table use the pre-registered booking (1 GW removed). Section 7 gives them per GW actually
removed.

Reading the table:
- **Draw placement.** Mode 1 chain 0 visited the "flexible lignite" end at draw 319, so the 12 draws cover both
  ends.
- **Price and imports.** The mean ΔpDE is 2.97 at the "rigid lignite" end and 2.81 at the other. That difference
  is smaller than the spread within each end (0.3–0.5), so the two ends cannot be distinguished on price. Imports
  hardly differ either.
- **Fuel split.** The mean fuel split of the German response hardly moves (lignite 0.10–0.12, coal 0.06–0.07).
  What moves is the distribution across MTUs: at the median, the "flexible lignite" end has lignite 0.09 and coal
  0.01, the "rigid lignite" end 0.05 and 0.05. The ridge matters for the fuel split hour by hour, and little for the
  price or imports.

**Alternative priors.** No eleven-zone refit under alternative priors has been run. In the two-zone benchmark, the
two ends (basins) give an outage ΔpDE of +2.77 and +3.45. In the eleven-zone model the two ends cannot be distinguished on price. A prior-sensitivity
refit (a tight prior on σ_L at each end) is about 25 minutes per fit and needs its own approval. It remains open.

## 7. The counterfactual: 1 GW of German gas, announced before the auction

**Design (pre-registered).**
- Sample: 312 MTUs fixed in advance (seed 20261121).
- Per posterior draw (6 per basin): recover u_t from the observed outcome; hold u_t, all held exchanges, the omitted
  hubs and the inferred constraints fixed; remove 1 GW of the cheapest available flexible gas; re-clear all eleven
  zones exactly. Report outage minus re-cleared baseline.
- Network headroom on non-binding rows: replay of the published (calibrated) domain, and η ∈ {0.10, 0.25, 0.50, 1.00}
  of the published headroom. The same domain is used in baseline and outage.

**Gas actually removed.** The outage removes at most the flexible gas running above its floor. In 17.8% of balanced
draw-MTUs that is less than 1 GW, and the mean removed at replay is 0.89 GW. The pre-registered table books other gas
as ΔG + 1, so it counts the gas not removed as German replacement. It therefore overstates Germany's own share and
understates imports. ΔpDE is unaffected as a number, but it is a response to less than 1 GW in those MTUs.
`fullcut11.py` recomputes everything on the MTUs where the full 1 GW is removed in every scenario.

**Result per GW actually removed** (full-cut panel, 2,428 draw-MTUs, mean over MTUs):

| Headroom | ΔpDE mean (median) | German net imports | DE own (lignite / coal / other gas) |
|---|---|---|---|
| replay = η 1.00 | +3.04 / +3.10 (+2.24 / +2.25) | 0.61 | 0.39 (0.11 / 0.07 / 0.21) |
| η = 0.10 | +3.73 / +3.83 (+2.81 / +2.85) | 0.53 | 0.47 (0.13 / 0.08–0.09 / 0.25) |

Each pair is mode 0 / mode 1. Replacement by zone at replay, per GW removed: NL 0.14, BE 0.09, DK1 0.08, PL 0.07,
CZ 0.06, AT 0.05, NO2 0.05, FR 0.04, DK2 0.02, SE4 0.004. The reading rule on this panel: median +26% in both basins,
own share +0.08. Not material.

**Pre-registered table** (balanced panel, 79% of draw-MTUs, 1 GW booked as removed; range across draws):

| Headroom | ΔpDE (EUR/MWh) | German net imports (GW per GW) | DE own (lignite / coal / other gas) |
|---|---|---|---|
| replay = η 1.00 | +2.82 to +2.94 [2.68, 3.17] (median +2.13 to +2.16) | 0.54–0.55 | 0.46 (0.11 / 0.06–0.07 / 0.29) |
| η = 0.10 | +3.42 to +3.58 [3.24, 3.83] (median +2.63 to +2.71) | 0.47–0.48 | 0.52–0.53 |

The ranges span both basins and their draws.

**Pre-registered reading rule** (material if the median ΔpDE changes by more than 30% or the German own share by more
than 0.15 between η = 0.10 and 1.00): not met in either basin.
- Pooled: +23% (mode 0) and +26% (mode 1).
- Per draw: +21% to +29%.
- Bounding the η = 0.10 failures: +22%.

Network headroom therefore does not change the answer materially. Larger effects come from the subgroups:
- hybrid-coupling MTUs: +3.3 to +3.5 vs +2.7;
- MTUs where the published domain excludes the observed outcome: +4.2 vs +2.2;
- MTUs with a generic inferred constraint, where imports are held mechanically and Germany covers 0.65–0.70 itself.

**Against the two-zone benchmark:** +2.77 and +3.45 by basin. In the eleven-zone model:
- the basins nearly agree;
- imports replace 0.61 GW per GW actually removed, against 0.58 and 0.74 in the two-zone basins;
- the price effect sits at the lower two-zone value.

## 8. What the results are conditional on

- **Held quantities.** Rest-of-Europe schedules, flows among the Nordic zones and the omitted Core hubs are held at
  observed values. So are inferred binding constraints, both in baseline and outage.
- **Price-taking.** Every firm is a price-taker (section 10 shows what a strategic offer would add).
- **The ridge.** The lignite/coal split is a ridge. The pooled draws are conditional on the chain starts.
- **Solver failures.** At η = 0.10, 48% of hybrid-coupling MTUs do not converge. They are under-represented in the
  balanced panel; the bound above shows that the verdict does not depend on them.

## 9. External check: the weather-response slope

Expectations and thresholds are in `EXPECTATIONS_11ZONE.md`, written before the run; the script is `slope11.py`.

**Estimand.** The model's dp_DE/dD_DE: German demand +1 GW, everything else held, replay domain. It is computed on
1,829 random fit-sample MTUs and 8 posterior draws (four per basin, both chains of each). It is weighted by the squared
residual German demand of the hour, as in the empirical regression. 92.6% of the solves converged, and 3 MTUs reached
the price cap and were masked.

| Subset | Model, weighted mean (range across draws) | Model, median MTU | Empirical [90% band] | Empirical / model |
|---|---|---|---|---|
| All | 3.50 (3.37–3.69) | 2.33 | 2.38 [2.17, 2.60] | 0.68 |
| Low-price days | 3.84 | 2.44 | 2.61 | 0.68 |
| Middle | 2.81 | 2.12 | 2.10 | 0.75 |
| High-price days | 3.77 | 2.57 | 2.37 | 0.63 |
| Hybrid coupling | 3.59 | 2.86 | 2.38 | 0.66 |
| Generic inferred constraint | 5.49 | 3.89 | 2.38 | 0.43 |

Of each 1 GW of extra German demand, 0.65 GW is met by imports.

Verdicts:
- **E1 (model steeper): met.**
- **E2 (closer than the two-zone 0.67): not met.** The weighted ratio is the same as the benchmark's.
- **E3 (pattern): met** for high prices and generic MTUs; hybrid-coupling MTUs differ only marginally (0.66 against
  0.69 elsewhere).
- **E4 (sanity): met.** On the outage MTUs, the median demand-shock ΔpDE is within 16% of the outage ΔpDE, and
  identical in most MTUs, as it should be when the gas cut does not hit the floor.

**Reading.** The pre-specified comparison is the weighted mean against the regression coefficient: 3.50 against
2.38. On that comparison the model is too steep by about a half, as the two-zone benchmark was. The median MTU slope,
2.33, is descriptive only, since the regression coefficient is itself a weighted mean. The weighted mean is pulled up
by a heavy tail:
- the 90th percentile is 5.7 and the 99th 18;
- the top 5% of MTUs carry a quarter of the mean.

Where the steepest 5% (slope above 7.9) sit:

| | Share of MTUs | Share of the steepest 5% |
|---|---|---|
| Price ≤ 0 | 7% | 25% |
| Price 0–40 | 6% | 22% |
| Price > 200 | 7% | 34% |
| Generic inferred constraint | 9% | 34% |
| Hybrid coupling | 32% | 57% |

The tail sits at the two ends of the supply curves: near the floors at low and negative prices, where the logistic
curves are flat, and near scarcity. It is also over-represented in MTUs with a generic inferred constraint, where
imports are held. Held rest-of-Europe exchanges are therefore not the main suspect; the curve ends are.

Solves failed in 7% of MTUs (26% of generic MTUs, 21–22% at prices ≤ 0 or > 200). In the outage study failures were
the steep cases, so 3.50 is probably a lower bound on the model's weighted slope.

The outage means (+3.0 to +3.1 EUR/MWh per GW removed) are therefore probably overstated through the same tail. The
check did not feed back into the model.

## 10. Strategic bidding (analytic extension; not estimated)

**Who and what.** Firm j holds a portfolio given by a share of each German fuel curve, capacities κ_jf·α_f K_f,
and a forward position f_j (GW sold before the auction). The model contains no ownership or forward data, so the
shares are parametric and nothing here is estimated.

**Objective.** The firm offers a supply function before u is known, maximising expected spot profit:

    max_{b_j(·)}  E_u [ p(u) · (q_j(u) − f_j) − C_j(q_j(u)) ]

Here p(u) clears the eleven-zone market given every rival's price-taking curve and the network.

**Offer.** At each realisation, ex post optimality (the supply-function-equilibrium condition) gives

    b_j(q) = C_j'(q) + (q − f_j) · ∂p/∂q_j,      ∂p/∂q_j = (dp/dD) / (1 − κ_j)

Here κ_j = S'_j / S'_total is the firm's share of the market-wide supply slope, including every coupled neighbour
reached through the network. dp/dD = 1/S'_total is the slope from section 9. The residual demand the firm faces
excludes its own units and includes the import response, so the coupling enters through dp/dD.

The offer shifts depending on the forward position:
- **net long (q > f_j):** the offer moves up by the markup;
- **fully hedged (q = f_j):** no markup;
- **net short (q < f_j):** the offer moves below cost (Allaz–Vila).

**Size** (EUR/MWh; markup = (q − f_j) · (dp/dD) / (1 − κ_j), at the empirical dp/dD = 2.38 and, in brackets, the
model's weighted 3.50):

| Uncontracted output q − f_j | κ_j = 0.05 | κ_j = 0.15 | κ_j = 0.30 |
|---|---|---|---|
| −1 GW | −2.5 (−3.7) | −2.8 (−4.1) | −3.4 (−5.0) |
| +1 GW | +2.5 (+3.7) | +2.8 (+4.1) | +3.4 (+5.0) |
| +2 GW | +5.0 (+7.4) | +5.6 (+8.2) | +6.8 (+10.0) |
| +4 GW | +10.0 (+14.7) | +11.2 (+16.5) | +13.6 (+20.0) |

For scale, the outage study puts Germany's own share of the market-wide response at about 0.46. A firm controlling a
third of the German flexible thermal slope therefore has κ_j ≈ 0.15.

**Effect on the outage counterfactual.**
- **The outage owner, if net long,** loses output and so markup on its remaining units. It also gains on them from
  the higher price.
- **Rivals** each increase output by their share of the 1 GW. If they are net long, their markup rises by
  Δq_k · (dp/dD) / (1 − κ_k). That is about +0.4 EUR/MWh on its offer curve for a firm with κ = 0.15 that picks up
  0.15 GW.

The markup formula above is for one strategic firm facing a price-taking fringe. With several strategic firms, each
faces rivals' offer slopes rather than their cost slopes. The residual demand is then steeper and markups larger (a
supply-function equilibrium), so the rival-markup figure is indicative only. Strategic rivals amplify the
price-taking ΔpDE. Hedged rivals (q = f) do not.

**Why it is not estimated here.** The cost parameters were fitted under price-taking. Any average markup is already
absorbed in a_f, h_f and α_f, so adding the markup on top of the fitted curve would double-count it. Estimating
conduct needs ownership shares by unit and some measure of forward positions. With those, the markup term enters the
offer curve, and the cost parameters are re-identified from variation in forward cover and in κ_j, e.g. across
outages of rivals' units.

## 11. Open

- **Tight-headroom solves.** Hybrid-coupling MTUs at η = 0.10 still fail to converge 48% of the time. A solver that
  handles very tight domains (an interior-point method on the KKT system) would close this.
- **The steep tail** found by the external check sits at the ends of the supply curves (floors and scarcity) and
  in MTUs with generic inferred constraints. A richer curve shape at the ends, such as a mixture or a separate
  scarcity segment, is the first thing to try.
- **The outage counterfactual under alternative priors.** It needs two eleven-zone refits (a tight prior at each end
  of the ridge, about 25 minutes each) plus outage re-runs on the v2 records (about 70 minutes each). Each needs
  approval.
- **Strategic conduct.** It needs unit ownership and forward-cover data (section 10).
- **Spec update.** `ELEVEN_ZONE_SPEC.md` still describes the link legs under hybrid coupling as schedules, and its
  section 6 is the pre-fit plan. This write-up supersedes both.
