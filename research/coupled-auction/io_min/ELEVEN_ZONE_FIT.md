# Pooled eleven-zone fit (2026-11-21)

Script `fit11.py`; records from `build11.py`.

## Setup

- **Sample:** 800 fit MTUs, drawn from the 2,000-MTU sample; MTUs with a missing link schedule excluded.
- **Chains:** two runs, each with two chains started at the medians of one two-zone basin.
- **Sampler:** 400 warmup and 400 draws per chain. No divergences.

## Convergence

The German gas block and all neighbour curves agree across all four chains. The lignite/coal split does not:
- run seeded in mode 0: max r-hat 1.54 (a_L);
- run seeded in mode 1: max r-hat 2.53 (σ_L), with its two chains disagreeing.

The lignite floor and dispersion lie on a ridge in the eleven-zone likelihood as well. The outage study therefore
draws across all four chains, so that this uncertainty is carried, not hidden.

## German block (chain medians)

| Start | F_L | F_C | gas floor φ0 | σ_L | σ_C | σ_G | α_G | h_G | μ_DE |
|---|---|---|---|---|---|---|---|---|---|
| mode 0, chain 0 | 0.91 | 0.78 | 0.33 | 0.55 | 0.10 | 0.23 | 0.90 | 2.49 | −2.54 |
| mode 0, chain 1 | 0.96 | 0.77 | 0.35 | 0.58 | 0.10 | 0.23 | 0.89 | 2.49 | −2.55 |
| mode 1, chain 0 | 1.36 | 0.34 | 0.33 | 0.35 | 0.48 | 0.23 | 0.89 | 2.49 | −2.59 |
| mode 1, chain 1 | 2.20 | 0.23 | 0.34 | 0.15 | 0.56 | 0.23 | 0.88 | 2.48 | −2.61 |

Compared with the two-zone benchmark:
- the gas parameters move to one point from both starts;
- the gas floor is low (as mode 1 had), consistent with the 1.6 GW observed gas output at negative prices;
- the mean gas heat rate is 2.49, a plausible engineering value, against 1.5–1.7 in the two-zone fit.

## Neighbours (posterior medians)

Order: FR, NL, BE, AT, CZ, PL, DK1, DK2, SE4, NO2. The supply curve is S = s̄(α + β·G((p − a − b·Lc) / (σ·Lc))).

| | FR | NL | BE | AT | CZ | PL | DK1 | DK2 | SE4 | NO2 |
|---|---|---|---|---|---|---|---|---|---|---|
| α | −0.40 | 0.13 | −1.77 | −0.19 | 0.23 | −0.10 | −12.7 | −5.3 | −15.5 | −3.4 |
| β | 2.86 | 1.90 | 6.54 | 2.74 | 1.47 | 2.02 | 23.6 | 9.9 | 36.5 | 9.8 |
| a | −70.5 | 3.9 | −32.0 | 21.6 | 27.4 | 53.1 | 3.1 | 12.7 | 35.2 | −20.2 |
| b | 1.33 | 0.96 | 1.41 | 1.03 | 0.69 | 0.44 | 0.56 | 0.45 | 0.56 | 1.12 |
| σ | 1.22 | 0.35 | 0.69 | 0.45 | 0.58 | 0.52 | 0.80 | 0.88 | 0.83 | 0.30 |

The four Nordic zones have small mean net supply needs and large swings. Their large β with negative α is a
reduced-form fit of a steep curve around a small mean; their parameters do not have a physical reading.

## Performance note

The first attempt ran on 2,000 MTUs with two processes and was stopped after 3 hours without output: thread
oversubscription gave a load of 12 on 4 cores. With one thread per chain, the 800-MTU fit takes 25 minutes.
