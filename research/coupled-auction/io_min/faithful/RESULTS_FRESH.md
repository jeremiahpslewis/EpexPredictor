# out_fresh_strict_rlmatched vs out_fresh_strict

Days (18): 2026-11-09, 2026-11-10, 2026-11-11, 2026-11-12, 2026-11-13, 2026-11-14, 2026-11-15, 2026-11-16, 2026-11-17, 2026-11-18, 2026-11-19, 2026-11-20, 2026-11-21, 2026-11-22, 2026-11-23, 2026-11-24, 2026-11-25, 2026-11-26. Differences are variant minus base; negative = variant better. Intervals: paired circular 7-day block bootstrap, 90 %.

## Price MAE

| member | base | variant | variant − base |
|---|---|---|---|
| trees | 2.205 | 2.218 | +0.012 [-0.096, +0.106] |
| Timestral | 2.316 | 2.274 | -0.043 [-0.075, -0.013] |
| market route | 2.362 | 2.315 | -0.046 [-0.093, -0.004] |
| issued blend | 2.173 | 2.141 | -0.031 [-0.088, +0.015] |

Per day (base → variant): 2026-11-09: trees 5.856→6.011, timestral 6.897→7.011, market 6.479→6.629, issued 6.292→6.430; 2026-11-10: trees 4.265→4.238, timestral 4.415→4.288, market 4.221→4.438, issued 4.107→4.206; 2026-11-11: trees 2.682→2.866, timestral 3.283→3.460, market 3.346→3.504, issued 3.006→3.170; 2026-11-12: trees 2.644→3.162, timestral 1.711→1.496, market 1.329→1.164, issued 1.449→1.309; 2026-11-13: trees 1.492→1.784, timestral 1.456→1.523, market 1.585→1.412, issued 1.389→1.443; 2026-11-14: trees 0.964→1.010, timestral 1.417→1.380, market 1.679→1.637, issued 1.286→1.277; 2026-11-15: trees 2.005→1.853, timestral 2.245→2.360, market 2.450→2.474, issued 2.047→2.160; 2026-11-16: trees 3.079→2.787, timestral 4.060→3.970, market 3.651→3.587, issued 3.596→3.447; 2026-11-17: trees 1.687→1.642, timestral 1.284→1.153, market 2.124→2.021, issued 1.673→1.580; 2026-11-18: trees 0.957→1.069, timestral 1.234→1.339, market 1.015→0.978, issued 0.967→1.016; 2026-11-19: trees 1.215→1.549, timestral 1.673→1.552, market 1.552→1.704, issued 1.193→1.346; 2026-11-20: trees 1.754→1.806, timestral 1.337→1.331, market 0.668→0.601, issued 1.175→1.195; 2026-11-21: trees 0.733→0.936, timestral 0.601→0.610, market 1.331→0.962, issued 0.779→0.640; 2026-11-22: trees 1.331→1.540, timestral 0.521→0.612, market 0.549→0.696, issued 0.667→0.836; 2026-11-23: trees 1.321→0.970, timestral 2.064→1.722, market 1.574→1.239, issued 1.525→1.095; 2026-11-24: trees 1.537→1.546, timestral 1.378→1.339, market 2.758→2.796, issued 1.825→1.832; 2026-11-25: trees 2.430→2.060, timestral 3.215→3.071, market 3.185→3.048, issued 2.942→2.725; 2026-11-26: trees 3.746→3.089, timestral 2.905→2.708, market 3.019→2.787, issued 3.187→2.835

## Residual load on D (GW)

Split by the variant's grid vintage per slot (`rlc_k`).

| series | slots | base MAE | variant MAE | diff | base bias | variant bias |
|---|---|---|---|---|---|---|
| tree RL | all | 2.45 | 2.15 | -0.299 [-0.640, +0.083] | -1.81 | -0.83 |
| tree RL | d1 | 1.59 | 1.59 | +0.000 [-0.000, +0.000] | -0.83 | -0.83 |
| tree RL | d2+ | 2.75 | 2.34 | -0.405 [-0.859, +0.118] | -2.16 | -0.83 |
| Timestral-mixed RL | all but last 3 | 1.95 | 1.85 | -0.103 [-0.215, +0.031] | | |

## Model sets on d2+ slots (stored, not all fed)

| model set | RL MAE | RL bias | wind bias | solar bias | load bias | wind MAE | solar MAE | load MAE |
|---|---|---|---|---|---|---|---|---|
| app (cutoff-trained) | 2.75 | -2.16 | +1.44 | -0.20 | -0.92 | 2.31 | 0.77 | 1.38 |
| matched (d2 weather, from 2025-10-01) | 2.34 | -0.83 | +0.02 | -0.10 | -0.91 | 1.66 | 0.75 | 1.44 |
| control (app weather, matched rows) | 2.90 | -2.22 | +1.58 | -0.18 | -0.83 | 2.33 | 0.72 | 1.35 |

## Load forecast on D, app-weather forms (GW, all slots)

| form | load MAE | load bias |
|---|---|---|
| app load, uncorrected | 1.34 | -0.77 |
| held-out load, uncorrected | 1.66 | -1.09 |
| PRIMARY held-out + own residual (window to gate − latency) | 1.58 | -0.97 |
| held-out + own residual, study window (to cutoff) | 1.36 | -0.50 |
| diagnostic: app load + held-out residual (gate window) | 1.34 | -0.65 |

Residuals per day (GW): 2026-11-09: {'insample_app': -0.4606, 'app': -0.1215, 'matched': 0.9733, 'insample_matched': 0.3251, 'studywin_insample_app': 0.415, 'studywin_app': 1.883, 'studywin_matched': 2.1854, 'studywin_insample_matched': 0.5685}; 2026-11-10: {'insample_app': 0.9272, 'app': 1.7912, 'matched': 2.0647, 'insample_matched': 0.7773, 'studywin_insample_app': 0.4592, 'studywin_app': 2.5937, 'studywin_matched': 2.2639, 'studywin_insample_matched': 0.4482}; 2026-11-11: {'insample_app': 0.7351, 'app': 1.9881, 'matched': 1.2058, 'insample_matched': 0.4546, 'studywin_insample_app': 0.3001, 'studywin_app': 2.7891, 'studywin_matched': 2.0244, 'studywin_insample_matched': 0.5831}; 2026-11-12: {'insample_app': 0.3632, 'app': 0.3875, 'matched': 0.7853, 'insample_matched': -0.0265, 'studywin_insample_app': -0.0244, 'studywin_app': 0.8564, 'studywin_matched': 1.3749, 'studywin_insample_matched': 0.1696}; 2026-11-13: {'insample_app': -0.1248, 'app': -0.4174, 'matched': -0.3145, 'insample_matched': -0.1996, 'studywin_insample_app': -0.308, 'studywin_app': -0.9574, 'studywin_matched': -1.046, 'studywin_insample_matched': -0.5709}; 2026-11-14: {'insample_app': 0.462, 'app': 1.9127, 'matched': 1.8663, 'insample_matched': 0.0228, 'studywin_insample_app': 0.0729, 'studywin_app': 0.924, 'studywin_matched': 1.5375, 'studywin_insample_matched': 0.345}; 2026-11-15: {'insample_app': -0.0301, 'app': -0.7477, 'matched': 0.1775, 'insample_matched': -0.0154, 'studywin_insample_app': -0.3118, 'studywin_app': -0.2991, 'studywin_matched': -0.1823, 'studywin_insample_matched': -0.6696}; 2026-11-16: {'insample_app': -0.656, 'app': -1.719, 'matched': -1.1741, 'insample_matched': -0.5727, 'studywin_insample_app': 0.2671, 'studywin_app': 0.6477, 'studywin_matched': 0.2278, 'studywin_insample_matched': -0.3258}; 2026-11-17: {'insample_app': -0.6718, 'app': 0.9252, 'matched': 1.4978, 'insample_matched': -0.0101, 'studywin_insample_app': -0.0586, 'studywin_app': 0.8681, 'studywin_matched': 0.5476, 'studywin_insample_matched': -0.2644}; 2026-11-18: {'insample_app': -0.5154, 'app': -0.6434, 'matched': -0.9349, 'insample_matched': -0.6087, 'studywin_insample_app': 0.148, 'studywin_app': 1.4156, 'studywin_matched': 1.4385, 'studywin_insample_matched': 0.3035}; 2026-11-19: {'insample_app': -0.1065, 'app': 0.5966, 'matched': 2.1356, 'insample_matched': 0.043, 'studywin_insample_app': -0.1986, 'studywin_app': 0.2192, 'studywin_matched': 1.0145, 'studywin_insample_matched': -0.3133}; 2026-11-20: {'insample_app': 0.2227, 'app': 1.63, 'matched': 1.7008, 'insample_matched': 0.0854, 'studywin_insample_app': 0.4685, 'studywin_app': 3.3137, 'studywin_matched': 3.1446, 'studywin_insample_matched': 0.5025}; 2026-11-21: {'insample_app': 0.232, 'app': 2.4509, 'matched': 2.4405, 'insample_matched': 0.6045, 'studywin_insample_app': -0.0022, 'studywin_app': 2.0374, 'studywin_matched': 1.5534, 'studywin_insample_matched': -0.0117}; 2026-11-22: {'insample_app': -0.7375, 'app': -2.2591, 'matched': -2.5159, 'insample_matched': -0.8805, 'studywin_insample_app': -0.3739, 'studywin_app': -1.4229, 'studywin_matched': -1.8016, 'studywin_insample_matched': -0.2642}; 2026-11-23: {'insample_app': -0.251, 'app': -0.9409, 'matched': -0.5164, 'insample_matched': -0.1562, 'studywin_insample_app': 0.1398, 'studywin_app': -0.3781, 'studywin_matched': -0.1795, 'studywin_insample_matched': -0.1054}; 2026-11-24: {'insample_app': -0.2351, 'app': -0.4106, 'matched': -0.3185, 'insample_matched': -0.2897, 'studywin_insample_app': 0.1509, 'studywin_app': 0.4368, 'studywin_matched': 0.1453, 'studywin_insample_matched': 0.3144}; 2026-11-25: {'insample_app': 0.4161, 'app': 1.3537, 'matched': 0.4483, 'insample_matched': 0.2875, 'studywin_insample_app': 0.433, 'studywin_app': 1.5847, 'studywin_matched': 0.9566, 'studywin_insample_matched': 0.4878}; 2026-11-26: {'insample_app': 0.1954, 'app': 0.8034, 'matched': 0.0526, 'insample_matched': -0.0894, 'studywin_insample_app': 0.1606, 'studywin_app': 0.5137, 'studywin_matched': 0.2165, 'studywin_insample_matched': -0.125}

## Load forms at RL level (wind/solar as in matched; load per form)

Diff = form − F0, per-day RL MAE, paired 7-day block bootstrap. Decision rule L5: a price pass only for the F1–F4 form with the largest all-slot gain ≥ 0.10 GW whose interval excludes 0.

| form | RL MAE all | diff all | RL MAE d1 | diff d1 | RL MAE d2+ | diff d2+ | load MAE | load bias |
|---|---|---|---|---|---|---|---|---|
| F0: fed in matched: cutoff-trained load | 2.15 | — | 1.59 | — | 2.34 | — | 1.39 | -0.76 |
| F1: PRIMARY held-out + own resid, gate window | 2.19 | +0.044 [-0.023, +0.117] | 1.66 | +0.063 [-0.086, +0.219] | 2.38 | +0.038 [-0.022, +0.104] | 1.48 | -0.90 |
| F2: held-out + own resid, study window | 2.03 | -0.117 [-0.214, -0.014] | 1.34 | -0.254 [-0.415, -0.077] | 2.27 | -0.068 [-0.175, +0.041] | 1.29 | -0.51 |
| F3: cutoff-trained + held-out resid, gate window | 2.15 | -0.002 [-0.084, +0.077] | 1.62 | +0.022 [-0.159, +0.192] | 2.33 | -0.010 [-0.085, +0.074] | 1.36 | -0.60 |
| F4: cutoff-trained + held-out resid, study window | 2.08 | -0.069 [-0.182, +0.045] | 1.40 | -0.198 [-0.409, +0.028] | 2.32 | -0.024 [-0.130, +0.079] | 1.35 | -0.20 |
| REF: published TSO load at the gate (load error 0) | 1.71 | -0.435 [-0.588, -0.283] | 1.23 | -0.361 [-0.645, -0.093] | 1.88 | -0.461 [-0.671, -0.255] | 0.00 | +0.00 |

L5 decision: request a price pass for F2 (-0.117 GW)

Wiring checks per day: 2026-11-09: {'variant': 'matched', 'stage2_minus_base_maxabs': 0.0, 'mixed_maxabs': 0.0, 'spy_calls': 2, 'spy_maxabs': 0.0, 'variant_minus_base_maxabs': 2.2687635089792977}; 2026-11-10: {'variant': 'matched', 'stage2_minus_base_maxabs': 5.684341886080802e-14, 'mixed_maxabs': 0.0, 'spy_calls': 2, 'spy_maxabs': 0.0, 'variant_minus_base_maxabs': 3.9020616894852083}; 2026-11-11: {'variant': 'matched', 'stage2_minus_base_maxabs': 0.0, 'mixed_maxabs': 0.0, 'spy_calls': 2, 'spy_maxabs': 0.0, 'variant_minus_base_maxabs': 3.857987027542883}; 2026-11-12: {'variant': 'matched', 'stage2_minus_base_maxabs': 7.105427357601002e-15, 'mixed_maxabs': 0.0, 'spy_calls': 2, 'spy_maxabs': 0.0, 'variant_minus_base_maxabs': 4.991757081630965}; 2026-11-13: {'variant': 'matched', 'stage2_minus_base_maxabs': 4.742872761198669e-13, 'mixed_maxabs': 0.0, 'spy_calls': 2, 'spy_maxabs': 0.0, 'variant_minus_base_maxabs': 6.683837344936581}; 2026-11-14: {'variant': 'matched', 'stage2_minus_base_maxabs': 0.0, 'mixed_maxabs': 0.0, 'spy_calls': 2, 'spy_maxabs': 0.0, 'variant_minus_base_maxabs': 2.547559280435493}; 2026-11-15: {'variant': 'matched', 'stage2_minus_base_maxabs': 0.0, 'mixed_maxabs': 0.0, 'spy_calls': 2, 'spy_maxabs': 0.0, 'variant_minus_base_maxabs': 5.405197815918076}; 2026-11-16: {'variant': 'matched', 'stage2_minus_base_maxabs': 0.0, 'mixed_maxabs': 0.0, 'spy_calls': 2, 'spy_maxabs': 0.0, 'variant_minus_base_maxabs': 7.970128006898705}; 2026-11-17: {'variant': 'matched', 'stage2_minus_base_maxabs': 0.0, 'mixed_maxabs': 0.0, 'spy_calls': 2, 'spy_maxabs': 0.0, 'variant_minus_base_maxabs': 5.994028232259751}; 2026-11-18: {'variant': 'matched', 'stage2_minus_base_maxabs': 0.0, 'mixed_maxabs': 0.0, 'spy_calls': 2, 'spy_maxabs': 0.0, 'variant_minus_base_maxabs': 4.972848815346106}; 2026-11-19: {'variant': 'matched', 'stage2_minus_base_maxabs': 3.552713678800501e-15, 'mixed_maxabs': 0.0, 'spy_calls': 2, 'spy_maxabs': 0.0, 'variant_minus_base_maxabs': 5.203523881993917}; 2026-11-20: {'variant': 'matched', 'stage2_minus_base_maxabs': 0.0, 'mixed_maxabs': 0.0, 'spy_calls': 2, 'spy_maxabs': 0.0, 'variant_minus_base_maxabs': 5.1498514115613325}; 2026-11-21: {'variant': 'matched', 'stage2_minus_base_maxabs': 0.0, 'mixed_maxabs': 0.0, 'spy_calls': 2, 'spy_maxabs': 0.0, 'variant_minus_base_maxabs': 6.493204237964715}; 2026-11-22: {'variant': 'matched', 'stage2_minus_base_maxabs': 0.0, 'mixed_maxabs': 0.0, 'spy_calls': 2, 'spy_maxabs': 0.0, 'variant_minus_base_maxabs': 4.524815861417665}; 2026-11-23: {'variant': 'matched', 'stage2_minus_base_maxabs': 0.0, 'mixed_maxabs': 0.0, 'spy_calls': 2, 'spy_maxabs': 0.0, 'variant_minus_base_maxabs': 6.974922490564708}; 2026-11-24: {'variant': 'matched', 'stage2_minus_base_maxabs': 0.0, 'mixed_maxabs': 0.0, 'spy_calls': 2, 'spy_maxabs': 0.0, 'variant_minus_base_maxabs': 4.072288036741682}; 2026-11-25: {'variant': 'matched', 'stage2_minus_base_maxabs': 0.0, 'mixed_maxabs': 0.0, 'spy_calls': 2, 'spy_maxabs': 0.0, 'variant_minus_base_maxabs': 5.6702438836047335}; 2026-11-26: {'variant': 'matched', 'stage2_minus_base_maxabs': 0.0, 'mixed_maxabs': 0.0, 'spy_calls': 2, 'spy_maxabs': 0.0, 'variant_minus_base_maxabs': 5.559117684042668}

Runtime per day (s): 2026-11-09 409.8, 2026-11-10 362.6, 2026-11-11 363.5, 2026-11-12 339.5, 2026-11-13 342.1, 2026-11-14 319.6, 2026-11-15 328.2, 2026-11-16 308.1, 2026-11-17 316.0, 2026-11-18 408.2, 2026-11-19 366.2, 2026-11-20 362.6, 2026-11-21 339.5, 2026-11-22 335.5, 2026-11-23 322.6, 2026-11-24 323.4, 2026-11-25 313.8, 2026-11-26 318.9

## Verdicts against EXPECTATIONS.md addendum 4 and its amendment (fresh days, one look)

18 days 2026-11-09..11-26, 0 failed days, scored once. About 3 blocks of 7 days: intervals are unstable.
The L5 line above is printed by the scorer; L5 was a rule for the used window and is not applied to fresh days.

- **N1 met.** Matched d2+ tree RL MAE below base on 11 of 18 days (bar: ≥ 11).
- **N2 met.** d2+ bias −2.16 → −0.83 GW.
- **N3 met.** Blend −0.031 [−0.088, +0.015]. N4 void.
- **N5 met.** All leak and wiring assertions passed.
- **N6 not met.** Matched d2+ bias −0.83, outside ±0.6. The wind bias is gone (+1.44 → +0.02); what remains is load
  (−0.91 on d2+, −0.83 on d1 slots too), which matched-age weather does not address.
- **N7 met.** Matched − control d2+ RL MAE −0.561 [−1.068, +0.057] (separate computation, same bootstrap).
- **N8 met.** Trees +0.012, blend −0.031, Timestral −0.043, market route −0.046.
- **N9 partly met.** REF (published load) −0.435 [−0.588, −0.283] below F0. F2 is −0.117 [−0.214, −0.014], beyond
  the expected 0.10; on the 61-day window it was +0.034 [−0.032, +0.112], so its sign is not stable across windows.
