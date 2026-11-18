"""Stability diagnostics (notes 18.2 d): gas share / gas response at 30 GW across origins, and the difference between forecasts of the same hour from consecutive daily fits."""
import numpy as np, os
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
for lab, runs in (("as the candidate", ("hw2", "hs2", "ha")), ("with the prior", ("kw", "ks", "ka"))):
    S_, H_ = [], []
    for period, run in zip(("winter", "summer", "autumn"), runs):
        I = np.load(f"{SP}/out/inputs_{period}.npz")
        for c in I["cutoffs"]:
            z = np.load(f"{SP}/out/struct_{run}/{str(c)[:10]}.npz"); g = float(z["gref"]); b = float(np.interp(30.0, z["grid"], z["b_curve"])); h = float(np.interp(30.0, z["grid"], z["h_curve"])); S_.append(g * h / (b + g * h)); H_.append(h / 1.85)
    S_, H_ = np.array(S_), np.array(H_); print(f"{lab:17s}: gas response at 30 GW (h / 1.85) range {H_.min():.2f}..{H_.max():.2f}, sd of the change between consecutive origins {np.diff(H_).std():.3f}; gas share range {S_.min():.2f}..{S_.max():.2f}")
I = np.load(f"{SP}/out/inputs_autumn.npz"); a = I["actual"].astype(float); cut = I["cuts"]; pairs = [j for j in range(len(a) - 1) if cut[j + 1] - cut[j] == 96]
for lab, f in (("as the candidate", "structfc_ha_known_dyn2xxh"), ("with the prior", "structfc_ka_known_dyn2xxh"), ("as the candidate, 12000 steps", "structfc_ca_known_dyn2xxh")):
    v = np.load(f"{SP}/out/{f}.npz")["curve"].astype(float); d = np.array([v[j + 1, :56] - v[j, 96:152] for j in pairs]); print(f"autumn, same hour from consecutive daily fits, published loads, curve alone, {lab}: mean absolute difference {np.nanmean(np.abs(d)):.2f} ct")
