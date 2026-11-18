"""Each border's exchange curve fitted to the scheduled exchange alone (same form and priors as in the model, no other part of the likelihood),
next to what the joint fit of version 12 gives. Hours with a price difference above 0.5 EUR/MWh."""
import numpy as np, pandas as pd
from scipy.optimize import minimize
import star as S
SP = S.lc.SP; tag = "2026-09-18"; z = np.load(f"{SP}/out/star_v12/{tag}_s0.npz"); q = S.unpack({k[4:]: z[k] for k in z.files if k.startswith("raw_")})
I = np.load(f"{SP}/out/inputs_autumn.npz"); d0 = int(I["cuts"][[str(c)[:10] for c in I["cutoffs"]].index(tag)]); d1 = d0 - 96; lo = max(d1 - S.TRAIN_DAYS * 96, 15 * 96); lo += (-lo) % 4
H = S.window(lo, d1); week = (H["day"] - H["day"].min()) // 7; d = S.data_of(H, week, int(week.max()) + 1); P = H["P"]; sg = lambda u: 1 / (1 + np.exp(-np.clip(u, -60, 60))); res = {}
print("border: joint fit (export cap / import cap / offset / scale)  ->  exchange data alone  | scheduled exchange 2%..98% | prior centre")
for k, zn in enumerate(S.ZN[1:]):
    dp = P[:, k + 1] - P[:, 0]; x = d["x"][:, k]; m_ = d["xok"][:, k] & (np.abs(dp) > 0.5) & np.isfinite(dp); dpv, xv, wx = dp[m_], x[m_], d["wx"][m_]
    def nll(t):
        ce, ci, m0, kap, s_, gw, gs = np.exp(t[0]), np.exp(t[1]), t[2], np.exp(t[3]), np.exp(t[4]), t[5], t[6]; r = (xv - (-ci + (ce + ci) * sg((dpv - m0) / kap) + gw * wx[:, 0] + gs * wx[:, 1])) / s_
        return (np.sum(2.5 * np.log1p(r * r / 4.0) + np.log(s_)) * S.LW + 0.5 * ((t[0] - np.log(S.NTC[k])) / 0.5) ** 2 + 0.5 * ((t[1] - np.log(S.NTC[k])) / 0.5) ** 2 + 0.5 * (m0 / 10) ** 2 + 0.5 * ((t[3] - np.log(10.0)) / 0.7) ** 2 + 0.5 * (gw / 0.3) ** 2 + 0.5 * (gs / 0.3) ** 2)
    best = None
    for k0 in (np.log(3.0), np.log(10.0), np.log(30.0)):
        r_ = minimize(nll, np.array([np.log(max(np.percentile(xv, 97), 0.2)), np.log(max(-np.percentile(xv, 3), 0.2)), 0.0, k0, np.log(0.7), 0.0, 0.0]), method="L-BFGS-B")
        if best is None or r_.fun < best.fun: best = r_
    t = best.x; res[zn] = t
    print(f"  {zn}: {q['Ce'][k]:4.1f} / {q['Ci'][k]:4.1f} / {q['m'][k]:+5.1f} / {q['kappa'][k]:4.1f}   ->   {np.exp(t[0]):4.1f} / {np.exp(t[1]):4.1f} / {t[2]:+5.1f} / {np.exp(t[3]):4.1f}; wind {t[5]:+.2f}, sun {t[6]:+.2f} GW per sd; noise scale {np.exp(t[4]):.2f} GW | {np.percentile(xv, 2):+.1f}..{np.percentile(xv, 98):+.1f} | {S.NTC[k]:.1f}   ({m_.sum()} hours)")
np.savez(f"{SP}/out/exch_alone_{tag}.npz", **res)
