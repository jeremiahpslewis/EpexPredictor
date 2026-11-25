"""Score candidates against the base of the same protocol on the same days. usage: score.py PROTO [PERIOD] [TEST] [cand ...]
Corrections (c): +mean = add the mean D-1 residual (published - forecast, per component) to every lead;
+decay = add the mean residual of the last 6 h of D-1 x exp(-(lead+1)/96 qh). The D-1 forecast uses weather RES (d1 in autumn:
the 1-day-old run exists for all of D-1 at the gate; d2h in spring) and a model that never saw D-1."""
import sys, os, numpy as np
import common as c
proto = sys.argv[1]; period = sys.argv[2] if len(sys.argv) > 2 else "autumn"; test = sys.argv[3] if len(sys.argv) > 3 else ("gate" if period == "autumn" else "d2h")
cands = sys.argv[4:] or ["base", "trd2", "stack", "trd1", "w80", "split", "recency", "trd2_rec", "trd1_rec"]
RES = "d1" if period == "autumn" else "d2h"; comps = ("wind", "solar", "load")
def load(n):
    f = f"{c.OUT}/{n}_{proto}_{period}.npz"; return np.load(f) if os.path.exists(f) else None
def rl_of(R, corr=None):
    F = {}
    for t in comps:
        f = R[f"{t}_{test}"].astype(float)
        if corr:
            r = R[f"{t}_prevtrue"] - R[f"{t}_prev_{RES}"]
            if corr == "mean": f = f + r.mean(1, keepdims=True)
            elif corr == "decay": f = f + r[:, -24:].mean(1, keepdims=True) * np.exp(-(np.arange(c.HZ) + 1) / 96.0)[None]
        F[t] = f
    return F["load"] - F["wind"] - F["solar"], F
B = load("base"); T = B["load_true"] - B["wind_true"] - B["solar_true"]; base_rl, _ = rl_of(B)
print(f"== RL MAE GW, protocol {proto}, period {period}, test weather {test}, {len(T)} days; diff vs base: paired block bootstrap by day (7), 5..95%")
print(f"{'candidate':<16}" + "".join(f"{s:>30}" for s in c.SEG))
rows = []
for n in cands:
    R = load(n)
    if R is None: continue
    for corr in (None, "mean", "decay"):
        f, F = rl_of(R, corr); lab = n + (f"+{corr}" if corr else "")
        m = c.mae_seg(f, T); cells = []
        for s in c.SEG:
            d, lo, hi = c.boot(f, base_rl, T, s)
            cells.append(f"{m[s]:.2f} ({d:+.2f} [{lo:+.2f},{hi:+.2f}])" if lab != "base" else f"{m[s]:.2f}")
        comp = " ".join(f"{t[0]}{np.nanmean(np.abs(F[t] - R[f'{t}_true'])[:, :96]):.2f}" for t in comps)
        print(f"{lab:<16}" + "".join(f"{x:>30}" for x in cells) + f"   day1 comp {comp}")
