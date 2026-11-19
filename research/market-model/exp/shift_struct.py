"""Structural forms for the three parts of the shift against Timestral's forecast of the same part. Notes 21.123."""
import numpy as np
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; Z = np.load(f"{SP}/out/shift15_lite.npz"); X = Z["rl"] - Z["other"]; U = X - Z["Sp"]; ND = 56; out = {}
def parts(v):  # (days, 96) -> level (days,), hourly profile (days, 24), within hour (days, 24, 4)
    h = v.reshape(len(v), 24, 4).mean(2); l = h.mean(1); return l, h - l[:, None], v.reshape(len(v), 24, 4) - h[:, :, None]
def ols(A, y):
    ok = np.isfinite(A).all(1) & np.isfinite(y); return np.linalg.lstsq(A[ok], y[ok], rcond=None)[0]
for per, tag in (("winter", "mid"), ("summer", "mid"), ("autumn", "v2"), ("winterf", "mid"), ("summerf", "mid")):
    I = np.load(f"{SP}/out/inputs_{per}.npz"); T = np.load(f"{SP}/out/tsshift_{tag}_lite_{per}.npz"); cuts = I["cuts"].astype(int); rows = []
    for j, c in enumerate(cuts):
        u = U[c:c + 96]; f = T["median"][j].astype(float)[:96]; xf = T["xh"][j].astype(float)[:96]
        if np.isnan(u).any(): continue
        hu = U[c - ND * 96:c].reshape(ND, 96); hx = X[c - ND * 96:c].reshape(ND, 96); ul, up, uq = parts(hu); xl, xp, xq = parts(hx)
        b = ols(xq.reshape(-1, 1), uq.ravel())[0]                                                        # within hour
        A = np.zeros((ND * 24, 25)); A[np.arange(ND * 24), np.tile(np.arange(24), ND)] = 1; A[:, 24] = xp.ravel(); co = ols(A, up.ravel()); kh, l = co[:24], co[24]   # profile
        wk = ((np.arange(ND) + 0) % 7); dow = np.array([(c // 96 - ND + i) % 7 for i in range(ND)])     # day index modulo 7 as a stand-in for the weekday pattern
        D = np.column_stack([np.ones(ND - 1), ul[:-1], xl[1:] - xl[:-1]]); g = ols(D, ul[1:])           # level
        fl, fp, fq = parts(f[None]); tl, tp, tq = parts(u[None]); xfl, xfp, xfq = parts(xf[None])
        sl = g[0] + g[1] * ul[-1] + g[2] * (xfl[0] - xl[-1]); sp = kh + l * xfp[0]; sq = b * xfq[0]
        rows.append(dict(tl=tl[0], tp=tp[0], tq=tq[0], fl=fl[0], fp=fp[0], fq=fq[0], sl=sl, sp=sp, sq=sq, b=b, l=l, r=g[1], g=g[2], u=u, f=f))
    R = {k: np.array([r[k] for r in rows]) for k in rows[0]}; out[per] = R; n = len(rows)
    mse = lambda a, t: float(((a - t) ** 2).mean()); line = [f"{per:8s} {n:3d} days | coefficients (median): within hour b {np.median(R['b']):.2f}, profile l {np.median(R['l']):.2f}, level r {np.median(R['r']):.2f} g {np.median(R['g']):.2f}"]
    for nm, t, f_, s in (("level", "tl", "fl", "sl"), ("profile", "tp", "fp", "sp"), ("within hour", "tq", "fq", "sq")):
        a, b_, c_ = mse(R[f_], R[t]), mse(R[s], R[t]), mse(0.5 * (R[f_] + R[s]), R[t]); line.append(f"{nm}: RMSE Timestral {np.sqrt(a):.2f}  structural {np.sqrt(b_):.2f} ({100 * (np.sqrt(b_ / a) - 1):+.0f}%)  average {np.sqrt(c_):.2f} ({100 * (np.sqrt(c_ / a) - 1):+.0f}%)")
    def whole(l_, p_, q_): return (l_[:, None, None] + p_[:, :, None] + q_).reshape(n, 96)
    base = mse(R["f"], R["u"]); hyb = {"structural within hour": whole(R["fl"], R["fp"], R["sq"]), "average within hour": whole(R["fl"], R["fp"], 0.5 * (R["fq"] + R["sq"])), "average profile": whole(R["fl"], 0.5 * (R["fp"] + R["sp"]), R["fq"]), "average all three": whole(0.5 * (R["fl"] + R["sl"]), 0.5 * (R["fp"] + R["sp"]), 0.5 * (R["fq"] + R["sq"])), "structural alone": whole(R["sl"], R["sp"], R["sq"])}
    line.append("whole shift RMSE: Timestral " + f"{np.sqrt(base):.2f} | " + "  ".join(f"{k} {100 * (np.sqrt(mse(v, R['u']) / base) - 1):+.1f}%" for k, v in hyb.items()))
    print("\n   ".join(line)); np.savez_compressed(f"{SP}/out/shift_struct_{per}.npz", **{k: v for k, v in hyb.items()}, ok=np.array([~np.isnan(U[c:c + 96]).any() for c in cuts]))
