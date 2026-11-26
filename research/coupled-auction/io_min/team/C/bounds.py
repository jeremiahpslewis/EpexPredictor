"""Hindsight bounds for combining the three champion members (labelled BOUND: fitted on the scored days)."""
import numpy as np
from panel import champion, day_mae, MEM
P = champion(); X = P["X"]; a = P["actual"]; D = len(a)
S61 = np.arange(61); S18 = np.arange(62, 80)
g = np.arange(0, 1.0001, 0.05)
W = np.array([(x, y, 1 - x - y) for x in g for y in g if x + y <= 1.0001]); W[:, 2] = np.clip(W[:, 2], 0, 1)
def comb(w, X=X): return np.einsum("dsk,k->ds", X, w)
def score(F, idx): return np.nanmean(day_mae(F, a)[idx])
def best_w(mask_days, slot_mask=None, bias=False):
    """L1-best simplex weight on given days (and slots); returns w, (bias)"""
    Xs = X[mask_days]; As = a[mask_days]
    if slot_mask is not None: Xs = np.where(slot_mask[mask_days][..., None], Xs, np.nan)
    E = np.einsum("dsk,wk->wds", Xs, W) - As[None]
    if bias:
        b = np.nanmedian(E.reshape(len(W), -1), 1); L = np.nanmean(np.abs(E - b[:, None, None]).reshape(len(W), -1), 1)
        i = np.nanargmin(L); return W[i], b[i]
    L = np.nanmean(np.abs(E).reshape(len(W), -1), 1); i = np.nanargmin(L); return W[i], 0.0
eq = X.mean(-1); res = []
def rep(name, F, note=""):
    res.append((name, score(F, S61), score(F, S18), note)); print(f"{name:48s} 61d {score(F,S61):.3f}  18d {score(F,S18):.3f}  {note}")
rep("equal thirds (champion)", eq)
for k, m in zip(MEM, range(3)): rep(f"member {k}", X[..., m])
rep("median of 3", np.median(X, -1))
# --- diagnostics
E = X - a[..., None]; same = (np.sign(E).min(-1) == np.sign(E).max(-1))
ae = np.abs(eq - a); print("share of slots all-same-side:", np.nanmean(same[S61]).round(3), " share of equal-blend abs error from those slots:", (np.nansum(ae[S61] * same[S61]) / np.nansum(ae[S61])).round(3))
ok = np.isfinite(a[S61]); print("member error corr (61d):\n", np.corrcoef(np.stack([E[S61][..., k][ok] for k in range(3)])).round(2))
orc = np.where(same, np.abs(E).min(-1), 0.0); orc[np.isnan(a)] = np.nan
rep("ORACLE per-slot convex (floor of any weighting)", a + orc, "BOUND")
rep("ORACLE per-slot best member", a + np.take_along_axis(np.abs(E), np.nanargmin(np.where(np.isnan(E), 1e9, np.abs(E)), -1)[..., None], -1)[..., 0], "BOUND")
# per-day oracle simplex
Fd = np.empty_like(eq)
for d in range(D):
    L = np.nanmean(np.abs(np.einsum("sk,wk->ws", X[d], W) - a[d]), 1); Fd[d] = X[d] @ W[np.nanargmin(L)]
rep("ORACLE per-day simplex weights", Fd, "BOUND")
# fixed weights fit on 61 (bound on 61, out-of-sample on 18)
w, _ = best_w(S61); rep(f"fixed simplex fit61 w={w.round(2)}", comb(w), "61=BOUND,18=OOS")
w, b = best_w(S61, bias=True); rep(f"fixed simplex+bias fit61 w={w.round(2)} b={b:.2f}", comb(w) - b, "61=BOUND,18=OOS")
w18, _ = best_w(S18); print("   (best fixed w on 18 alone:", w18.round(2), ")")
# blocks
qh = P["qh"]; 
def blockwise(blocks_of, bias=False, fit=S61, name=""):
    F = np.full_like(eq, np.nan); ws = {}
    for bl in np.unique(blocks_of[np.isfinite(blocks_of)]):
        m = blocks_of == bl; w, b = best_w(fit, m, bias); F[m] = (comb(w) - b)[m]; ws[bl] = (w, b)
    return F, ws
for nb in (4, 6, 12, 24, 96):
    blk = np.floor(qh / (96 / nb)); F, ws = blockwise(blk); rep(f"ToD {nb} blocks simplex fit61", F, "61=BOUND,18=OOS")
    if nb == 6: print("   6-block weights:", {int(k): v[0].round(2).tolist() for k, v in ws.items()})
for nb in (6, 24):
    blk = np.floor(qh / (96 / nb)); F, ws = blockwise(blk, bias=True); rep(f"ToD {nb} blocks simplex+bias fit61", F, "61=BOUND,18=OOS")
# bias correction of equal thirds by ToD (median error), fit on 61
for nb in (6, 24):
    blk = np.floor(qh / (96 / nb)); F = eq.copy()
    for bl in range(nb):
        m = blk == bl; F[m] = eq[m] - np.nanmedian((eq - a)[S61][m[S61]])
    rep(f"equal thirds - median bias by {nb} ToD blocks fit61", F, "61=BOUND,18=OOS")
# regimes from gate-known quantities
wd = P["wd"]; rep("weekday/sat/sun weights fit61", blockwise(np.where(wd < 5, 0, wd - 4))[0], "61=BOUND,18=OOS")
lvl = np.repeat(np.nanmean(eq, 1, keepdims=True), 100, 1); q = np.nanpercentile(lvl[S61, 0], [33, 67])
rep("forecast price level tertile weights fit61", blockwise(np.digitize(lvl, q).astype(float))[0], "61=BOUND,18=OOS")
rl = P["rl_mixed"]; q = np.nanpercentile(rl[S61], [33, 67]); rlb = np.where(np.isnan(rl), np.nan, np.digitize(rl, q).astype(float))
rep("RL tertile (per slot) weights fit61", blockwise(rlb)[0], "61=BOUND,18=OOS")
sprd = X.max(-1) - X.min(-1); q = np.nanpercentile(sprd[S61], [33, 67])
rep("member spread tertile weights fit61", blockwise(np.where(np.isnan(sprd), np.nan, np.digitize(sprd, q).astype(float)))[0], "61=BOUND,18=OOS")
# leave-one-week-out for ToD-6 and ToD-24 (bound but less overfit)
for nb in (6, 24, 96):
    blk = np.floor(qh / (96 / nb)); F = np.full_like(eq, np.nan)
    for wk in range(9):
        te = np.arange(wk * 7, min(61, wk * 7 + 7)); tr = np.setdiff1d(S61, te)
        Fw, _ = blockwise(blk, fit=tr); F[te] = Fw[te]
    rep(f"ToD {nb} blocks leave-one-week-out (61d)", F, "LOWO")
np.save("/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad/C_bounds.npy", np.array(res, dtype=object), allow_pickle=True)
