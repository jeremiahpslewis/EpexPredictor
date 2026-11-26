"""Walk-forward combination learners: at origin d only days < d are used (D-1 prices are public before the D gate).
Burn-in days fall back to equal thirds and stay in the mean."""
import sys, numpy as np
sys.path.insert(0, "/home/user/EpexPredictor/research/coupled-auction/io_min/distill")
from panel import champion, day_mae, MEM
g = np.arange(0, 1.0001, 0.05)
W = np.array([(x, y, max(0., 1 - x - y)) for x in g for y in g if x + y <= 1.0001])

def l1w(X, a):
    E = np.einsum("nk,wk->wn", X, W) - a[None]; return W[np.argmin(np.mean(np.abs(E), 1))]

def run(P, method, win=None, k=20, nb=1, lam=1.0, burn=7, bias=False, decay=None):
    X, a, qh = P["X"], P["actual"], P["qh"]; D = len(a); eq = X.mean(-1); F = eq.copy()
    blk = np.floor(qh / (96 / nb))
    for d in range(burn, D):
        lo = 0 if win is None else max(0, d - win); tr = np.arange(lo, d); n = len(tr)
        for bl in range(nb):
            m_tr = (blk[tr] == bl) & np.isfinite(a[tr]); m_d = blk[d] == bl
            Xt, at = X[tr][m_tr], a[tr][m_tr]
            if method == "weights":
                w = l1w(Xt, at); s = n / (n + k); w = s * w + (1 - s) * np.ones(3) / 3
                f = X[d][m_d] @ w
                if bias: f = f - lam * s * np.median(Xt @ w - at)
            elif method == "bias":      # equal thirds minus shrunk median error
                s = n / (n + k); f = eq[d][m_d] - lam * s * np.median(Xt.mean(-1) - at)
            F[d][m_d] = f
    return F

if __name__ == "__main__":
    P = champion(); a = P["actual"]; eq = P["X"].mean(-1); S61, S18 = np.arange(61), np.arange(62, 80)
    from common import boot
    def sc(F): dm = day_mae(F, a); return dm[S61].mean(), dm[S18].mean()
    print("equal thirds", np.round(sc(eq), 3))
    out = []
    for method in ("weights", "bias"):
        for win in (None, 28, 14):
            for k in (0, 7, 28):
                for nb in (1, 6, 24):
                    for bias in ((False, True) if method == "weights" else (True,)):
                        F = run(P, method, win=win, k=k, nb=nb, bias=bias); r = sc(F)
                        out.append((r[0], r[1], method, win, k, nb, bias)); print(f"{method:8s} win={str(win):4s} k={k:2d} nb={nb:2d} bias={bias!s:5s}  61d {r[0]:.3f}  18d {r[1]:.3f}", flush=True)
