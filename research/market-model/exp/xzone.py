import numpy as np
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
Z = ["NL", "BE", "AT", "FR", "DK1", "DK2", "SE4"]
for n, lab in (("ho_dw", "last ten weeks"), ("yr_dw", "previous year")):
    z = np.load(f"{SP}/out/{n}.npz"); A = {k: z[f"actual_{k}"][:, :96].astype(float) for k in Z + ["DE"]}; E = {k: A[k] - z[f"s2_{k}"][:, :96].astype(float) for k in A}
    e = E["DE"]; pk = A["DE"] > 30; under = e > 5; over = e < -5
    print(f"\n=== {lab}: first forecast day, {e.size} quarter hours; DE above 30 ct: {pk.sum()}, DE under-forecast by >5 ct: {under.sum()}, over-forecast by >5 ct: {over.sum()}")
    print(f"{'zone':<5} corr of errors  same in DE peak hours  share of DE under-forecasts where zone is also >5 under  zone's mean error then  DE-zone price gap in peaks  corr with zone's error a day earlier")
    for k in Z:
        ok = np.isfinite(E[k]) & np.isfinite(e)
        lagc = np.corrcoef(E[k][:-1][ok[:-1] & ok[1:]], e[1:][ok[:-1] & ok[1:]])[0, 1] if n == "ho_dw" else np.nan
        print(f"{k:<5} {np.corrcoef(E[k][ok], e[ok])[0,1]:+.2f}            {np.corrcoef(E[k][ok & pk], e[ok & pk])[0,1]:+.2f}                   {100*np.mean(E[k][ok & under] > 5):3.0f}%                                               {E[k][ok & under].mean():+5.1f}                  {np.mean((A['DE'] - A[k])[ok & pk]):+5.1f}                       {lagc:+.2f}")
    ok = np.isfinite(e); print(f"DE    own error a day earlier: {np.corrcoef(e[:-1].ravel(), e[1:].ravel())[0,1]:+.2f}" if n == "ho_dw" else "")
    # how much of the DE error is common to the neighbours
    X = np.stack([E[k].ravel() for k in Z], 1); y = e.ravel(); ok = np.isfinite(X).all(1) & np.isfinite(y); X = np.c_[X[ok], np.ones(ok.sum())]; y = y[ok]
    b = np.linalg.lstsq(X, y, rcond=None)[0]; print(f"share of the DE error variance shared with the neighbours' errors at the same time: {1 - np.var(y - X @ b) / np.var(y):.0%}; in DE peak hours: ", end="")
    p = pk.ravel()[ok]; print(f"{1 - np.var((y - X @ b)[p]) / np.var(y[p]):.0%}")
