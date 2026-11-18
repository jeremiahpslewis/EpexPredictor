"""Weekly fits of the index model (second version) on a fixed calendar, cached. Fit dated T uses the 540 days before T (local midnight).
usage: ocx_fits.py SHARD NSHARDS [START END]"""
import sys, os, time, numpy as np, pandas as pd
import oc, ocx2
from ocx2 import fitx2, hx
V = os.environ.get("OCXV", ""); SFX = "" if not V else "_" + V
if V == "t":
    import ocxt; fitx2 = ocxt.fit_t
if V in ("c", "cn"):
    _f2 = fitx2; fitx2 = lambda h1: _f2(h1, steps=8000)                 # current priors, converged
if V in ("p1", "p2", "p3"):
    os.environ["PT"] = "1" if V == "p2" else "0"; os.environ["EL"] = "1" if V == "p3" else "0"
    import ocxp; fitx2 = ocxp.fit_p
FD = f"{oc.SP}/out/ocxw{SFX}"; os.makedirs(FD, exist_ok=True)
def fit_dates(start="2025-06-02", end="2026-11-09"): return [str(d.date()) for d in pd.date_range(start, end, freq="7D")]
def get_fit(date):
    f = f"{FD}/{date}.npz"
    if not os.path.exists(f):
        A, _ = fitx2(hx(date)); np.savez(f + ".tmp.npz", **A); os.replace(f + ".tmp.npz", f)
    return {k: np.asarray(v, float) for k, v in np.load(f).items()}
if __name__ == "__main__":
    sh, ns = int(sys.argv[1]), int(sys.argv[2]); ds = fit_dates(*sys.argv[3:5])[sh::ns]; t0 = time.time()
    for d in ds: get_fit(d); print(d, f"{time.time() - t0:.0f}s", flush=True)
