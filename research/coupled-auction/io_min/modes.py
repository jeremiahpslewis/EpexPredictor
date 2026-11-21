"""Locate the modes of the two-zone fit and split the log density by block. Usage: modes.py PANEL_DIR POST.npz NCHAINS"""
import sys, numpy as np, pandas as pd, jax, jax.numpy as jnp
from numpyro.infer.util import log_density
from numpyro import handlers
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import model as Mo
Pn = pd.read_parquet(f"{sys.argv[1]}/panel.parquet"); x, obs, brk = Mo.to_inputs(Pn)
S = dict(np.load(sys.argv[2])); nc = int(sys.argv[3]); n = len(S["h_G"]) // nc
keys = ["alpha_L", "alpha_C", "alpha_G", "F_L", "F_C", "phi0", "phi1", "sigma_L", "sigma_C", "sigma_G", "h_G", "a_C", "a_L"]
for c in range(nc):
    print(f"chain {c}:", {k: round(float(np.median(S[k][c * n:(c + 1) * n])), 3) for k in keys})
def blocks(i):
    par = {k: jnp.asarray(v[i]) for k, v in S.items() if k in ("Lcorr",) or (v.ndim >= 1 and len(v) == len(S["h_G"]) and k not in ("divergences",))}
    tr = handlers.trace(handlers.substitute(Mo.model, par)).get_trace(x, obs, brk)
    out = {}
    for name, site in tr.items():
        if site["type"] == "sample" and not site.get("is_observed", False): out.setdefault("prior", 0.0); out["prior"] += float(site["fn"].log_prob(site["value"]).sum())
        elif site["type"] == "sample": out[name] = float(site["fn"].log_prob(site["value"]).sum())
    return out
for c in range(nc):
    idx = range(c * n, (c + 1) * n, max(1, n // 40)); B = [blocks(i) for i in idx]
    print(f"chain {c} mean log density by block:", {k: round(float(np.mean([b[k] for b in B])), 1) for k in B[0]})
# implied supply at typical conditions: thermal output by fuel at p = 60, 100, 150 (mean panel x)
xm = {k: jnp.asarray(float(np.mean(np.asarray(v)))) for k, v in x.items()}
for c in range(nc):
    th = {k: jnp.asarray(np.median(S[k][c * n:(c + 1) * n])) for k in Mo.THETA_KEYS}
    rows = []
    for p in (-10, 30, 60, 100, 150, 250):
        Sf, dS = Mo.fuel_supply(th, xm, jnp.asarray(float(p)))
        rows.append((p, *[round(float(Sf[f]), 2) for f in Mo.FUELS], round(float(sum(Sf.values())), 2), round(float(sum(dS.values())), 3)))
    print(f"chain {c} (p, L, C, G, total, slope GW per EUR):", rows)
