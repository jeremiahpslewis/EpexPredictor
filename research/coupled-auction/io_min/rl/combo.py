"""POST-HOC combinations from stored runs (see EXPECTATIONS.md E2c). usage: combo.py PROTO [TEST]"""
import sys, os, numpy as np
import common as c
proto = sys.argv[1]; test = sys.argv[2] if len(sys.argv) > 2 else "gate"; comps = ("wind", "solar", "load")
def L(n): f = f"{c.OUT}/{n}_{proto}_autumn.npz"; return np.load(f) if os.path.exists(f) else None
B = L("base"); T = B["load_true"] - B["wind_true"] - B["solar_true"]
near = (np.arange(c.HZ) < c.GATE_D1)[None]
def get(R, t, corr=None, k=test):
    f = R[f"{t}_{k}"].astype(float)
    if corr == "decay": f = f + (R[f"{t}_prevtrue"] - R[f"{t}_prev_d1"])[:, -24:].mean(1, keepdims=True) * np.exp(-(np.arange(c.HZ) + 1) / 96.0)[None]
    return f
def combo(spec, corr_comps=()):
    F = {}
    for t in comps:
        src = spec[t]; corr = "decay" if t in corr_comps else None
        if isinstance(src, tuple): F[t] = np.where(near, get(L(src[0]), t, corr), get(L(src[1]), t, corr))
        else: F[t] = get(L(src), t, corr)
    return F["load"] - F["wind"] - F["solar"], F
base, _ = combo({t: "base" for t in comps})
specs = {
    "base+decay(load)": ({t: "base" for t in comps}, ("load",)),
    "VM": ({t: ("base", "trd2") for t in comps}, ()),
    "VM+decay(load)": ({t: ("base", "trd2") for t in comps}, ("load",)),
    "VM+decay(all)": ({t: ("base", "trd2") for t in comps}, comps),
    "COMP": ({"load": "base", "wind": "nots", "solar": "nots"}, ()),
    "COMP+decay(load)": ({"load": "base", "wind": "nots", "solar": "nots"}, ("load",)),
    "VM1": ({t: ("base", "trd1") for t in comps}, ()),
    "VM1+decay(all)": ({t: ("base", "trd1") for t in comps}, comps),
    "VMstack+decay(load)": ({t: ("base", "stack") for t in comps}, ("load",)),
}
print(f"== POST-HOC combos, {proto}, test {test}: MAE (diff vs base [5,95])")
for n, (sp, cc) in specs.items():
    try: f, F = combo(sp, cc)
    except (TypeError, KeyError): continue
    m = c.mae_seg(f, T)
    print(f"{n:<22}" + "".join(f"{m[s]:6.2f} ({c.boot(f, base, T, s)[0]:+.2f} [{c.boot(f, base, T, s)[1]:+.2f},{c.boot(f, base, T, s)[2]:+.2f}])   " for s in c.SEG)
          + " day1 comp " + " ".join(f"{t[0]}{np.nanmean(np.abs(F[t] - B[f'{t}_true'])[:, :96]):.2f}" for t in comps))
