"""
The pre-registered table (prereg/PREREG_OUTAGE.md, "Output: the first table"). Usage: summarise11.py RESULTS.pkl OUT.md
Balanced panel: for each posterior draw, only MTUs where all five scenarios converged, so differences across headroom
are not differences in which MTUs dropped out. The unbalanced panel and the failure shares are reported alongside.
Replacement shares: each zone's supply response dS_z (GW) per 1 GW removed; DE split into lignite, coal and other gas
(other gas = dG + 1, since the outage itself removes 1 GW).
"""
import sys, pickle
import numpy as np, pandas as pd
sys.path.insert(0, __file__.rsplit("/", 1)[0]); import net11 as N
rows = pickle.load(open(sys.argv[1], "rb")); SC = ["replay", "eta0.10", "eta0.25", "eta0.50", "eta1.00"]
recs = []
for r in rows:
    for s in SC:
        o = r["res"][s]
        recs.append(dict(basin=r["post"].split("post_")[-1].split(".")[0], draw=r["draw"], t=r["t"], scen=s, ok=o["ok"] and not o["cap"],
                         dpDE=o["dp"][0], dimp=o["dimport"], dL=o["dL"], dC=o["dC"], dGother=o["dG"] + 1.0,
                         **{f"dS_{z}": o["dS"][i] for i, z in enumerate(N.Z) if z != "DE"}, base_err=o["base_err_p"],
                         generic=r["flags"]["generic"], pub_excl=r["flags"]["pub_excl"], ahc=r["flags"]["ahc"], miss=r["flags"]["link_missing"],
                         held=o.get("held_bounds", 0)))
D = pd.DataFrame(recs)
allok = D.groupby(["basin", "draw", "t"]).ok.transform("all"); D["bal"] = allok
cols = ["dpDE", "dimp", "dL", "dC", "dGother"] + [f"dS_{z}" for z in N.Z if z != "DE"]
def table(sub):
    per_draw = sub.groupby(["basin", "scen", "draw"])[cols].mean()
    g = per_draw.groupby(["basin", "scen"])
    m, lo, hi = g.mean(), g.quantile(0.05), g.quantile(0.95)
    med = sub.groupby(["basin", "scen"])[cols].median()
    return m, lo, hi, med
out = ["# Eleven-zone outage study: first table", "",
       f"Source: `{sys.argv[1].split('/')[-1]}`. {D.t.nunique()} outage MTUs, {D.groupby('basin').draw.nunique().to_dict()} posterior draws per basin.", ""]
fail = D.groupby(["basin", "scen"]).ok.apply(lambda x: 1 - x.mean()).unstack().round(3)
out += ["## Share of draw-MTUs not converged (or at the price cap), by scenario", "", fail.to_markdown(), ""]
out += [f"Balanced panel (all five scenarios converged): {D[D.scen == 'replay'].bal.mean():.1%} of draw-MTUs.", ""]
for name, sub in (("balanced panel", D[D.bal]), ("unbalanced (each scenario's converged MTUs)", D[D.ok])):
    m, lo, hi, med = table(sub)
    out += [f"## Mean over MTUs, {name}: posterior mean and [5%, 95%] across draws", ""]
    T = pd.DataFrame({c: [f"{m.loc[i, c]:+.3f} [{lo.loc[i, c]:+.3f}, {hi.loc[i, c]:+.3f}]" for i in m.index] for c in cols}, index=[f"{a} / {b}" for a, b in m.index])
    out += [T.to_markdown(), ""]
    out += [f"Median across MTUs ({name}):", "", med.round(3).to_markdown(), ""]
    if name.startswith("balanced"):
        dom = m["dL"] + m["dC"] + m["dGother"]; out += ["German own replacement (lignite + coal + other gas, GW per GW removed):", "", dom.round(3).to_frame("DE share").to_markdown(), ""]
        rule = []
        for b in m.index.get_level_values(0).unique():
            p10, p100 = med.loc[(b, "eta0.10"), "dpDE"], med.loc[(b, "eta1.00"), "dpDE"]
            s10, s100 = dom.loc[(b, "eta0.10")], dom.loc[(b, "eta1.00")]
            rule.append(f"- {b}: median ΔpDE η=0.10 {p10:+.2f} vs η=1.00 {p100:+.2f} ({(p10 - p100) / abs(p100) if p100 else float('nan'):+.0%}); "
                        f"DE own share {s10:.2f} vs {s100:.2f} (Δ {s10 - s100:+.2f}). Material by the pre-registered rule: "
                        f"{abs(p10 - p100) > 0.30 * abs(p100) or abs(s10 - s100) > 0.15}")
        out += ["## Reading rule (pre-registered: >30% change in median ΔpDE or >0.15 in DE own share between η=0.10 and 1.00)", ""] + rule + [""]
for flag in ("generic", "pub_excl", "ahc", "miss"):
    sub = D[D.bal & (D.scen == "replay")]; g = sub.groupby(flag).dpDE.mean()
    out += [f"- replay ΔpDE by flag `{flag}`: " + ", ".join(f"{k}: {v:+.3f} (n={int((sub[flag] == k).sum())})" for k, v in g.items())]
out += ["", f"Baseline replay max price error among converged: {D[D.ok].base_err.max():.3f} EUR/MWh; median held wrong-sign bounds per MTU: {D.held.median():.0f}."]
open(sys.argv[2], "w").write("\n".join(out)); print("\n".join(out))
