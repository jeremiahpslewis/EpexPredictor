"""
Supplement to the pre-registered table: the gas actually removed. de_supply caps the outage at the flexible gas
running above its floor, so in some MTUs less than 1 GW is removed, while summarise11.py books other gas as dG + 1.
This recomputes, per draw-MTU and scenario, the GW removed (S_G(p) - S_G^outage(p) at the outage price), and reports:
(a) the share of MTUs with a partial cut, and (b) the table on MTUs where the full 1 GW is removed in every scenario.
Usage: fullcut11.py E_DIR OUT.md
"""
import sys, pickle, numpy as np, pandas as pd
E, OUT = sys.argv[1], sys.argv[2]; sys.argv = ["cf11.py", f"{E}/recs2000_v2.pkl", "/dev/null", "1", f"{E}/post_mode0.npz"]
sys.path.insert(0, __file__.rsplit("/", 1)[0]); import cf11 as C, net11 as N
recs = {str(r["t"]): r for r in pickle.load(open(f"{E}/recs2000_v2.pkl", "rb"))["recs"] if r["kind"] == "outage"}
P = {b: dict(np.load(f"{E}/post_{b}.npz", allow_pickle=True)) for b in ("mode0", "mode1")}
SC = ["replay", "eta0.10", "eta0.25", "eta0.50", "eta1.00"]; R = []
for row in pickle.load(open(f"{E}/cf11_v2.pkl", "rb")):
    b = row["post"].split("post_")[-1].split(".")[0]; i = row["draw"]; r = recs[row["t"]]
    th = {k: float(P[b][k][i]) for k in [str(x) for x in P[b]["de_keys"]]}
    for s in SC:
        o = row["res"][s]; ok = o["ok"] and not o["cap"]
        cut = np.nan
        if ok:
            po = r["p"][0] + o["dp"][0]
            cut = C.de_supply(th, r["de"], po)[0]["G"] - C.de_supply(th, r["de"], po, 1.0)[0]["G"]
        R.append(dict(basin=b, draw=i, t=row["t"], scen=s, ok=ok, cut=cut, dpDE=o["dp"][0], imports=o["dimport"], lignite=o["dL"], coal=o["dC"],
                      other_gas=o["dG"] + cut if ok else np.nan, **{z: o["dS"][k] for k, z in enumerate(N.Z) if z != "DE"}))
D = pd.DataFrame(R)
g = D.groupby(["basin", "draw", "t"]); D["bal"] = g.ok.transform("all"); D["full"] = D.bal & g.cut.transform(lambda c: (c > 0.99).all())
out = ["# Outage study: gas actually removed (supplement)", "",
       f"Balanced draw-MTUs with a partial cut (< 0.99 GW removed in some scenario): {1 - D[D.bal].groupby(['basin','draw','t']).full.first().mean():.1%}. "
       f"Mean GW removed on the balanced panel at replay: {D[D.bal & (D.scen == 'replay')].cut.mean():.3f}.", ""]
F = D[D.full]; cols = ["dpDE", "imports", "lignite", "coal", "other_gas"] + [z for z in N.Z if z != "DE"]
m = F.groupby(["basin", "scen", "draw"])[cols].mean().groupby(["basin", "scen"]).mean()
out += [f"## Full-cut panel ({F.groupby(['basin','draw','t']).ngroups} draw-MTUs): mean over MTUs, GW per GW removed", "", m.round(3).to_markdown(), ""]
med = F.groupby(["basin", "scen"]).dpDE.median(); own = m.lignite + m.coal + m.other_gas
out += ["Median ΔpDE on the full-cut panel:", "", med.round(3).to_frame().to_markdown(), "", "## Reading rule on the full-cut panel", ""]
for b in ("mode0", "mode1"):
    p10, p100 = med[(b, "eta0.10")], med[(b, "eta1.00")]; s10, s100 = own[(b, "eta0.10")], own[(b, "eta1.00")]
    out += [f"- {b}: median ΔpDE {p10:+.2f} vs {p100:+.2f} ({p10 / p100 - 1:+.0%}); DE own {s10:.2f} vs {s100:.2f} (Δ {s10 - s100:+.2f}); "
            f"material: {abs(p10 - p100) > 0.3 * abs(p100) or abs(s10 - s100) > 0.15}"]
open(OUT, "w").write("\n".join(out)); print("\n".join(out))
