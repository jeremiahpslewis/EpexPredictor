"""P10: how prices rose with gas at each load level, same season a year apart, against what each functional form implies."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import live_common as lc
SP = lc.SP; T = "2026-09-18"; d = np.load(f"{SP}/out/struct_pch/debug_{T}.npz"); idx = lc.load()["DE"].index; t = idx[d["pos"]].tz_convert("Europe/Berlin"); eua = np.load(f"{SP}/out/eua_15min.npz")["eua"][d["pos"]]
A = (t >= "2025-07-20") & (t < "2025-09-18"); B = (t >= "2026-07-20") & (t < "2026-09-18"); gA, gB = float(d["gas"][A].mean()), float(d["gas"][B].mean()); cA, cB = float(np.nanmean(eua[A])), float(np.nanmean(eua[B])); LA, LB = 1.85 * gA + 0.37 * cA, 1.85 * gB + 0.37 * cB
edges = np.arange(10, 56, 5.0); rng = np.random.default_rng(0); rows = []
for lo in edges[:-1]:
    a = d["y"][A & (d["x"] >= lo) & (d["x"] < lo + 5)]; b = d["y"][B & (d["x"] >= lo) & (d["x"] < lo + 5)]
    if len(a) < 40 or len(b) < 40: continue
    da = pd.Series(a).groupby(d["day"][A & (d["x"] >= lo) & (d["x"] < lo + 5)]).median().to_numpy(); db = pd.Series(b).groupby(d["day"][B & (d["x"] >= lo) & (d["x"] < lo + 5)]).median().to_numpy()   # day-level medians: errors come as whole days
    r = [np.median(rng.choice(db, len(db))) / np.median(rng.choice(da, len(da))) for _ in range(2000)]; rows.append(dict(x=lo + 2.5, pA=np.median(a) / 10, pB=np.median(b) / 10, ratio=np.median(b) / np.median(a), lo=np.percentile(r, 5), hi=np.percentile(r, 95), nA=len(a), nB=len(b), dA=len(da), dB=len(db)))
R = pd.DataFrame(rows); form = {}
for nm, name in (("as the candidate", "pch"), ("with the prior", "pcg")):
    z = np.load(f"{SP}/out/struct_{name}/{T}.npz"); b = np.interp(R.x, z["grid"], z["b_curve"]); h = np.interp(R.x, z["grid"], z["h_curve"]); form[nm] = (b + gB * h) / (b + gA * h)
    p = np.load(f"{SP}/out/struct_{name}/par_{T}.npz"); s = np.exp(np.concatenate([-np.cumsum((0.02 * p["v_e"])[::-1])[::-1], np.zeros(1)])); wkA, wkB = d["week"][A], d["week"][B]; form[nm + " scale"] = float(s[wkB].mean() / s[wkA].mean())
R["cand"] = form["as the candidate"]; R["prior"] = form["with the prior"]; R.to_csv(f"{SP}/out/fit_p10.csv", index=False); pd.set_option("display.width", 200)
print(f"period A 20 Jul - 17 Sep 2025: gas {gA:.0f}, carbon {cA:.0f}, gas-fired cost {LA:.0f} | period B, 2026: gas {gB:.0f}, carbon {cB:.0f}, gas-fired cost {LB:.0f} | cost ratio {LB / LA:.2f}, gas ratio {gB / gA:.2f}")
print(f"weekly price scale, period B / period A: as the candidate {form['as the candidate scale']:.2f}, with the prior {form['with the prior scale']:.2f}"); print(R.round(2).to_string(index=False))
np.savez(f"{SP}/out/fit_p10.npz", LA=LA, LB=LB, gA=gA, gB=gB, sc_c=form["as the candidate scale"], sc_p=form["with the prior scale"])
