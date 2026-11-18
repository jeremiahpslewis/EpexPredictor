"""Forecast change per GW of residual load (= minus 1 GW of wind or solar) and per 10% gas, first forecast day, by tightness."""
import os, numpy as np
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
s = np.load(f"{SP}/out/sens_E.npz"); rl = s["rl_used"][:, :96].astype(float); base = s["base"][:, :96].astype(float)
d = lambda k: s[k][:, :96].astype(float) - base
lo, hi = np.nanpercentile(rl, [33, 67]); groups = {f"residual load < {lo:.0f} GW": rl < lo, f"{lo:.0f}-{hi:.0f} GW": (rl >= lo) & (rl < hi), f">= {hi:.0f} GW": rl >= hi}
print("Trees with residual load (PR #2), change of the forecast in EUR/MWh (= 10 x ct/kWh), mean [10th..90th percentile]")
f = lambda x: f"{10*x.mean():+6.1f} [{10*np.percentile(x,10):+.1f}..{10*np.percentile(x,90):+.1f}]"
print(f"{'':<26}{'per +1 GW (from -1/+1)':<28}{'+2 GW':<26}{'-2 GW':<26}{'gas +10%':<26}gas -10%")
for g, m in groups.items():
    print(f"{g:<26}{f((d('rl_p1') - d('rl_m1'))[m] / 2):<28}{f(d('rl_p2')[m]):<26}{f(d('rl_m2')[m]):<26}{f(d('gas_p10')[m]):<26}{f(d('gas_m10')[m])}")
m = np.ones_like(rl, bool); print(f"{'all':<26}{f((d('rl_p1') - d('rl_m1'))[m] / 2):<28}{f(d('rl_p2')[m]):<26}{f(d('rl_m2')[m]):<26}{f(d('gas_p10')[m]):<26}{f(d('gas_m10')[m])}")
x = (d("rl_p1") - d("rl_m1")) / 2; print(f"share of quarter hours where +1 GW lowers the forecast: {100*np.mean(x < -0.005):.0f}%; where it changes nothing (< 0.05 EUR/MWh): {100*np.mean(np.abs(x) < 0.005):.0f}%")
if all(os.path.exists(f"{SP}/out/tsens_{k}.npz") for k in (0, -1, 1)):
    t0, tm, tp = (np.load(f"{SP}/out/tsens_{k}.npz")["median"][:, :96].astype(float) for k in (0, -1, 1))
    idx = np.searchsorted(s["cutoffs"], np.load(f"{SP}/out/tsens_0.npz")["cutoffs"]); r = rl[idx]
    y = (tp - tm) / 2; xt = x[idx]
    print("\nTimestral with residual load (22 forecasts), per +1 GW, EUR/MWh; trees on the same forecasts for comparison")
    for g, (a_, b_) in {"low": (-1e9, lo), "middle": (lo, hi), "high": (hi, 1e9)}.items():
        m = (r >= a_) & (r < b_); print(f"  {g:<8} Timestral {f(y[m])}   trees {f(xt[m])}")
    print(f"  all      Timestral {f(y)}   trees {f(xt)}   correlation of the two per quarter hour {np.corrcoef(y.ravel(), xt.ravel())[0,1]:.2f}")
