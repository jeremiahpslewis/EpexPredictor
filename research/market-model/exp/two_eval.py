import sys, os, numpy as np
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
I = np.load(f"{SP}/out/inputs_autumn.npz"); a = I["actual"].astype(float); n = len(a); rng = np.random.default_rng(0)
ts = np.load(f"{SP}/out/tsl_123_8k.npz")["vint"].astype(float); ref = np.load(f"{SP}/out/rl_E_pes_T.npz")["vint"].astype(float)
def ci(p1, p0, sl):
    e1, e0 = np.abs(p1 - a)[:, sl].mean(1), np.abs(p0 - a)[:, sl].mean(1)
    r = [100 * (e1[i].mean() / e0[i].mean() - 1) for i in (np.concatenate([(s + np.arange(7)) % n for s in rng.integers(0, n, 10)])[:n] for _ in range(3000))]
    return f"{100 * (e1.mean() / e0.mean() - 1):+4.0f}% [{np.percentile(r, 2.5):+.0f}..{np.percentile(r, 97.5):+.0f}]"
def sc(p):
    e = np.abs(p - a); hi = a[:, :96] > 30; d = (p - a)[:, :96]
    return f"{e[:, :32].mean():6.2f}{e[:, :56].mean():8.2f}{e[:, 56:96].mean():8.2f}  {np.abs(d[hi]).mean():5.1f} /{d[hi].mean():+5.1f}"
rows = [("two-stage trees, tree RL forecast (as reported)", "rl_E_pes_T"), ("  same, next-day figures from an own forecast", "two_autumn_own_d1"), ("  Timestral RL forecast", "two_autumn_ts_d1"), ("  average of both RL forecasts", "two_autumn_avg_d1"),
        ("published RL known (as reported)", "rl_K_pes_T"), ("  same, next-day figures from an own forecast", "two_autumn_known_d1"), ("  same, next-day rows dropped", "two_autumn_known_drop")]
print(f"autumn, {n} forecasts, two-stage trees; MAE ct/kWh{'':<8}{'night':>6}{'34-48h':>8}{'48-58h':>8}  >30ct MAE/bias   34-48 h vs as reported      night vs as reported")
P = {}
for lab, f in rows:
    if not os.path.exists(f"{SP}/out/{f}.npz"): continue
    p = np.load(f"{SP}/out/{f}.npz")["vint"].astype(float); P[f] = p
    base = ref if "known" not in f and f != "rl_K_pes_T" else P.get("rl_K_pes_T", p)
    print(f"{lab:<50}{sc(p)}   {ci(p, base, slice(0, 56)) if p is not base else '':<26}  {ci(p, base, slice(0, 32)) if p is not base else ''}")
for lab, f in (("blend with Timestral price model, tree RL (PR #2)", "rl_E_pes_T"), ("blend, Timestral RL", "two_autumn_ts_d1"), ("blend, average RL", "two_autumn_avg_d1")):
    if f in P: print(f"{lab:<50}{sc((P[f] + ts) / 2)}   {ci((P[f] + ts) / 2, (ref + ts) / 2, slice(0, 56)) if f != 'rl_E_pes_T' else '':<26}  {ci((P[f] + ts) / 2, (ref + ts) / 2, slice(0, 32)) if f != 'rl_E_pes_T' else ''}")
