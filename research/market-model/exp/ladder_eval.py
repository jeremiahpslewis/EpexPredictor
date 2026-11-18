import os, numpy as np
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
L = np.load(f"{SP}/out/ladder.npz"); a = np.load(f"{SP}/out/live_ho.npz")["actual"].astype(float); assert (L["cutoffs"] == np.load(f"{SP}/out/live_ho.npz")["cutoffs"]).all()
rows = [("calendar + EU weather + gas (no German inputs)", "E_eu_gas_cal_no_rl"), ("residual load + calendar (no gas)", "R_rl_cal_only"), ("residual load + gas + calendar", "A_rl_gas_cal"), ("  + Europe-wide weather", "B_plus_eu"), ("  + German point weather (= stage 1)", "C_plus_de_weather"), ("stage 1 without residual load (PR #1 stage 1)", "N_no_rl")]
def sc(p):
    e = np.abs(p - a); hi = a[:, :96] > 30; d = (p - a)[:, :96]
    return f"{e[:, :32].mean():6.2f}{e[:, :56].mean():8.2f}{e[:, 56:96].mean():8.2f}   {np.abs(d[hi]).mean():5.1f} /{d[hi].mean():+5.1f}"
print(f"{'autumn, 66 forecasts, MAE ct/kWh':<50}{'--- own residual-load forecast ---':^44}{'--- published residual load as if known ---':^46}")
print(f"{'':<50}{'night':>6}{'34-48h':>8}{'48-58h':>8}   >30ct MAE/bias   {'night':>6}{'34-48h':>8}{'48-58h':>8}   >30ct MAE/bias")
for lab, k in rows:
    own = sc(L[f"{k}__own"].astype(float)); kn = sc(L[f"{k}__known"].astype(float)) if f"{k}__known" in L.files else ""
    print(f"{lab:<50}{own}      {kn}")
own2 = np.load(f"{SP}/out/rl_E_pes_T.npz")["vint"].astype(float); kn2 = np.load(f"{SP}/out/rl_K_pes_T.npz")["vint"].astype(float) if os.path.exists(f"{SP}/out/rl_K_pes_T.npz") else None
print(f"{'  + stage 2: other zones (= PR #2 trees)':<50}{sc(own2)}      {sc(kn2) if kn2 is not None else ''}")
print(f"{'PR #1 two-stage, no residual load':<50}{sc(np.load(f'{SP}/out/live_ho.npz')['vint'].astype(float))}")
