"""Structural model runs vs the trees on the same origins. usage: seval.py PERIOD RUN [RUN...]"""
import sys, glob, os, numpy as np
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
period, runs = sys.argv[1], sys.argv[2:]
I = np.load(f"{SP}/out/inputs_{period}.npz"); tags = [str(c)[:10] for c in I["cutoffs"]]; a_all = I["actual"].astype(float)
B = {r: {os.path.basename(f)[:-4]: f for f in glob.glob(f"{SP}/out/struct_{r}/2*.npz")} for r in runs}
common = [i for i, t in enumerate(tags) if all(t in B[r] for r in runs)]; sel = np.array(common); a = a_all[sel]
if len(common) < len(tags) and not os.environ.get("ALLOW_PARTIAL"): sys.exit(f"INCOMPLETE: only {len(common)} of {len(tags)} forecasts present for {runs} - not evaluated (ALLOW_PARTIAL=1 to override)")
C = {}
LAD = f"{SP}/out/ladder_{period}.npz" if os.path.exists(f"{SP}/out/ladder_{period}.npz") else f"{SP}/out/ladder.npz"
TWO = {"autumn": ("rl_E_pes_T", "rl_K_pes_T")}.get(period, (f"two_{period}_own", f"two_{period}_known"))
if os.path.exists(LAD) and (period == "autumn" or "ladder_" in LAD):
    L = np.load(LAD)
    for lab, k in (("trees: RL + gas + calendar", "A_rl_gas_cal"), ("trees: + EU weather", "B_plus_eu"), ("trees: + German weather (stage 1)", "C_plus_de_weather")):
        if f"{k}__own" in L.files: C[lab] = (L[f"{k}__own"][sel].astype(float), L[f"{k}__known"][sel].astype(float))
    if all(os.path.exists(f"{SP}/out/{f}.npz") for f in TWO):
        C["trees: two stages (PR #2)"] = (np.load(f"{SP}/out/{TWO[0]}.npz")["vint"][sel].astype(float), np.load(f"{SP}/out/{TWO[1]}.npz")["vint"][sel].astype(float))
    if period == "autumn":
        ts = np.load(f"{SP}/out/tsl_123_8k.npz")["vint"][sel].astype(float); C["blend PR #2 (trees + Timestral)"] = ((C["trees: two stages (PR #2)"][0] + ts) / 2, (C["trees: two stages (PR #2)"][1] + ts) / 2)
variants = os.environ.get("VARIANTS", "").split(",") if os.environ.get("VARIANTS") else [""]
for r in runs:
    for v in variants:
        sfx = f"_{v}" if v else ""
        C[f"structural ({r}{sfx})"] = tuple(np.stack([np.load(B[r][tags[i]])[f"{arm}{sfx}"] for i in common]) for arm in ("own", "known"))
def sc(p):
    e = np.abs(p - a); hi = a[:, :96] > 30; d = (p - a)[:, :96]
    return f"{e[:, :32].mean():6.2f}{e[:, :56].mean():8.2f}{e[:, 56:96].mean():8.2f}  {np.abs(d[hi]).mean():5.1f} /{d[hi].mean():+5.1f}"
print(f"{period}: {len(common)} forecasts; MAE ct/kWh" + " " * 16 + f"{'--- own residual-load forecast ---':^42}{'--- published residual load known ---':^44}")
print(f"{'':<44}{'night':>6}{'34-48h':>8}{'48-58h':>8}  >30ct MAE/bias  {'night':>6}{'34-48h':>8}{'48-58h':>8}  >30ct MAE/bias")
for lab, (o, k) in C.items(): print(f"{lab:<44}{sc(o)}    {sc(k)}")
if os.environ.get("ENSEMBLE"):
    r = runs[-1]; s_o, s_k = C[f"structural ({r})"]; t_o, t_k = C["trees: two stages (PR #2)"]
    print(f"{'50/50 structural + two-stage trees':<44}{sc((s_o + t_o) / 2)}    {sc((s_k + t_k) / 2)}")
    print(f"{'1/3 each: structural, trees, Timestral':<44}{sc((s_o + t_o + ts) / 3)}    {sc((s_k + t_k + ts) / 3)}")
if os.environ.get("EXTRA"):
    r = runs[-1]
    for k in os.environ["EXTRA"].split(","):
        p = np.stack([np.load(B[r][tags[i]])[k] for i in common]); print(f"{'structural (' + r + ') ' + k:<44}{sc(p)}")
