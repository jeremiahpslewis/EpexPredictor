"""
Is the multi-hour margin response of German gas output (lambda ~0.84-0.93 on m + lambda (m_win - m)) unit commitment?
Per-unit ENTSO-E actual generation (gas, >=100 MW; 50HZ, AMPRION, TENNET), 2025-08-12 .. 2026-08-11.
Data are sampled on alternate days, so every state/run statistic is computed within contiguous observed stretches only,
with left/right censoring tracked. Usage: unit_commitment.py DATA_DIR FIG_PATH. Prints JSON; writes only the figure.
"""
import glob, gzip, json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

SP, FIG = Path(sys.argv[1]), Path(sys.argv[2])
T0, T1 = pd.Timestamp("2025-08-12", tz="Europe/Berlin"), pd.Timestamp("2026-08-12", tz="Europe/Berlin")
THR, ALWAYS = 0.20, 0.90
rng = np.random.default_rng(0); R = {}

# ---------- system margin on the full continuous grid (as gas_curve2.py) ----------
X = pd.read_parquet(SP / "out/recon_hourly.parquet")
with gzip.open(SP / "appdata/gasprices_the_v1_DE_LU.json.gz", "rt") as f: g = json.load(f)["gasprice"]
gas = pd.Series({pd.Timestamp(int(k), unit="ms", tz="UTC"): v for k, v in g.items()}).sort_index()
eua = pd.read_parquet(SP / "out/eua_daily.parquet")["eua"].sort_index()
X["gas"] = gas.reindex(X.index, method="ffill")
X["co2"] = eua.reindex(X.index.tz_convert("Europe/Berlin").tz_localize(None).normalize(), method="ffill").to_numpy()
X["m"] = X.p - (1.85 * X.gas + 0.37 * X.co2)
grid = X.m.resample("1h").mean()
X["m_win"] = grid.rolling(7, center=True, min_periods=5).mean().reindex(X.index)
X["dm"] = (grid.shift(-1) - grid.shift(1)).reindex(X.index)

# ---------- units ----------
fr = []
for t in ["50HZ", "AMPRION", "TENNET"]:
    parts = [pd.read_parquet(f) for f in sorted(glob.glob(str(SP / f"entsoe/units_all/DE_{t}_gas_*.parquet")))]
    d = pd.concat(parts).sort_index(); d = d[~d.index.duplicated()]
    d = d[[c for c in d.columns if c.endswith("Fossil Gas | Actual Aggregated")]]
    d.columns = [f"{t}:{c.split(' | ')[0]}" for c in d.columns]
    fr.append(d.tz_convert("UTC"))
U = pd.concat(fr, axis=1).sort_index()
U = U[(U.index >= T0) & (U.index < T1)]
cap = U.quantile(0.99)
U = U.loc[:, cap >= 50]; cap = cap[U.columns]            # drop units that never produce meaningfully
S = X.reindex(U.index)
R["sampling"] = dict(units=int(U.shape[1]), hours=int(len(U)), days=int(len(set(U.index.tz_convert("Europe/Berlin").date))),
                     cap_GW=round(float(cap.sum()) / 1e3, 2), nan_share=round(float(U.isna().values.mean()), 4))
cov = U.sum(axis=1, min_count=1) / 1e3 / S.gasg
R["sampling"]["unit_sum_over_gasg_median"] = round(float(cov.median()), 3)

ON = (U.values > THR * cap.values) .astype(float); ON[np.isnan(U.values)] = np.nan
onshare = np.nanmean(ON, 0)
cls = np.where(onshare >= ALWAYS, "always", np.where(onshare > 0.01, "cycler", "rare"))
R["classes"] = {k: dict(n=int((cls == k).sum()), cap_GW=round(float(cap.values[cls == k].sum()) / 1e3, 2)) for k in ["always", "cycler", "rare"]}

ut = U.index.asi8 // 3_600_000_000
hour_loc = U.index.tz_convert("Europe/Berlin").hour.to_numpy()
day_loc = U.index.tz_convert("Europe/Berlin").normalize()
m = S.m.to_numpy(); month = U.index.tz_convert("Europe/Berlin").month.to_numpy()
winter = np.isin(month, [10, 11, 12, 1, 2, 3])

# ---------- runs and down-times per unit, within contiguous observed stretches ----------
runs = []   # (unit, state, i0, i1 exclusive, left_cens, right_cens)
for j in range(U.shape[1]):
    o = ON[:, j]; ok = ~np.isnan(o)
    brk = np.r_[True, (np.diff(ut) != 1) | (ok[1:] != ok[:-1])]
    seg = np.r_[np.where(brk)[0], len(o)]
    for a, b in zip(seg[:-1], seg[1:]):
        if not ok[a]: continue
        x = o[a:b]; ch = np.r_[0, np.where(np.diff(x) != 0)[0] + 1, len(x)]
        for c0, c1 in zip(ch[:-1], ch[1:]):
            runs.append((j, int(x[c0]), a + c0, a + c1, c0 == 0, c1 == len(x), a + len(x)))
rd = pd.DataFrame(runs, columns=["j", "on", "i0", "i1", "lc", "rc", "send"])
rd["len"] = rd.i1 - rd.i0; rd["cls"] = cls[rd.j]
rd["type"] = np.select([rd.lc & rd.rc, rd.lc, rd.rc], ["whole", "left", "right"], "complete")
cyc = rd[rd.cls == "cycler"]
R["run_types_cyclers"] = {s: cyc[cyc.on == s].type.value_counts(normalize=True).round(3).to_dict() for s in (1, 0)}

def km(L, ev):
    """Kaplan-Meier survival S(t) for t = 1..24 from lengths L with event flags ev."""
    out = []; s = 1.0
    for t in range(1, 25):
        at = (L >= t).sum(); e = ((L == t) & ev).sum()
        if at: s *= 1 - e / at
        out.append(s)
    return np.array(out)

def len_stats(df):
    st = df[~df.lc]                                  # observed start
    elig = st[(st.send - st.i0) >= 3]                # at least 3 h of observation after the start
    short = float(((elig.len < 3) & ~elig.rc).mean())
    S_ = km(st.len.to_numpy(), (~st.rc).to_numpy())
    med = int(np.argmax(S_ <= 0.5) + 1) if (S_ <= 0.5).any() else ">24"
    return dict(n_obs_start=int(len(st)), share_lt3h=round(short, 3), km_median_h=med,
                km_S=S_.round(3).tolist(), complete_median_h=float(st[~st.rc].len.median()),
                share_lt6h=round(float(((elig.len < 6) & ~elig.rc)[(elig.send - elig.i0) >= 6].mean()), 3))
R["run_lengths"] = {"runs": len_stats(cyc[cyc.on == 1]), "downs": len_stats(cyc[cyc.on == 0])}
for k in ("runs", "downs"): R["run_lengths"][k].pop("km_S")
KM_RUN, KM_DOWN = km(*(lambda d: (d.len.to_numpy(), (~d.rc).to_numpy()))(cyc[(cyc.on == 1) & ~cyc.lc])), km(*(lambda d: (d.len.to_numpy(), (~d.rc).to_numpy()))(cyc[(cyc.on == 0) & ~cyc.lc]))
# all units incl. always-on, for reference
R["run_lengths"]["runs_all_units_share_lt3h"] = len_stats(rd[rd.on == 1])["share_lt3h"]

# ---------- E2: starts and the margin ----------
st = cyc[(cyc.on == 1) & ~cyc.lc].copy()
st["m_start"] = m[st.i0]
st["m_run"] = [np.nanmean(m[a:b]) for a, b in zip(st.i0, st.i1)]
st["neg_h"] = [np.nansum(m[a:b] < 0) for a, b in zip(st.i0, st.i1)]
st["win"] = winter[st.i0]
st["cap"] = cap.values[st.j]
# implied gross margin per MW of capacity over the run (EUR/MW): output/cap * m summed over the run
Uv = U.values
st["gm_per_MW"] = [np.nansum(Uv[a:b, j] / cap.values[j] * m[a:b]) for j, a, b in zip(st.j, st.i0, st.i1)]
comp = st[~st.rc]
def e2(d, c):
    return dict(n=int(len(d)), median_m_start=round(float(d.m_start.median()), 1), median_m_run=round(float(d.m_run.median()), 1),
                median_run_minus_start=round(float((d.m_run - d.m_start).median()), 1),
                share_start_neg=round(float((d.m_start < 0).mean()), 3),
                share_runhours_neg=round(float(d.neg_h.sum() / d.len.sum()), 3),
                share_runs_mrun_pos=round(float((d.m_run > 0).mean()), 3),
                share_runs_mrun_pos_and_start_neg=round(float(((d.m_run > 0) & (d.m_start < 0)).mean()), 3),
                complete_gm_per_MW_median=round(float(c.gm_per_MW.median()), 0),
                complete_gm_per_MW_q25=round(float(c.gm_per_MW.quantile(0.25)), 0))
R["starts"] = {"all": e2(st, comp), "winter": e2(st[st.win], comp[comp.win]), "summer": e2(st[~st.win], comp[~comp.win])}
# null: cycler off-hours at positive margin; cycler on-hours at negative margin (all on-hours)
cj = np.where(cls == "cycler")[0]; Oc = ON[:, cj]
mm = np.repeat(m[:, None], len(cj), 1); okm = ~np.isnan(Oc) & ~np.isnan(mm)
R["null"] = dict(share_cycler_onhours_negm=round(float(((Oc == 1) & (mm < 0) & okm).sum() / ((Oc == 1) & okm).sum()), 3),
                 share_cycler_offhours_posm=round(float(((Oc == 0) & (mm > 0) & okm).sum() / ((Oc == 0) & okm).sum()), 3),
                 share_all_hours_negm=round(float(np.nanmean(m[~np.isnan(m)] < 0)), 3))
# margin event study around observed starts (cyclers)
ev = np.full((len(st), 16), np.nan)
for k, (i0, send) in enumerate(zip(st.i0, st.send)):
    for r_ in range(-3, 13):
        ii = i0 + r_
        if 0 <= ii < len(m) and ut[ii] - ut[i0] == r_: ev[k, r_ + 3] = m[ii]
EV = np.nanmedian(ev, 0)

# ---------- fleet decomposition (E1) ----------
# carried-over evening units: capacity of cycler runs started 16:00-19:59 (observed start) still on at 20-23 h same day
carry = np.zeros(len(U))
for j, a, b in zip(st.j, st.i0, st.i1):
    if 16 <= hour_loc[a] < 20:
        for i in range(a, b):
            if hour_loc[i] >= 20 and day_loc[i] == day_loc[a]: carry[i] += cap.values[j] / 1e3
Uf = np.where(np.isnan(Uv), 0, Uv); valid_h = np.isnan(Uv).mean(1) < 0.05
capon = np.nansum(np.where(ON == 1, cap.values, 0), 1) / 1e3   # GW of capacity on
Y = Uf.sum(1) / 1e3
capon_c = np.nansum(np.where(ON[:, cj] == 1, cap.values[cj], 0), 1) / 1e3
Yc = Uf[:, cj].sum(1) / 1e3
gap = (S.m_win - S.m).to_numpy()
lead = (grid.shift(-1) + grid.shift(-2) + grid.shift(-3)).div(3).reindex(U.index).to_numpy() - m
lag = (grid.shift(1) + grid.shift(2) + grid.shift(3)).div(3).reindex(U.index).to_numpy() - m
Non = np.nansum(ON == 1, 1); blk = np.digitize(hour_loc, [6, 10, 16, 20])
D = pd.DataFrame(dict(Y=Y, C=capon, u=Y / np.where(capon > 0, capon, np.nan), Yc=Yc, Cc=capon_c, gap=gap, m=m,
                      mb=np.floor(m / 5), mo=month, carry=carry, N=Non, G=S.gasg.to_numpy(), lead=lead, lag=lag, win=winter, b=blk, day=day_loc, h=hour_loc, dm=S.dm.to_numpy()))
D = D[valid_h & D.gap.notna() & D.lead.notna() & D.lag.notna() & D.m.notna() & (D.C > 0)].copy()
D["grp"] = D.mb.astype(int).astype(str) + "_" + D.mo.astype(str) + "_" + D.b.astype(str)            # m-bin x month x block
D["grp_h"] = D.mb.astype(int).astype(str) + "_" + D.h.astype(str) + "_" + D.win.astype(str)         # m-bin x clock hour x season
D["grp_h10"] = np.floor(D.m / 10).astype(int).astype(str) + "_" + D.h.astype(str) + "_" + D.mo.astype(str)  # 10-EUR bin x hour x month
D["grp_hm"] = D.h.astype(str) + "_" + D.mo.astype(str)                                             # for the response to m itself

VARS = ["Y", "C", "u", "N"]
def slopes_fast(d, grp, x="gap"):
    w = d[VARS + [x]] - d.groupby(grp)[VARS + [x]].transform("mean")
    return {v: float((w[x] * w[v]).sum() / (w[x] ** 2).sum()) for v in VARS}
ubar, Cbar = D.u.mean(), D.C.mean()
days = D.day.unique(); dgrp = {d_: idx for d_, idx in D.groupby("day").groups.items()}
def decomp(grp, x="gap", boot=True):
    sl_ = slopes_fast(D, grp, x); out = dict(slope_Y_GW_per_EUR=round(sl_["Y"], 4), slope_Ccap_on_GW_per_EUR=round(sl_["C"], 4),
        slope_units_on_per_EUR=round(sl_["N"], 4), share_via_capacity_on=round(ubar * sl_["C"] / sl_["Y"], 3),
        share_via_load_factor=round(Cbar * sl_["u"] / sl_["Y"], 3))
    if boot:
        bs = []
        for _ in range(300):
            pick = rng.choice(days, len(days)); b_ = slopes_fast(D.loc[np.concatenate([dgrp[d_] for d_ in pick])], grp, x)
            bs.append([b_["Y"], ubar * b_["C"] / b_["Y"]])
        bs = np.array(bs); out["slope_Y_90"] = np.quantile(bs[:, 0], [0.05, 0.95]).round(4).tolist()
        out["share_via_capacity_on_90"] = np.quantile(bs[:, 1], [0.05, 0.95]).round(3).tolist()
    return out, sl_
dec_h, sl = decomp("grp_h")
R["decomposition"] = dict(hours=int(len(D)), u_mean=round(float(ubar), 3), C_mean=round(float(Cbar), 2),
                          gap_mbin_hour_season=dec_h, gap_mbin10_hour_month=decomp("grp_h10", boot=False)[0],
                          gap_mbin_month_block=decomp("grp")[0], response_to_m_hour_month=decomp("grp_hm", x="m", boot=False)[0])
# lead vs lag: is the window response forward- or backward-looking? joint OLS after demeaning in m-bin x hour x season cells
w = D[["C", "Y", "lead", "lag"]] - D.groupby("grp_h")[["C", "Y", "lead", "lag"]].transform("mean")
def leadlag(w):
    Xl = w[["lead", "lag"]].to_numpy(); return {v: np.linalg.lstsq(Xl, w[v].to_numpy(), rcond=None)[0] for v in ["C", "Y"]}
ll = leadlag(w); bl = []
for _ in range(300):
    pick = rng.choice(days, len(days)); bl.append(np.r_[tuple(leadlag(w.loc[np.concatenate([dgrp[d_] for d_ in pick])]).values())])
bl = np.array(bl)
R["lead_lag"] = dict(C_lead=round(float(ll["C"][0]), 4), C_lag=round(float(ll["C"][1]), 4), Y_lead=round(float(ll["Y"][0]), 4), Y_lag=round(float(ll["Y"][1]), 4),
                     ci90_C_lead_lag_Y_lead_lag=np.quantile(bl, [0.05, 0.95], 0).round(4).T.tolist(), corr_lead_lag=round(float(np.corrcoef(w.lead, w.lag)[0, 1]), 3))

# ---------- logistic width of capacity-on vs margin by block (E4) ----------
def fit_logistic(mv, yv, mo, A=None):
    """y = F[month] + A sigmoid((m-mu)/s); grid over (mu, s), linear LS for F, A (or A held fixed)."""
    mos = np.unique(mo); Mo = (mo[:, None] == mos[None]).astype(float); best = None
    for mu in np.arange(-60, 41, 2.0):
        for s in np.exp(np.linspace(np.log(3), np.log(80), 40)):
            sg = 1 / (1 + np.exp(-(mv - mu) / s))
            if A is None: Z = np.c_[Mo, sg]; coef, *_ = np.linalg.lstsq(Z, yv, rcond=None); r = ((yv - Z @ coef) ** 2).sum()
            else: c0, *_ = np.linalg.lstsq(Mo, yv - A * sg, rcond=None); coef = np.r_[c0, A]; r = ((yv - A * sg - Mo @ c0) ** 2).sum()
            if coef[-1] > 0 and (best is None or r < best[0]): best = (r, mu, s, coef[-1])
    return dict(mu=round(best[1], 1), s=round(float(best[2]), 1), A=round(float(best[3]), 2))
mid_ = D[(D.h >= 10) & (D.h < 16)]
eve_ = D[D.h >= 20]
W = {"Y_midday": fit_logistic(mid_.m.to_numpy(), mid_.Y.to_numpy(), mid_.mo.to_numpy()),
     "Y_evening": fit_logistic(eve_.m.to_numpy(), eve_.Y.to_numpy(), eve_.mo.to_numpy()),
     "C_midday": fit_logistic(mid_.m.to_numpy(), mid_.C.to_numpy(), mid_.mo.to_numpy()),
     "C_evening": fit_logistic(eve_.m.to_numpy(), eve_.C.to_numpy(), eve_.mo.to_numpy()),
     "C_evening_minus_carry": fit_logistic(eve_.m.to_numpy(), (eve_.C - eve_.carry).to_numpy(), eve_.mo.to_numpy())}
allY = fit_logistic(D.m.to_numpy(), D.Y.to_numpy(), D.mo.to_numpy()); W["Y_allhours"] = allY
for blk_, d_ in [("midday", mid_), ("evening", eve_)]:
    W[f"Y_{blk_}_sharedA"] = fit_logistic(d_.m.to_numpy(), d_.Y.to_numpy(), d_.mo.to_numpy(), A=allY["A"])
allG = fit_logistic(D.m.to_numpy(), D.G.to_numpy(), D.mo.to_numpy())
for blk_, d_ in [("midday", mid_), ("evening", eve_)]:
    W[f"gasg_{blk_}"] = fit_logistic(d_.m.to_numpy(), d_.G.to_numpy(), d_.mo.to_numpy())
    W[f"gasg_{blk_}_sharedA"] = fit_logistic(d_.m.to_numpy(), d_.G.to_numpy(), d_.mo.to_numpy(), A=allG["A"])
W["ratio_gasg"] = round(W["gasg_evening"]["s"] / W["gasg_midday"]["s"], 2)
W["ratio_gasg_sharedA"] = round(W["gasg_evening_sharedA"]["s"] / W["gasg_midday_sharedA"]["s"], 2)
W["ratio_Y_sharedA"] = round(W["Y_evening_sharedA"]["s"] / W["Y_midday_sharedA"]["s"], 2)
for k in ["Y", "C"]: W[f"ratio_{k}"] = round(W[f"{k}_evening"]["s"] / W[f"{k}_midday"]["s"], 2)
W["ratio_C_minus_carry"] = round(W["C_evening_minus_carry"]["s"] / W["C_midday"]["s"], 2)
ev_st = st[(hour_loc[st.i0] >= 16) & (hour_loc[st.i0] < 20)]
on22 = []; neg_late = []; late = []
for j, a, b, send in zip(ev_st.j, ev_st.i0, ev_st.i1, ev_st.send):
    i22 = [i for i in range(a, send) if hour_loc[i] == 22 and day_loc[i] == day_loc[a]]
    if i22: on22.append(i22[0] < b)
    for i in range(a, b):
        if hour_loc[i] >= 21 and day_loc[i] == day_loc[a]:
            late.append(1); neg_late.append(m[i] < 0)
W.update(n_evening_starts=int(len(ev_st)), share_on_at_22=round(float(np.mean(on22)), 3),
         share_21_23_runhours_negm=round(float(np.mean(neg_late)), 3), n_21_23_runhours=int(len(late)),
         carry_mean_GW_20_24=round(float(eve_.carry.mean()), 3), C_mean_GW_20_24=round(float(eve_.C.mean()), 2))
R["evening"] = W

# ---------- E5: hysteresis, capacity on at equal margin, falling vs rising ----------
pooled = fit_logistic(D.m.to_numpy(), D.C.to_numpy(), D.mo.to_numpy())
nf = D[(np.abs(D.m - pooled["mu"]) < 2 * pooled["s"])].copy()
nf["dir"] = np.select([nf.dm < -5, nf.dm > 5], [-1, 1], 0); nf = nf[nf.dir != 0]
def hyst(d, keys, v="C"):
    g = d.groupby(keys + ["dir"])[v].mean().unstack("dir").dropna()
    n = d.groupby(keys + ["dir"]).size().unstack("dir").reindex(g.index)
    w = np.minimum(n[-1], n[1]); return float(((g[-1] - g[1]) * w).sum() / w.sum())
dgrp2 = {d_: idx for d_, idx in nf.groupby("day").groups.items()}; dys = list(dgrp2)
H = {}
for name, keys in [("pooled_mbin", ["mb"]), ("within_block", ["mb", "b"]), ("within_block_month", ["mb", "b", "mo"]), ("within_hour", ["mb", "h"])]:
    est = hyst(nf, keys); bb = []
    for _ in range(300):
        pick = rng.choice(dys, len(dys)); bb.append(hyst(nf.loc[np.concatenate([dgrp2[d_] for d_ in pick])], keys))
    H[name] = dict(GW=round(est, 3), ci90=np.quantile(bb, [0.05, 0.95]).round(3).tolist())
estN = hyst(nf, ["mb", "h"], "N"); bb = []
for _ in range(300):
    pick = rng.choice(dys, len(dys)); bb.append(hyst(nf.loc[np.concatenate([dgrp2[d_] for d_ in pick])], ["mb", "h"], "N"))
H["within_hour_units"] = dict(n_units=round(estN, 2), ci90=np.quantile(bb, [0.05, 0.95]).round(2).tolist())
H.update(n_falling=int((nf.dir == -1).sum()), n_rising=int((nf.dir == 1).sum()), nonflat=[round(pooled["mu"] - 2 * pooled["s"], 1), round(pooled["mu"] + 2 * pooled["s"], 1)])
H["gap_mean_falling"] = round(float(nf[nf.dir == -1].gap.mean()), 1); H["gap_mean_rising"] = round(float(nf[nf.dir == 1].gap.mean()), 1)
H["falling_hour_mean"] = round(float(nf[nf.dir == -1].h.mean()), 1); H["rising_hour_mean"] = round(float(nf[nf.dir == 1].h.mean()), 1)
R["hysteresis"] = H
print(json.dumps(R, indent=1, default=str))

# ---------- figure ----------
fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
a = ax[0, 0]; t = np.arange(1, 25)
a.step(t, KM_RUN, where="post", label="on-runs (cyclers)", color="#c0392b"); a.step(t, KM_DOWN, where="post", label="down-times (cyclers)", color="#2c3e50")
a.axvline(3, ls=":", c="grey"); a.set_xlabel("hours since start of state"); a.set_ylabel("share still in state (Kaplan-Meier)")
a.set_title(f"Run lengths: {R['run_lengths']['runs']['share_lt3h']:.0%} of runs < 3 h, median {R['run_lengths']['runs']['km_median_h']} h"); a.legend(); a.set_ylim(0, 1.02)
a = ax[0, 1]; rr = np.arange(-3, 13)
a.plot(rr, EV, "o-", c="#c0392b", label="median margin around start"); a.axhline(0, c="k", lw=0.8)
a.axhline(st.m_run.median(), ls="--", c="#c0392b", label=f"median run-average margin {st.m_run.median():.0f}")
a.axvline(0, ls=":", c="grey"); a.set_xlabel("hours relative to start hour (cyclers)"); a.set_ylabel("margin m, EUR/MWh"); a.legend(fontsize=8)
a.set_title(f"Starts: {R['starts']['all']['share_start_neg']:.0%} at m<0; {R['starts']['all']['share_runhours_neg']:.0%} of run-hours at m<0")
a = ax[1, 0]
lab = ["fleet output", "capacity on × mean load", "load factor × mean cap"]
vals = [sl["Y"], ubar * sl["C"], Cbar * sl["u"]]
a.bar(lab, vals, color=["#7f8c8d", "#c0392b", "#2c3e50"]); a.axhline(0, c="k", lw=0.8)
a.set_ylabel("GW per EUR/MWh of (m_win − m)\nat fixed m, clock hour, season"); a.set_title(f"Window response: {dec_h['share_via_capacity_on']:.0%} via units on/off (90% {dec_h['share_via_capacity_on_90'][0]:.0%}–{dec_h['share_via_capacity_on_90'][1]:.0%})")
a = ax[1, 1]
for dir_, c_, lb in [(-1, "#c0392b", "margin falling"), (1, "#2c3e50", "margin rising")]:
    q = nf[nf.dir == dir_].groupby("mb").C.agg(["mean", "size"]); q = q[q["size"] >= 15]
    a.plot(q.index * 5 + 2.5, q["mean"], "o-", c=c_, label=lb)
a.set_xlabel("current margin m, EUR/MWh (5-EUR bins)"); a.set_ylabel("capacity on, GW"); a.legend()
lo_, hi_ = H['within_hour']['ci90']
a.set_title(f"Hysteresis not detected: falling − rising = {H['within_hour']['GW']:+.2f} GW\nat equal m and clock hour (90% {lo_:+.2f} to {hi_:+.2f})", fontsize=10)
fig.suptitle(f"{dec_h['share_via_capacity_on']:.0%} of the gas window response is units switching on/off; runs last a median "
             f"{R['run_lengths']['runs']['km_median_h']} h, {R['run_lengths']['runs']['share_lt3h']:.0%} are < 3 h; no hysteresis, no wider evening curve\n"
             "German gas units ≥100 MW (50Hertz, Amprion, TenneT), ENTSO-E per-unit data on alternate days, 2025-08-12 to 2026-08-11", fontsize=11, weight="bold")
fig.tight_layout(); fig.savefig(FIG, dpi=130)
