"""Offline sizing of cross-border / fundamental drivers against the champion's day-by-window level miss.

Target: e[d, w] = mean over window w of (issued blend - actual) on day d, champion out_strict (61 days 09-08..11-07,
ALREADY LOOKED AT) and out_fresh_strict (18 days 11-09..11-26, already looked at by S/anatomy; neighbours' ENTSO-E
data ends 11-09, so fresh days only for gas / EUA / FR nuclear).
Candidates x[d, w] (see TIMING in the README of this folder for when each is published):
  nb_load_<z> : neighbour z's day-ahead load forecast (ENTSO-E 6.1.B), window mean on D minus the same window on D-1, GW
  nb_ws_<z>   : neighbour z's day-ahead wind+solar forecast (14.1.D), same change. NOT gate-known (sizing only)
  nb_rl_<z>   : load - wind - solar, same change (NOT gate-known: contains 14.1.D)
  fr_nuc      : FR nuclear unavailable GW (pregate view) on D minus on D-1; fr_nuc_lvl its level on D
  gas         : THE gas day D-1 minus gas day D-2 (EUR/MWh); EUA: last auction <= D-2 minus the one before
  de_rl_err   : DE tree RL error (forecast - true), window mean, GW: the RL lever's proxy, for reference
Score: causal walk-forward linear correction of the blend, e ~ a_w + b x (OLS on all days strictly before d, from
day 15), blend_corr = issued - pred, MAE over all quarter hours vs the champion and vs the intercept-only (a_w) model.
usage: size_drivers.py SP
"""
import glob, os, sys
import numpy as np, pandas as pd

SP = sys.argv[1]
NB = f"{SP}/entsoe/neighbours/derived"
ZONES = ["FR", "NL", "BE", "AT", "CZ", "PL", "CH", "DK_1", "DK_2", "SE_4", "NO_2"]
WIN = [-1, 5, 9, 15, 20, 23]; WL = ["00-06", "06-10", "10-16", "16-21", "21-24"]


def load_champ(d):
    rows = []
    for f in sorted(glob.glob(f"{SP}/faithful/{d}/*.npz")):
        z = np.load(f)
        t = pd.to_datetime(z["idx"], unit="us", utc=True)
        rows.append(pd.DataFrame({"t": t, "day": os.path.basename(f)[:-4], "issued": z["issued"], "actual": z["actual"],
                                  "trees": z["trees"], "timestral": z["timestral"], "market": z["market"],
                                  "rle": z["rl_trees"] - z["rl_true"]}))
    return pd.concat(rows, ignore_index=True)


Q = pd.concat([load_champ("out_strict").assign(set="61"), load_champ("out_fresh_strict").assign(set="fresh18")], ignore_index=True)
Q["tl"] = Q.t.dt.tz_convert("Europe/Berlin")
Q["win"] = pd.cut(Q.tl.dt.hour, WIN, labels=WL).astype(str)
Q["e"] = Q.issued - Q.actual


def winmean(s: pd.Series, name):
    """15-min series (tz-aware) -> DataFrame day (local D), win, value; plus the same window on D-1 for the change"""
    s = s.copy(); s.index = s.index.tz_convert("Europe/Berlin")
    df = pd.DataFrame({"v": s.values, "day": s.index.strftime("%Y-%m-%d"), "win": pd.cut(s.index.hour, WIN, labels=WL).astype(str)})
    m = df.groupby(["day", "win"]).v.mean().unstack()
    prev = m.copy(); prev.index = (pd.to_datetime(prev.index) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    ch = (m - prev.reindex(m.index)).stack().rename(name)
    return ch, m.stack().rename(name + "_lvl")


feats = []
for z in ZONES:
    try:
        ld = pd.read_parquet(f"{NB}/{z}_load_15min.parquet").iloc[:, 0] / 1000
        ws = pd.read_parquet(f"{NB}/{z}_ws_15min.parquet").sum(axis=1, min_count=1) / 1000
    except FileNotFoundError:
        continue
    ws = ws.reindex(ld.index).interpolate(limit=4)
    for nm, s in (("nb_load_" + z, ld), ("nb_ws_" + z, ws), ("nb_rl_" + z, ld - ws)):
        ch, lv = winmean(s, nm); feats += [ch, lv]
# sums over the coupled neighbours (CWE + PL/CZ + DK)
core = ["FR", "NL", "BE", "AT", "CZ", "PL", "CH", "DK_1", "DK_2"]
for kind in ("load", "ws", "rl"):
    parts = []
    for z in core:
        fn = f"{NB}/{z}_{'ws' if kind == 'ws' else 'load'}_15min.parquet"
        ld = pd.read_parquet(f"{NB}/{z}_load_15min.parquet").iloc[:, 0] / 1000
        ws = pd.read_parquet(f"{NB}/{z}_ws_15min.parquet").sum(axis=1, min_count=1).reindex(ld.index).interpolate(limit=4) / 1000
        parts.append({"load": ld, "ws": ws, "rl": ld - ws}[kind])
    ch, lv = winmean(sum(parts), f"nb_{kind}_ALL"); feats += [ch, lv]
F = pd.concat(feats, axis=1)
F.index.names = ["day", "win"]

# daily (window-independent) drivers
days = sorted(Q.day.unique())
dd = pd.DataFrame(index=days)
fr = pd.read_parquet(f"{SP}/team/I/fr_nuc.parquet")
fr.index = fr.index.tz_convert("UTC").strftime("%Y-%m-%d")   # value at D 12:00 UTC (pregate = started by D-1 12:00)
dd["fr_nuc_lvl"] = fr["pregate"].reindex(days).values
dd["fr_nuc"] = (fr["pregate"] - fr["pregate"].shift(1)).reindex(days).values
g = pd.read_json(f"{SP}/live_data/gasprices_the_v1_DE_LU.json.gz")["gasprice"]
gd = g.resample("1D").last()          # gas day indexed at its 00:00 UTC; gas day D-1 is known at the gate (app)
gd.index = gd.index.strftime("%Y-%m-%d")
for d in days:
    dm1 = (pd.Timestamp(d) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"); dm2 = (pd.Timestamp(d) - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    dd.loc[d, "gas"] = gd.get(dm1, np.nan) - gd.get(dm2, np.nan)
    dd.loc[d, "gas_lvl"] = gd.get(dm1, np.nan)
x = pd.read_excel(f"{SP}/ext/eex/eua2026.xlsx", header=None, skiprows=6).iloc[:, [1, 6]].dropna()
x.columns = ["date", "p"]; x["date"] = pd.to_datetime(x["date"]); x = x.sort_values("date").groupby("date").p.mean()
for d in days:
    known = x[x.index <= pd.Timestamp(d) - pd.Timedelta(days=2)]
    dd.loc[d, "eua"] = known.iloc[-1] - known.iloc[-2] if len(known) > 1 else np.nan
    dd.loc[d, "eua_lvl"] = known.iloc[-1] if len(known) else np.nan

E = Q.groupby(["day", "win"]).agg(e=("e", "mean"), rle=("rle", "mean"), set=("set", "first")).reset_index()
E = E.merge(F.reset_index(), on=["day", "win"], how="left").merge(dd, left_on="day", right_index=True, how="left")
E = E.rename(columns={"rle": "de_rl_err"})
Q = Q.merge(E.drop(columns=["e", "set"]), on=["day", "win"], how="left")
Q = Q.sort_values(["day", "t"]).reset_index(drop=True)

base_mae = lambda q: float(np.abs(q.e).mean())


def walk(q: pd.DataFrame, cols, ridge=0.0, warm=14):
    """causal per-day correction; returns per-day (champ MAE, intercept-only MAE, with-x MAE)"""
    E1 = q.groupby(["day", "win"]).agg(e=("e", "mean"), **{c: (c, "first") for c in cols}).reset_index()
    ds = sorted(q.day.unique()); out = []
    W = pd.get_dummies(E1.win).astype(float)
    for i, d in enumerate(ds):
        if i < warm:
            continue
        tr = E1.day.isin(ds[:i]) & E1[cols].notna().all(axis=1); te = E1.day == d
        if te.sum() == 0 or E1.loc[te, cols].isna().any().any():
            continue
        A0, A1 = W[tr].to_numpy(), np.c_[W[tr].to_numpy(), E1.loc[tr, cols].to_numpy()]
        y = E1.loc[tr, "e"].to_numpy()
        b0 = np.linalg.lstsq(A0, y, rcond=None)[0]
        R = ridge * np.eye(A1.shape[1]); R[:W.shape[1], :W.shape[1]] = 0
        b1 = np.linalg.solve(A1.T @ A1 + R, A1.T @ y)
        p0 = W[te].to_numpy() @ b0; p1 = np.c_[W[te].to_numpy(), E1.loc[te, cols].to_numpy()] @ b1
        pw0, pw1 = dict(zip(E1.loc[te, "win"], p0)), dict(zip(E1.loc[te, "win"], p1))
        qd = q[q.day == d]
        out.append((d, np.abs(qd.e).mean(), np.abs(qd.e - qd.win.map(pw0)).mean(), np.abs(qd.e - qd.win.map(pw1)).mean()))
    return pd.DataFrame(out, columns=["day", "champ", "icpt", "corr"]).set_index("day")


rng = np.random.default_rng(1)


def ci(v, n=3000, blk=7):
    v = np.asarray(v); n0 = len(v); st = []
    for _ in range(n):
        s = rng.integers(0, n0, int(np.ceil(n0 / blk)))
        idx = np.concatenate([(np.arange(k, k + blk) % n0) for k in s])[:n0]
        st.append(v[idx].mean())
    return np.percentile(st, [5, 95])


cands = [c for c in E.columns if c not in ("day", "win", "e", "set")]
if os.environ.get("MODE") == "combo":   # combinations given the RL proxy (61 days), then exit
    q61 = Q[Q.set == "61"]
    for cols in (["de_rl_err"], ["de_rl_err", "nb_ws_BE"], ["de_rl_err", "nb_ws_ALL"], ["de_rl_err", "nb_rl_ALL"],
                 ["de_rl_err", "nb_ws_FR", "nb_ws_NL", "nb_ws_BE"], ["nb_ws_BE", "nb_ws_FR", "nb_ws_NL"],
                 ["de_rl_err", "nb_rl_FR"], ["de_rl_err", "fr_nuc_lvl"], ["de_rl_err", "eua"], ["nb_ws_ALL", "nb_load_ALL"]):
        for ridge in (0.0, 5.0):
            wf = walk(q61, cols, ridge=ridge); dl = wf["corr"] - wf["champ"]; lo, hi = ci(dl.to_numpy())
            print(f"  {'+'.join(cols):40s} ridge {ridge:3.0f} days {len(wf)} corr-champ {dl.mean():+.3f} [{lo:+.3f},{hi:+.3f}] corr-icpt {(wf['corr'] - wf['icpt']).mean():+.3f}")
    sys.exit()
print("=== univariate: corr with the day-window blend miss e (issued - actual), and causal walk-forward correction")
print("    61-day set (09-08..11-07, looked at). corr_w: within-window corr after removing the window mean")
rows = []
q61 = Q[Q.set == "61"]
E61 = E[E.set == "61"]
for c in cands:
    ok = E61[c].notna()
    if ok.sum() < 100:
        continue
    r = np.corrcoef(E61.loc[ok, c], E61.loc[ok, "e"])[0, 1]
    wf = walk(q61, [c])
    dlt = wf["corr"] - wf["champ"]; dli = wf["corr"] - wf["icpt"]
    lo, hi = ci(dlt.to_numpy())
    rows.append((c, ok.sum(), r, len(wf), wf.champ.mean(), wf.icpt.mean() - wf.champ.mean(), dlt.mean(), lo, hi, dli.mean()))
R = pd.DataFrame(rows, columns=["cand", "n", "corr", "days", "champ", "icpt-champ", "corr-champ", "lo90", "hi90", "corr-icpt"]).sort_values("corr-champ")
print(R.round(3).to_string(index=False))
R.to_csv(f"{SP}/team/I/size_drivers_61.csv", index=False)

print("\n=== fresh 18 days (11-09..11-26), daily drivers only, walk-forward trained on 61 + earlier fresh days")
qa = Q.copy()
for c in ["fr_nuc", "fr_nuc_lvl", "gas", "gas_lvl", "eua", "eua_lvl", "de_rl_err"]:
    wf = walk(qa, [c], warm=61)
    wf = wf[wf.index >= "2026-11-09"]
    if len(wf):
        d = wf["corr"] - wf["champ"]
        print(f"  {c:12s} days {len(wf):2d} champ {wf.champ.mean():.3f} icpt {wf.icpt.mean() - wf.champ.mean():+.3f} with x {d.mean():+.3f}  vs icpt {(wf['corr'] - wf['icpt']).mean():+.3f}")
E.to_parquet(f"{SP}/team/I/daywin_frame.parquet")
