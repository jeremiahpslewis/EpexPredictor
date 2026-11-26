"""Agent P, step 1: error anatomy of the trees and the market route (vs Timestral) on the stored faithful days.
Usage: anatomy.py  -> writes anatomy.md next to this file. Reads SP/faithful/out_strict (61 champion days) and
out_fresh_strict (18 fresh days). MAE ct/kWh; paired circular 7-day block bootstrap from distill/common.py."""
import os, sys, glob, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "distill"))
from common import SP, boot

M3 = ("trees", "timestral", "market")


def load(out, d0, d1):
    rows = []
    for fn in sorted(glob.glob(f"{SP}/faithful/{out}/*.npz")):
        d = os.path.basename(fn)[:10]
        if not (d0 <= d <= d1):
            continue
        z = np.load(fn)
        iv = np.asarray(z["idx"]); idx = pd.to_datetime(iv, unit="us" if np.abs(iv).max() < 1e17 else "ns", utc=True)
        n = len(idx)
        r = {"day": d, "idx": idx}
        for k in M3 + ("issued", "actual", "rl_trees", "rl_mixed", "rl_true"):
            r[k] = np.asarray(z[k], float)
        rows.append(r)
    return rows


def gas_daily():
    s = pd.read_json(f"{SP}/faithful/stores/gasprices_the_v1_DE_LU.json.gz")["gasprice"]
    s.index = s.index.tz_localize("UTC") if s.index.tz is None else s.index
    s = s.sort_index()
    return s


def frame(rows):
    """long frame: one row per quarter hour"""
    g = gas_daily()
    fr = []
    for r in rows:
        loc = r["idx"].tz_convert("Europe/Berlin")
        df = pd.DataFrame({k: r[k] for k in M3 + ("issued", "actual", "rl_trees", "rl_mixed", "rl_true")})
        df["day"] = r["day"]; df["h"] = loc.hour + loc.minute / 60; df["wd"] = loc.weekday
        D = pd.Timestamp(r["day"])
        gD1 = g.loc[:r["idx"][0] - pd.Timedelta(minutes=15)]
        df["gas_known"] = gD1.iloc[-1]                    # gas of D-1 (known at the gate)
        df["gas_D"] = g.reindex([r["idx"][0]], method="ffill").iloc[0] if r["idx"][0] <= g.index.max() else np.nan
        fr.append(df)
    df = pd.concat(fr, ignore_index=True)
    # D-1 same-slot price (naive day lag) and D-7 (naive week)
    return df


def mae(df, k, m=None):
    m = np.ones(len(df), bool) if m is None else np.asarray(m)
    return float(np.abs(df.loc[m, k] - df.loc[m, "actual"]).mean())


def tab_by(df, key, bins=None, labels=None):
    out = {}
    grp = pd.cut(df[key], bins, labels=labels, right=False) if bins is not None else df[key]
    for g, sub in df.groupby(grp, observed=True):
        out[str(g)] = {**{k: np.abs(sub[k] - sub["actual"]).mean() for k in M3 + ("issued",)},
                       **{f"bias {k}": (sub[k] - sub["actual"]).mean() for k in M3}, "n QH": len(sub)}
    return pd.DataFrame(out).T


def daily(df):
    e = {k: (df[k] - df["actual"]).abs().groupby(df["day"]).mean() for k in M3 + ("issued",)}
    out = pd.DataFrame(e)
    out["rl_err_trees"] = (df["rl_trees"] - df["rl_true"]).abs().groupby(df["day"]).mean()
    out["rl_bias_trees"] = (df["rl_trees"] - df["rl_true"]).groupby(df["day"]).mean()
    out["rl_err_mixed"] = (df["rl_mixed"] - df["rl_true"]).abs().groupby(df["day"]).mean()
    out["pmean"] = df["actual"].groupby(df["day"]).mean()
    out["pstd"] = df["actual"].groupby(df["day"]).std()
    out["nneg"] = (df["actual"] <= 0).groupby(df["day"]).sum()
    out["gas"] = df["gas_known"].groupby(df["day"]).first()
    out["wd"] = df["wd"].groupby(df["day"]).first()
    for k in M3:
        out[f"lvl {k}"] = (df[k] - df["actual"]).groupby(df["day"]).mean()
    return out


def section(name, df):
    L = [f"## {name}: {df['day'].nunique()} days", ""]
    L.append("Member MAE: " + ", ".join(f"{k} {mae(df, k):.3f}" for k in M3 + ("issued",)))
    L.append("")
    # level vs shape
    rows = {}
    for k in M3:
        e = (df[k] - df["actual"]); dm = e.groupby(df["day"]).transform("mean")
        rows[k] = {"MAE": e.abs().mean(), "bias": e.mean(), "MAE of daily mean error": e.groupby(df["day"]).mean().abs().mean(),
                   "MAE after removing daily mean": (e - dm).abs().mean()}
    L += ["### Level vs shape", "", pd.DataFrame(rows).T.round(3).to_markdown(), ""]
    L += ["### By local hour window (MAE, bias)", "", tab_by(df, "h", [0, 6, 8, 10, 13, 16, 18, 21, 24]).round(2).to_markdown(), ""]
    df = df.assign(pband=df["actual"])
    L += ["### By observed price band ct/kWh (conditions on outcome)", "", tab_by(df, "pband", [-99, 0, 5, 10, 15, 20, 99]).round(2).to_markdown(), ""]
    L += ["### By forecast residual load (rl_trees, GW)", "", tab_by(df, "rl_trees", [-50, 10, 20, 30, 40, 50, 99]).round(2).to_markdown(), ""]
    df = df.assign(rlerr=df["rl_trees"] - df["rl_true"])
    L += ["### By residual-load forecast error (rl_trees - true, GW)", "", tab_by(df, "rlerr", [-99, -6, -3, -1, 1, 3, 6, 99]).round(2).to_markdown(), ""]
    df = df.assign(daytype=np.where(df["wd"] >= 5, "weekend", "weekday"))
    L += ["### Weekday vs weekend", "", tab_by(df, "daytype").round(2).to_markdown(), ""]
    dd = daily(df)
    # correlations of daily member errors with daily drivers
    cor = {}
    for k in M3:
        cor[k] = {c: np.corrcoef(dd[k], dd[c])[0, 1] for c in ("rl_err_trees", "rl_err_mixed", "pstd", "pmean", "nneg", "gas")}
        cor[k]["corr(level err, rl_bias)"] = np.corrcoef(dd[f"lvl {k}"], dd["rl_bias_trees"])[0, 1]
        cor[k]["corr(level err, gas change)"] = np.corrcoef(dd[f"lvl {k}"][1:], np.diff(dd["gas"]))[0, 1] if len(dd) > 2 else np.nan
    L += ["### Correlation of daily MAE with daily drivers", "", pd.DataFrame(cor).T.round(2).to_markdown(), ""]
    # error correlations (complementarity)
    E = pd.DataFrame({k: df[k] - df["actual"] for k in M3})
    L += ["### Error correlations (QH / daily mean / demeaned)", "", E.corr().round(2).to_markdown(), "",
          E.groupby(df["day"]).mean().corr().round(2).to_markdown(), "",
          (E - E.groupby(df["day"]).transform("mean")).corr().round(2).to_markdown(), ""]
    # worst days
    w = dd.sort_values("issued", ascending=False).head(10)
    L += ["### Worst 10 days by blend MAE", "", w[list(M3) + ["issued", "rl_err_trees", "rl_bias_trees", "pmean", "pstd", "nneg", "gas"] + [f"lvl {k}" for k in M3]].round(2).to_markdown(), ""]
    # share of total abs error by window
    tot = {k: np.abs(df[k] - df["actual"]).sum() for k in M3}
    sh = {}
    for lo, hi in [(0, 6), (6, 10), (10, 16), (16, 21), (21, 24)]:
        m = (df["h"] >= lo) & (df["h"] < hi)
        sh[f"{lo:02d}-{hi:02d}"] = {k: np.abs(df.loc[m, k] - df.loc[m, "actual"]).sum() / tot[k] for k in M3}
    L += ["### Share of total absolute error by window", "", pd.DataFrame(sh).round(2).to_markdown(), ""]
    # oracle: best member per QH window vs per day
    L.append("Per-day oracle among the three members: " + f"{dd[list(M3)].min(1).mean():.3f}; best fixed member: {dd[list(M3)].mean().min():.3f}")
    # trees minus timestral per window, paired
    pp = []
    for k in ("trees", "market"):
        for lo, hi in [(0, 6), (6, 10), (10, 16), (16, 21), (21, 24), (0, 24)]:
            m = (df["h"] >= lo) & (df["h"] < hi)
            d = (np.abs(df.loc[m, k] - df.loc[m, "actual"]) - np.abs(df.loc[m, "timestral"] - df.loc[m, "actual"])).groupby(df.loc[m, "day"]).mean()
            b = boot(d.to_numpy())
            pp.append(f"- {k} − timestral, {lo:02d}-{hi:02d}: {d.mean():+.3f} [{b[0]:+.3f}, {b[1]:+.3f}]")
    L += ["", "### Paired vs Timestral by window", ""] + pp + [""]
    return L, dd


if __name__ == "__main__":
    out = ["# Agent P: error anatomy of the trees and the market route (post hoc; these days were looked at before)", ""]
    for name, o, d0, d1 in [("Champion 61 days 09-08..11-07", "out_strict", "2026-09-08", "2026-11-07"),
                            ("Fresh 18 days 11-09..11-26", "out_fresh_strict", "2026-11-09", "2026-11-26")]:
        df = frame(load(o, d0, d1))
        L, dd = section(name, df)
        out += L
        dd.to_csv(f"{SP}/P/daily_{o}.csv")
        df.to_parquet(f"{SP}/P/qh_{o}.parquet")
    open(os.path.join(HERE, "anatomy.md"), "w").write("\n".join(out))
    print("\n".join(out))
