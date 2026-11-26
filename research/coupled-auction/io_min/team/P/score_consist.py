"""Score the consist confirmation (CONSIST_PREREG.md). Usage: score_consist.py"""
import os, sys, glob
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "distill"))
from common import boot  # noqa: E402
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"


def dayfile(d):
    for o in ("out_strict", "out_fresh_strict"):
        fn = f"{SP}/faithful/{o}/{d}.npz"
        if os.path.exists(fn):
            return np.load(fn)


rows = []
for part in ("A", "B"):
    for fn in sorted(glob.glob(f"{SP}/P/market_ts2/{part}/*.npz")):
        d = os.path.basename(fn)[:10]; r = np.load(fn); z = dayfile(d); a = z["actual"][:96]
        c, p = r["consist96"][:96], r["repro96"][:96]
        mae = lambda f: np.abs(f - a).mean()
        rows.append(dict(day=d, part=part, nlog=int(r["nlog_consist96"]), consist=mae(c), repro=mae(p), stored=mae(z["market"][:96]),
                         blend_c=mae((z["trees"][:96] + z["timestral"][:96] + c) / 3), blend_r=mae((z["trees"][:96] + z["timestral"][:96] + p) / 3),
                         blend_champ=mae(z["issued"][:96]),
                         e_c=c - a, e_tr=z["trees"][:96] - a, e_ts=z["timestral"][:96] - a, e_m=z["market"][:96] - a))
df = pd.DataFrame(rows)
df["diff"] = df.consist - df.repro


def line(sub, label, col="diff"):
    x = sub[col].to_numpy(); lo, hi = boot(x) if len(x) > 7 else (np.nan, np.nan)
    return f"| {label} | {len(x)} | {x.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] | {(x < 0).sum()}/{len(x)} |"


out = ["| set | days | consist96 − repro96 | days better |", "|---|---|---|---|"]
for lab, sub in [("A (61 window)", df[df.part == "A"]), ("B (12 fresh)", df[df.part == "B"]), ("pooled A+B", df)]:
    out.append(line(sub, lab))
best = df["diff"].idxmin()
out.append(line(df.drop(best), f"pooled without best day ({df.day[best]}, {df['diff'][best]:+.2f})"))
for m, sub in df.groupby(df.day.str[:7]):
    out.append(line(sub, f"month {m}"))
A = df[df.part == "A"]
for lab, sub in [("A log 0-19 d", A[A.nlog < 20]), ("A log 20-39 d", A[(A.nlog >= 20) & (A.nlog < 40)]), ("A log 40-60 d", A[A.nlog >= 40])]:
    out.append(line(sub, lab))
out += ["", "| set | repro96 | consist96 | stored market | repro96 − stored | consist96 − stored | blend: champion | with repro96 | with consist96 | consist-blend − champion |", "|---|---|---|---|---|---|---|---|---|---|"]
for lab, sub in [("A", df[df.part == "A"]), ("B", df[df.part == "B"]), ("pooled", df)]:
    rs = (sub.repro - sub.stored).to_numpy(); cs = (sub.consist - sub.stored).to_numpy(); bc = (sub.blend_c - sub.blend_champ).to_numpy()
    f = lambda x: f"{x.mean():+.3f} [{boot(x)[0]:+.3f}, {boot(x)[1]:+.3f}]"
    out.append(f"| {lab} | {sub.repro.mean():.3f} | {sub.consist.mean():.3f} | {sub.stored.mean():.3f} | {f(rs)} | {f(cs)} | {sub.blend_champ.mean():.3f} | {sub.blend_r.mean():.3f} | {sub.blend_c.mean():.3f} | {f(bc)} |")
E = pd.DataFrame({k: np.concatenate(df[k].to_numpy()) for k in ("e_c", "e_tr", "e_ts", "e_m")}).rename(columns={"e_c": "consist96", "e_tr": "trees", "e_ts": "timestral", "e_m": "market (champion)"})
out += ["", "QH error correlations (pooled):", "", E.corr().round(2).to_markdown()]
print("\n".join(out))
df.drop(columns=["e_c", "e_tr", "e_ts", "e_m"]).to_csv(f"{SP}/P/consist_days.csv", index=False)
