"""Agent P: what a tree variant does to the equal-thirds blend and to the error correlations with Timestral and the market
route. Timestral and the market route are the stored faithful members (same RL variant as given).
Usage: blendcheck.py SCREEN_TAG VARIANT REF_VARIANT [out_strict|out_strict_rlmatched ...]"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "distill"))
from common import boot  # noqa: E402
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"


def main(tag, var, ref, outs):
    V = np.load(f"{SP}/P/screen/{tag}/{var}.npz"); R = np.load(f"{SP}/P/screen/{tag}/{ref}.npz")
    rows, errs = [], {k: [] for k in ("var", "ref", "champ_trees", "timestral", "market")}
    B = {k: [] for k in ("champion blend", "thirds: ref trees", "thirds: var trees", "thirds: avg(ref,var) trees")}
    act_all = []
    for d in sorted(V.files):
        z = None
        for o in outs:
            fn = f"{SP}/faithful/{o}/{d}.npz"
            if os.path.exists(fn):
                z = np.load(fn); break
        if z is None:
            continue
        a = z["actual"][:96]; v = V[d][:96, 1]; r = R[d][:96, 1]
        ts, mk, tr = z["timestral"][:96], z["market"][:96], z["trees"][:96]
        for k, f in (("var", v), ("ref", r), ("champ_trees", tr), ("timestral", ts), ("market", mk)):
            errs[k].append(f - a)
        B["champion blend"].append((tr + ts + mk) / 3)
        B["thirds: ref trees"].append((r + ts + mk) / 3)
        B["thirds: var trees"].append((v + ts + mk) / 3)
        B["thirds: avg(ref,var) trees"].append(((r + v) / 2 + ts + mk) / 3)
        act_all.append(a)
    E = pd.DataFrame({k: np.concatenate(v) for k, v in errs.items()})
    print(f"{tag} {var} vs {ref}; members from {outs}; {len(act_all)} days")
    print("QH error correlations:\n", E.corr().round(2).to_markdown())
    base = np.array([np.abs(b - a).mean() for b, a in zip(B["champion blend"], act_all)])
    for k, f in B.items():
        m = np.array([np.abs(b - a).mean() for b, a in zip(f, act_all)])
        lo, hi = boot(m - base)
        print(f"- {k}: {m.mean():.3f}  vs champion blend {(m - base).mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:] or ["out_strict", "out_fresh_strict"])
