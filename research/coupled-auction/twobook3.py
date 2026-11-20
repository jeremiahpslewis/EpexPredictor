"""Two books at the truth on the three common days of sbi_cf.py (notes 21.161): ATC-500 change in B's mean, book 1 vs book 2."""
import copy, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import vslice as V
from clearing import Line, Order, clear
from sbi_sim2 import TRUTH
from sbi_cf import DAYSEEDS

def hidden_flex(mm, rr, dq=1500.0, dp=3.0):   # as in demo.py
    m_ = copy.deepcopy(mm)
    for i, z in enumerate(mm.zones):
        for t in range(V.T):
            if rr.prices[i, t] < 3990: m_.orders += [Order(z, t, "buy", dq, float(rr.prices[i, t] - dp)), Order(z, t, "sell", dq, float(rr.prices[i, t] + dp))]
    return m_

info = V.Info(); out = []
for s in DAYSEEDS:
    day = V.draw_day(np.random.default_rng(s), info, TRUTH); m, acc, r = V.simulate(TRUTH, day, info)
    half = lambda mm: (lambda m1: (setattr(m1, "lines", [Line("A", "B", [500.0] * V.T, [500.0] * V.T)]), m1)[1])(copy.deepcopy(mm))
    mB = hidden_flex(m, r); rB = clear(mB, intervals=False)
    d1 = clear(half(m), intervals=False).prices[1].mean() - r.prices[1].mean(); d2 = clear(half(mB), intervals=False).prices[1].mean() - rB.prices[1].mean()
    out.append(dict(seed=s, same=float(np.abs(rB.prices - r.prices).max()), book1=round(float(d1), 2), book2=round(float(d2), 2),
                    mean_A=round(float(r.prices[0].mean()), 1), mean_B=round(float(r.prices[1].mean()), 1), at_limit=int((np.abs(np.abs(r.f[0]) - 1000) < 1e-6).sum())))
res = {"days": out, "book1_mean": round(float(np.mean([o["book1"] for o in out])), 2), "book2_mean": round(float(np.mean([o["book2"] for o in out])), 2)}
json.dump(res, open(Path(sys.argv[1]) / "twobook3.json", "w"), indent=1); print(json.dumps(res, indent=1))
