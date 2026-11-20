"""
Volume indeterminacy on the vertical slice (notes 21.155): the 594 MW case, the admissible flow range per quarter, and
whether the choice of rule matters. Writes vol_rule.json.
"""
import copy, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import vslice as V
from clearing import audit, clear

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
rng = np.random.default_rng(7); theta = V.draw_theta(rng); info = V.Info(); day = V.draw_day(rng, info)
TOL = 1e-6

def at_money(m, r, t):
    """accepted MW of at-the-money orders by zone and side in quarter t, and each zone's net position"""
    out = {}
    for i, z in enumerate(m.zones):
        p = r.prices[i, t]; rows = []
        for j, o in enumerate(m.orders):
            if o.zone == z and o.t == t and abs(o.p - p) < 1e-6:
                rows.append({"side": o.side, "price": o.p, "offered": round(o.q, 1), "accepted": round(float(r.x[j] * o.q), 1)})
        sell = sum(r.x[j] * o.q for j, o in enumerate(m.orders) if o.zone == z and o.t == t and o.side == "sell") + \
               sum(r.a[k] * b.q[t] for k, b in enumerate(m.blocks) if b.zone == z)
        buy = sum(r.x[j] * o.q for j, o in enumerate(m.orders) if o.zone == z and o.t == t and o.side == "buy")
        out[z] = {"price": round(float(p), 3), "at_money": rows, "net_position": round(float(sell - buy), 1)}
    out["flow_AB"] = round(float(r.f[0, t]), 1)
    return out

res = {}
# 1. the 594 MW case: netting 0.05 against 0.5, the surplus problem's own vertex (no rule), as before the fix
books = {}
for net in (0.05, 0.5):
    th = dict(theta, netting=net)
    m, acc, r = V.simulate(th, day, info, volume_rule=None)
    books[net] = (m, r)
dflow = np.abs(books[0.05][1].f[0] - books[0.5][1].f[0]); t = int(np.argmax(dflow))
res["case"] = {"quarter": t, "hour": t / 4, "max_flow_change": float(dflow.max()),
               "max_price_change": float(np.abs(books[0.05][1].prices - books[0.5][1].prices).max()),
               "netting_0.05": at_money(*books[0.05], t), "netting_0.5": at_money(*books[0.5], t)}
for net in (0.05, 0.5):
    m, r = books[net]
    m2 = copy.deepcopy(m); r2 = clear(m2, intervals=False, volume_rule="least_flow")
    res["case"][f"netting_{net}_least_flow"] = at_money(m2, r2, t)

# 2. on the base book: every rule gives the same prices and a consistent result; how wide is the admissible flow range?
m, acc, r = V.simulate(theta, day, info)
rules = {}
for rule in (None, "least_flow", "max_ab", "max_ba"):
    rr = clear(copy.deepcopy(m), intervals=False, volume_rule=rule)
    rules[str(rule)] = rr
    res.setdefault("audit", {})[str(rule)] = audit(m, rr)
    res.setdefault("prices_equal_least_flow", {})[str(rule)] = float(np.abs(rr.prices - rules.get("least_flow", rr).prices).max())
width = rules["max_ab"].f[0] - rules["max_ba"].f[0]
binding = np.abs(np.abs(rules["least_flow"].f[0]) - info.atc) < 1e-6
res["base_day"] = {"quarters_with_room": int((width > 1.0).sum()), "quarters_line_binding": int(binding.sum()),
                   "max_width": float(width.max()), "mean_width_where_room": float(width[width > 1].mean()) if (width > 1).any() else 0.0,
                   "equal_price_quarters": int((np.abs(r.prices[0] - r.prices[1]) < 1e-6).sum())}

# 3. the same over 200 drawn parameter sets and days (the training distribution): share of quarters with an admissible range
rng2 = np.random.default_rng(424242); widths, equal, bind = [], 0, 0
for i in range(200):
    th = V.draw_theta(rng2); d = V.draw_day(rng2, info)
    try:
        mm, _, rl = V.simulate(th, d, info)
    except Exception:
        continue
    a_ = clear(copy.deepcopy(mm), intervals=False, volume_rule="max_ab").f[0]; b_ = clear(copy.deepcopy(mm), intervals=False, volume_rule="max_ba").f[0]
    widths.append(a_ - b_); equal += int((np.abs(rl.prices[0] - rl.prices[1]) < 1e-6).sum()); bind += int((np.abs(np.abs(rl.f[0]) - info.atc) < 1e-6).sum())
W = np.array(widths)
res["drawn_days"] = {"days": len(W), "share_quarters_with_room": float((W > 1).mean()), "share_equal_price": equal / W.size,
                     "share_line_binding": bind / W.size, "width_quantiles_where_room": np.quantile(W[W > 1], [0.5, 0.9, 1.0]).tolist() if (W > 1).any() else []}
json.dump(res, open(OUT / "vol_rule.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in res.items() if k != "case"}, indent=1, default=float))
c = res["case"]; print("case quarter", c["quarter"], "hour", c["hour"], "flow change", c["max_flow_change"], "price change", c["max_price_change"])
for k in ("netting_0.05", "netting_0.5", "netting_0.05_least_flow", "netting_0.5_least_flow"): print(k, json.dumps(c[k]))
