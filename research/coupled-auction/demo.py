"""One draw end to end, the two counterfactuals, and two order books that fit the same observed day (writes demo.json, demo.png)."""
import copy, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import vslice as V
from clearing import Line, Market, Order, clear

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
rng = np.random.default_rng(7); theta = V.draw_theta(rng); info = V.Info(); day = V.draw_day(rng, info)
m, acc, r = V.simulate(theta, day, info, intervals=True)
res = {"theta": theta, "day": {"load_level": day["load_level"], "markup_day": day["markup_day"]}}

def summ(m, r, label):
    soc = {z: V.storage_path(m, r, acc, z)[0] for z in m.zones}
    return {"label": label, "price_A": r.prices[0].tolist(), "price_B": r.prices[1].tolist(), "flow": r.f[0].tolist(),
            "mean_A": float(r.prices[0].mean()), "mean_B": float(r.prices[1].mean()), "at_limit": int((np.abs(np.abs(r.f[0]) - m.lines[0].cap_fwd[0]) < 1e-6).sum()),
            "blocks": [[b.name, b.p, float(r.a[k])] for k, b in enumerate(m.blocks)], "welfare": float(r.welfare),
            "closing_soc": {z: float(soc[z][-1]) for z in soc}, "soc_A": soc["A"].tolist(), "max_interval_width": float(np.max(r.high - r.low)) if r.high is not None else None}

base = summ(m, r, "base"); res["base"] = base
res["accounting_residual"] = V.accounting(acc)

# counterfactual 1: transmission 1000 -> 500 MW, the same submitted orders
m1 = copy.deepcopy(m); m1.lines = [Line("A", "B", [500.0] * V.T, [500.0] * V.T)]
res["cf_atc"] = summ(m1, clear(m1, intervals=False), "ATC 500 MW, same orders")

# counterfactual 2: gas 40 -> 52 EUR/MWh; regenerate every bid that depends on it (gas offers and blocks, hydro water value,
# storage owners' price forecast); theta and the day's shocks unchanged
info2 = copy.deepcopy(info); info2.gas = 52.0
m2, acc2, r2 = V.simulate(theta, day, info2)
res["cf_gas"] = summ(m2, r2, "gas 52 EUR/MWh, bids regenerated")
# variant: storage bids held at the base day's (owners do not update their forecast)
ep = {z: r.prices[i] for i, z in enumerate(m.zones)}
m2b, acc2b = V.build_book(theta, day, info2, storage=True, expected_prices=ep)
res["cf_gas_storage_fixed"] = summ(m2b, clear(m2b, intervals=False), "gas 52, storage bids as on the base day")

# two order books that reproduce the same day: book 2 adds price-responsive orders just outside each cleared price
# (buy 1500 MW at p* - 3, sell 1500 MW at p* + 3 in every zone and quarter), all rejected at the base outcome
def hidden_flex(mm, rr, dq=1500.0, dp=3.0):
    m_ = copy.deepcopy(mm)
    for i, z in enumerate(mm.zones):
        for t in range(V.T):
            p = rr.prices[i, t]
            if p < 3990:   # not at the price cap
                m_.orders += [Order(z, t, "buy", dq, float(p - dp)), Order(z, t, "sell", dq, float(p + dp))]
    return m_
mB = hidden_flex(m, r); rB = clear(mB, intervals=False)
res["book2_same_outcome"] = {"max_price_diff": float(np.abs(rB.prices - r.prices).max()), "max_flow_diff": float(np.abs(rB.f - r.f).max()),
                             "same_blocks": bool(np.allclose(rB.a, r.a))}
mB1 = copy.deepcopy(mB); mB1.lines = [Line("A", "B", [500.0] * V.T, [500.0] * V.T)]
res["book2_cf_atc"] = summ(mB1, clear(mB1, intervals=False), "book 2, ATC 500 MW")
json.dump(res, open(OUT / "demo.json", "w"))

print("theta", {k: round(v, 2) for k, v in theta.items()}, "| accounting residual", res["accounting_residual"])
for key in ("base", "cf_atc", "cf_gas", "cf_gas_storage_fixed", "book2_cf_atc"):
    s = res[key]; print(f"{s['label']:38s} mean A {s['mean_A']:6.1f}  B {s['mean_B']:6.1f} | quarters at the limit {s['at_limit']:2d} | blocks accepted {[round(b[2]) for b in s['blocks']]} | closing inventory A {s['closing_soc']['A']:.0f} B {s['closing_soc']['B']:.0f}")
print("book 2 reproduces the base day:", res["book2_same_outcome"])
print("base: widest interval of valid prices", base["max_interval_width"])
d1 = np.array(res["cf_atc"]["price_A"]) - np.array(base["price_A"]); d2 = np.array(res["book2_cf_atc"]["price_A"]) - np.array(base["price_A"])
print(f"ATC 500, change in A's price: book 1 mean {d1.mean():+.1f}, max {d1.max():+.1f}; book 2 mean {d2.mean():+.1f}, max {d2.max():+.1f} EUR/MWh")
dB1 = np.array(res["cf_atc"]["price_B"]) - np.array(base["price_B"]); dB2 = np.array(res["book2_cf_atc"]["price_B"]) - np.array(base["price_B"])
print(f"ATC 500, change in B's price: book 1 mean {dB1.mean():+.1f}; book 2 mean {dB2.mean():+.1f} EUR/MWh")
