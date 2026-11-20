"""Worked example for the specification: two zones, three periods, step orders, one block, one ATC line."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from clearing import Block, Line, Market, Order, audit, brute_force, clear, surplus_split

PERIODS = ["night", "morning peak", "evening"]


def market(cap):
    o = []
    # Zone A (Germany-Luxembourg-like): wind offered below zero, lignite, hard coal, gas; price-taking load and a flexible buyer
    for t, (wind, load) in enumerate([(1800, 1600), (700, 3000), (400, 2300)]):
        o += [Order("A", t, "sell", wind, -5), Order("A", t, "sell", 900, 30), Order("A", t, "sell", 500, 70), Order("A", t, "sell", 800, 110),
              Order("A", t, "buy", load, 4000), Order("A", t, "buy", 300, 20)]
    # Zone B (neighbour): nuclear, hydro, gas; price-taking load and a flexible buyer
    for t, load in enumerate([900, 1300, 1200]):
        o += [Order("B", t, "sell", 400, 0), Order("B", t, "sell", 400, 50), Order("B", t, "sell", 800, 95),
              Order("B", t, "buy", load, 4000), Order("B", t, "buy", 200, 40)]
    blocks = [Block("coal unit in A, all day", "A", "sell", 75.0, [400, 400, 400], 1.0)]
    return Market(["A", "B"], 3, o, blocks, [Line("A", "B", [cap] * 3, [cap] * 3)])


def report(cap):
    m = market(cap); r = clear(m); cs, ps, cr = surplus_split(m, r); bf = brute_force(m)
    print(f"--- ATC {cap} MW each way: master solves {r.rounds}, cuts {r.cuts}")
    for t, name in enumerate(PERIODS):
        print(f"  {name:13s} price A {r.prices[0, t]:7.2f} [{r.low[0, t]:.2f}, {r.high[0, t]:.2f}]  B {r.prices[1, t]:7.2f} [{r.low[1, t]:.2f}, {r.high[1, t]:.2f}] EUR/MWh | flow A->B {r.f[0, t]:7.1f} MW")
    print(f"  block accepted ratio {r.a[0]:.2f}; block surplus at these prices {sum(400 * (r.prices[0, t] - 75) for t in range(3)):.0f} EUR")
    print(f"  economic surplus {r.welfare:,.0f} = consumer {cs:,.0f} + producer {ps:,.0f} + congestion rent {cr:,.0f} (sum {cs + ps + cr:,.0f})")
    print(f"  brute force over block selections: surplus {bf[0]:,.0f}, selection {bf[1]} | audit violations: {audit(m, r) or 'none'}")
    for t in range(3):
        acc = [(o.zone, o.side, o.p, round(r.x[j] * o.q, 1)) for j, o in enumerate(m.orders) if o.t == t and r.x[j] > 1e-6 and o.p not in (4000,)]
        print(f"   {PERIODS[t]:13s} accepted (zone, side, limit, MW): {acc}")
    return m, r


if __name__ == "__main__":
    for cap in (400, 1500):
        report(cap)
