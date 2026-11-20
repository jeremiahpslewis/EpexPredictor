"""Verification of the small clearing solver (see the specification document, section 'How the solver was verified')."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from clearing import Block, Line, Market, Order, audit, brute_force, clear, surplus_split

failures = []
def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok: failures.append(name)

# 1. Hand-solvable: one zone, one period. Supply 100 MW at 10, 100 MW at 30; demand 150 MW at 4000 -> price 30, 50 MW of the second step.
m = Market(["A"], 1, [Order("A", 0, "sell", 100, 10), Order("A", 0, "sell", 100, 30), Order("A", 0, "buy", 150, 4000)])
r = clear(m)
check("1 hand case: price 30, determinate", abs(r.prices[0, 0] - 30) < 1e-6 and abs(r.low[0, 0] - 30) < 1e-6 and abs(r.high[0, 0] - 30) < 1e-6, f"price {r.prices[0, 0]:.4f} in [{r.low[0, 0]:.2f}, {r.high[0, 0]:.2f}]")
check("1 hand case: 50 MW of the 30 step", abs(r.x[1] * 100 - 50) < 1e-6)

# 2. Hand-solvable two zones, congested line: A supply 200 at 10, demand 100; B supply 200 at 50, demand 150; ATC 30 A->B.
#    A exports 30: A price 10 (step at 10 partly used), B price 50 (step at 50 partly used), congestion rent 30 * 40 = 1200.
m = Market(["A", "B"], 1, [Order("A", 0, "sell", 200, 10), Order("A", 0, "buy", 100, 4000), Order("B", 0, "sell", 200, 50), Order("B", 0, "buy", 150, 4000)],
           [], [Line("A", "B", [30], [30])])
r = clear(m); cs, ps, cr = surplus_split(m, r)
check("2 congested pair: prices 10 / 50, flow 30", np.allclose(r.prices[:, 0], [10, 50]) and abs(r.f[0, 0] - 30) < 1e-6, f"prices {r.prices[:, 0]}, flow {r.f[0, 0]:.2f}")
check("2 congestion rent 1200", abs(cr - 1200) < 1e-6, f"{cr:.2f}")

# 3. Paradoxically accepted block must be cut. One zone, two periods. Period 0: demand 100 MW, supply 50 MW at 30 and
#    50 MW at 100. Period 1: demand 100 MW, supply 100 MW at 0. Block: sell 100 MW in both periods at 20.
#    With the block, surplus is higher (cost 4000 against 6500 without it), but then the 30 step in period 0 and the 0
#    step in period 1 are rejected, so prices are at most 30 and 0: the block's average price (at most 15) is below its
#    limit of 20 - paradoxically accepted. The selection must be cut; without the block, prices are 100 and 0.
m = Market(["A"], 2, [Order("A", 0, "sell", 50, 30), Order("A", 0, "sell", 50, 100), Order("A", 0, "buy", 100, 4000),
                      Order("A", 1, "sell", 100, 0), Order("A", 1, "buy", 100, 4000)],
           [Block("b", "A", "sell", 20.0, [100, 100], 1.0)])
r = clear(m); bf = brute_force(m)
check("3 a paradoxically accepted block is cut", len(r.cuts) == 1 and r.a[0] == 0, f"cuts {r.cuts}, block ratio {r.a[0]}")
# without the block every order is fully accepted, so the price is indeterminate: [100, 4000] in period 0, [0, 4000] in period 1
check("3 valid price intervals without the block: [100, 4000] and [0, 4000]", np.allclose(r.low[0], [100, 0]) and np.allclose(r.high[0], [4000, 4000]), f"low {r.low[0]}, high {r.high[0]}")
check("3 result equals brute force", abs(r.welfare - bf[0]) < 1e-6, f"{r.welfare:.1f} vs {bf[0]:.1f}")

# 4. Random markets: algorithm against brute force over all block selections, and the consistency audit.
rng = np.random.default_rng(1); n_ok = n_audit = n_identity = n_relax = 0; N = 300; n_cut = 0
for i in range(N):
    zones, T = ["A", "B", "C"][:rng.integers(1, 4)], int(rng.integers(1, 4)); orders = []
    for z in zones:
        for t in range(T):
            orders.append(Order(z, t, "buy", float(rng.integers(50, 300)), 4000.0))
            for _ in range(rng.integers(1, 4)): orders.append(Order(z, t, "buy", float(rng.integers(10, 150)), float(rng.integers(-20, 120))))
            for _ in range(rng.integers(2, 5)): orders.append(Order(z, t, "sell", float(rng.integers(20, 250)), float(rng.integers(-30, 150))))
    blocks = [Block(f"b{k}", zones[rng.integers(len(zones))], "sell" if rng.random() < 0.7 else "buy", float(rng.integers(0, 120)),
                    [float(rng.integers(0, 150)) for _ in range(T)], 1.0) for k in range(rng.integers(1, 4))]
    lines = [Line(zones[k], zones[k + 1], [float(rng.integers(0, 200)) for _ in range(T)], [float(rng.integers(0, 200)) for _ in range(T)]) for k in range(len(zones) - 1)]
    m = Market(zones, T, orders, blocks, lines)
    try:
        r = clear(m)
    except RuntimeError:
        continue
    bf = brute_force(m); n_cut += len(r.cuts) > 0
    n_ok += bf is not None and abs(r.welfare - bf[0]) <= 1e-6 * max(1, abs(bf[0]))
    n_audit += not audit(m, r) and bool(np.all(r.low <= r.prices + 1e-6)) and bool(np.all(r.prices <= r.high + 1e-6))
    cs, ps, cr = surplus_split(m, r); n_identity += abs(cs + ps + cr - r.welfare) <= 1e-6 * max(1, abs(r.welfare))
    if lines:   # relaxing every line cannot lower the surplus
        m2 = Market(zones, T, orders, blocks, [Line(l.frm, l.to, [c + 500 for c in l.cap_fwd], [c + 500 for c in l.cap_bwd]) for l in lines])
        n_relax += clear(m2).welfare >= r.welfare - 1e-6 * max(1, abs(r.welfare))
    else:
        n_relax += 1
check(f"4 random markets: equal to brute force ({N} markets, {n_cut} needed a cut)", n_ok == N, f"{n_ok} of {N}")
check("4 random markets: consistency audit clean", n_audit == N, f"{n_audit} of {N}")
check("4 random markets: surplus = consumer + producer + congestion rent", n_identity == N, f"{n_identity} of {N}")
check("4 random markets: relaxing lines never lowers surplus", n_relax == N, f"{n_relax} of {N}")

# 5. Period duration: the same market in quarter hours has a quarter of the surplus and the same prices.
m1 = Market(["A"], 1, [Order("A", 0, "sell", 100, 10), Order("A", 0, "sell", 100, 30), Order("A", 0, "buy", 150, 4000)])
m4 = Market(["A"], 1, [Order("A", 0, "sell", 100, 10), Order("A", 0, "sell", 100, 30), Order("A", 0, "buy", 150, 4000)], duration=0.25)
r1, r4 = clear(m1), clear(m4)
check("5 quarter-hour periods: same price, a quarter of the surplus", abs(r1.prices[0, 0] - r4.prices[0, 0]) < 1e-9 and abs(r4.welfare - r1.welfare / 4) < 1e-6, f"{r1.welfare:.1f} vs {r4.welfare:.1f}")

print("\nall checks passed" if not failures else f"\n{len(failures)} FAILED: {failures}")
sys.exit(1 if failures else 0)
