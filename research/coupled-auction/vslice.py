"""
Vertical slice of the generative model (two zones, one delivery day of 96 quarter hours): draw parameters and day-level
shocks, build both sides of each zone's order book with explicit capacity accounting and inventory-safe storage bids,
clear the coupled day, and report awards, flows, prices and closing storage inventory.

Every MW of available capacity is assigned to exactly one of: netted against own load, offered in a period order,
offered in a block, or not offered. `accounting()` checks that identity per zone, technology and quarter.
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from clearing import Block, Line, Market, Order, clear

T, DT = 96, 0.25                         # quarter hours, hours per period
HOURS = np.arange(T) / 4.0
PEAK = (HOURS >= 7) & (HOURS < 20)       # window of the thermal blocks

# installed capacity (MW), heat rate (MWh fuel per MWh), emission factor (t CO2 per MWh), other cost (EUR/MWh), start cost (EUR per MW started)
TECH = {
    "A": {"other": (8000, 0.0, 0.0, 5.0, 0.0), "lignite": (18000, 0.0, 1.10, 8.0, 0.0), "coal": (12000, 2.5, 0.85, 10.0, 0.0),
          "gas": (33000, 1.9, 0.37, 4.0, 0.0), "wind": (70000,), "solar": (100000,)},
    "B": {"nuclear": (6000, 0.0, 0.0, 9.0, 0.0), "hydro": (3000, 0.0, 0.0, 0.0, 0.0), "gas": (5000, 1.9, 0.37, 4.0, 0.0), "wind": (5000,)},
}
COAL_PRICE = 12.0                          # EUR/MWh thermal, coal; lignite cost is in "other cost"
STORAGE = {"A": dict(P=2000.0, E=6000.0, soc0=3000.0, eta=0.85), "B": dict(P=500.0, E=1500.0, soc0=750.0, eta=0.85)}


@dataclass
class Info:
    """what is known at the issue time, before gate closure on the day before delivery"""
    gas: float = 40.0                      # EUR/MWh
    co2: float = 75.0                      # EUR/t
    atc: float = 1000.0                    # MW each way
    wind_cf: dict = field(default_factory=lambda: {"A": 0.25 + 0.15 * np.sin(np.pi * HOURS / 24), "B": 0.30 + 0.0 * HOURS})
    solar_cf: dict = field(default_factory=lambda: {"A": np.clip(np.sin(np.pi * (HOURS - 7) / 10), 0, None) * 0.45})
    load: dict = field(default_factory=lambda: {"A": 52000 + 12000 * np.exp(-((HOURS - 12) / 4.5) ** 2) + 5000 * np.exp(-((HOURS - 19) / 2) ** 2) - 8000 * np.exp(-((HOURS - 3.5) / 2.5) ** 2),
                                                "B": 7000 + 1500 * np.exp(-((HOURS - 18.5) / 2.5) ** 2)})
    outages: dict = field(default_factory=lambda: {"A": {"coal": 1000.0}, "B": {"nuclear": 0.0}})


PRIORS = {  # theta, shared across days
    "markup": ("normal", 3.0, 2.0),          # EUR/MWh over cost, thermal
    "spread": ("lognormal", np.log(0.08), 0.3),  # spread of unit costs, share of cost
    "pi": ("beta", 18.0, 2.0),               # share of available thermal capacity offered in the auction
    "netting": ("beta", 4.0, 8.0),           # share of load netted inside portfolios
    "flex_share": ("beta", 2.0, 40.0),       # price-responsive share of load
    "flex_wtp": ("normal", 60.0, 15.0),      # willingness to pay of flexible demand, EUR/MWh
    "block_share": ("beta", 3.0, 7.0),       # share of offered coal and gas capacity offered as peak blocks
    "start_cost": ("lognormal", np.log(60.0), 0.3),  # EUR per MW started, spread over the block
    "mustrun": ("beta", 6.0, 6.0),           # share of offered lignite at a negative price (must-run)
}


def draw_theta(rng):
    th = {}
    for k, (kind, a, b) in PRIORS.items():
        th[k] = rng.normal(a, b) if kind == "normal" else (rng.lognormal(a, b) if kind == "lognormal" else rng.beta(a, b))
    return th


def draw_day(rng, info: Info):
    """phi_d: day-level shocks with persistence across quarters"""
    def ar1(sd, rho=0.97):
        x = np.zeros(T); x[0] = rng.normal(0, sd)
        for t in range(1, T): x[t] = rho * x[t - 1] + rng.normal(0, sd * np.sqrt(1 - rho ** 2))
        return x
    return {"load_level": {z: rng.normal(0, 0.02) for z in info.load},                        # retailers' load expectation error
            "wind_err": {z: ar1(0.04) for z in info.wind_cf},                                  # bidders' wind view against the issue-time forecast
            "solar_err": {z: rng.normal(0, 0.05) for z in info.solar_cf},                      # cloudiness of the day
            "forced": {z: {k: rng.beta(2, 30) for k in TECH[z] if len(TECH[z][k]) > 1} for z in TECH},  # forced outage share
            "markup_day": rng.normal(0, 1.0)}


def cost(z, k, info):
    C, hr, ef, oc, sc = TECH[z][k]
    fuel = info.gas if k == "gas" else (COAL_PRICE if k == "coal" else 0.0)
    return hr * fuel + ef * info.co2 + oc


def build_book(theta, day, info: Info, storage=True, expected_prices=None):
    """
    Order book of both zones and the capacity accounting. expected_prices (zone -> T) is the storage owners' price
    forecast; if None, storage is left out (used to make that forecast from the rest of the book).
    """
    orders, blocks, acc = [], [], {}
    for z in TECH:
        load = info.load[z] * (1 + day["load_level"][z])
        netted = theta["netting"] * load                                      # load served inside portfolios
        # sell side: available MW per technology and quarter, then its split
        avail = {}
        for k, spec in TECH[z].items():
            if k == "wind": avail[k] = spec[0] * np.clip(info.wind_cf[z] + day["wind_err"][z], 0, 1)
            elif k == "solar": avail[k] = spec[0] * np.clip(info.solar_cf[z] * (1 + day["solar_err"][z]), 0, 1)
            else: avail[k] = np.full(T, (spec[0] - info.outages.get(z, {}).get(k, 0.0)) * (1 - day["forced"][z][k]))
        # netting is served first from the cheapest available output, in merit order
        order_k = sorted(avail, key=lambda k: -1e9 if k in ("wind", "solar") else cost(z, k, info))
        remaining = netted.copy(); split = {}
        for k in order_k:
            n_k = np.minimum(avail[k], remaining); remaining = remaining - n_k
            rest = avail[k] - n_k
            if k in ("wind", "solar"):
                split[k] = dict(netted=n_k, period=rest, block=np.zeros(T), unoffered=np.zeros(T))
            else:
                offered = theta["pi"] * rest
                blk = np.zeros(T)
                if k in ("coal", "gas"):
                    unit = theta["block_share"] * offered.min()                     # one block profile, the same MW in every peak quarter
                    blk = np.where(PEAK, unit, 0.0)
                split[k] = dict(netted=n_k, period=offered - blk, block=blk, unoffered=rest - offered)
        acc[z] = dict(avail=avail, split=split, load=load, netted=netted, short=remaining)
        # period sell orders
        for k, s in split.items():
            if k in ("wind", "solar"):
                for t in range(T):
                    if s["period"][t] > 1e-6: orders.append(Order(z, t, "sell", float(s["period"][t]), -5.0 if k == "wind" else -1.0))
                continue
            c = cost(z, k, info) + theta["markup"] + day["markup_day"]
            steps = [(-0.1, 0.3), (0.0, 0.4), (0.15, 0.3)] if k != "hydro" else [(0.0, 0.5), (0.6, 0.5)]
            for t in range(T):
                q = s["period"][t]
                if k == "lignite":    # must-run share at a negative price, the rest around cost
                    if q * theta["mustrun"] > 1e-6: orders.append(Order(z, t, "sell", float(q * theta["mustrun"]), -20.0))
                    q = q * (1 - theta["mustrun"])
                if k == "hydro":      # water value: hydro offers near the gas cost of the zone
                    c = 0.8 * cost(z, "gas", info) if "gas" in TECH[z] else 50.0
                for dz, share in steps:
                    if q * share > 1e-6: orders.append(Order(z, t, "sell", float(q * share), float(c * (1 + dz * theta["spread"] / 0.08 * 0.5))))
            if s["block"].max() > 1e-6:   # peak block: running cost plus start cost spread over the block's energy
                hours = PEAK.sum() * DT
                blocks.append(Block(f"{z} {k} peak block", z, "sell", float(c + theta["start_cost"] / hours), [float(v) for v in s["block"]], 1.0))
        # buy side: price-taking load less netting, flexible demand
        for t in range(T):
            orders.append(Order(z, t, "buy", float(load[t] - netted[t] - theta["flex_share"] * load[t]), 4000.0))
            for dz, share in ((-0.3, 0.3), (0.0, 0.4), (0.3, 0.3)):
                orders.append(Order(z, t, "buy", float(theta["flex_share"] * load[t] * share), float(theta["flex_wtp"] * (1 + dz))))
        # storage: inventory-safe bids from the owners' price forecast
        if storage and expected_prices is not None:
            st = STORAGE[z]; ep = expected_prices[z]
            n_dis = int(st["soc0"] / (st["P"] * DT))                     # discharge offered never exceeds the opening inventory
            n_chg = int((st["E"] - st["soc0"]) / (st["eta"] * st["P"] * DT))  # charging offered never exceeds the free space
            dis = np.argsort(-ep)[:n_dis]; chg = np.argsort(ep)[:n_chg]
            chg = np.setdiff1d(chg, dis)
            p_lo, p_hi = float(np.quantile(ep, 0.30)), float(np.quantile(ep, 0.70))
            for t in dis: orders.append(Order(z, int(t), "sell", st["P"], p_hi))
            for t in chg: orders.append(Order(z, int(t), "buy", st["P"], p_lo))
            acc[z]["storage"] = dict(dis=dis, chg=chg, p_lo=p_lo, p_hi=p_hi)
    m = Market(["A", "B"], T, orders, blocks, [Line("A", "B", [info.atc] * T, [info.atc] * T)], duration=DT)
    return m, acc


def accounting(acc):
    """max absolute residual of available = netted + period + block + unoffered, over zones, technologies and quarters"""
    worst = 0.0
    for z, a in acc.items():
        for k, s in a["split"].items():
            worst = max(worst, float(np.abs(a["avail"][k] - (s["netted"] + s["period"] + s["block"] + s["unoffered"])).max()))
            worst = max(worst, float((-np.minimum(0, s["period"])).max()))      # no negative period offers (a block cannot reuse period MW)
    return worst


def storage_path(m, r, acc, z):
    """inventory through the day from the awards; checks it stays within [0, E]"""
    st = STORAGE[z]; soc = np.full(T + 1, st["soc0"])
    sell = np.zeros(T); buy = np.zeros(T)
    s = acc[z].get("storage")
    for j, o in enumerate(m.orders):
        if o.zone != z: continue
        if s is not None and o.side == "sell" and o.t in s["dis"] and abs(o.p - s["p_hi"]) < 1e-9 and abs(o.q - st["P"]) < 1e-9: sell[o.t] += r.x[j] * o.q
        if s is not None and o.side == "buy" and o.t in s["chg"] and abs(o.p - s["p_lo"]) < 1e-9 and abs(o.q - st["P"]) < 1e-9: buy[o.t] += r.x[j] * o.q
    for t in range(T): soc[t + 1] = soc[t] + DT * (st["eta"] * buy[t] - sell[t])
    return soc, sell, buy


def simulate(theta, day, info, intervals=False):
    """the whole draw: book without storage -> storage owners' price forecast -> full book -> clearing"""
    m0, _ = build_book(theta, day, info, storage=False)
    r0 = clear(m0, intervals=False)
    ep = {z: r0.prices[i] for i, z in enumerate(m0.zones)}
    m, acc = build_book(theta, day, info, storage=True, expected_prices=ep)
    return m, acc, clear(m, intervals=intervals)
