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
    tech: dict | None = None               # technology table per zone; None = TECH
    coal_price: float | None = None        # EUR/MWh thermal; None = COAL_PRICE
    # zone -> MW per quarter, exogenous net position (+ export) entered as a price-taking order: buy at the cap for an
    # export, sell at the floor for an import (notes 21.162: a conditional check with the observed net position)
    fixed_netpos: dict | None = None
    storage: dict | None = None            # storage per zone; None = STORAGE
    # flexible demand's willingness to pay scaled with the gas-fired cost (opt-in, notes 21.162): flex_wtp is read as the
    # value at the reference costs (gas 40, carbon 75) and multiplied by today's gas-fired cost over the reference one
    wtp_scales_with_gas: bool = False


def techs(info):
    return info.tech if info is not None and info.tech is not None else TECH


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


# opt-in extension (notes 21.156): the mean forced-outage share of thermal plant becomes a parameter instead of fixed at
# Beta(2, 30) (mean 0.0625). Kept out of PRIORS so that draws without it stay identical to the earlier runs.
PRIORS_EXT = {"forced_mean": ("beta", 4.0, 56.0)}
FORCED_CONC = 32.0                       # concentration of the day's forced-outage share around its mean


def draw_theta(rng, extended=False):
    th = {}
    for k, (kind, a, b) in {**PRIORS, **(PRIORS_EXT if extended else {})}.items():
        th[k] = rng.normal(a, b) if kind == "normal" else (rng.lognormal(a, b) if kind == "lognormal" else rng.beta(a, b))
    return th


def draw_day(rng, info: Info, theta=None):
    """phi_d: day-level shocks with persistence across quarters"""
    def ar1(sd, rho=0.97):
        x = np.zeros(T); x[0] = rng.normal(0, sd)
        for t in range(1, T): x[t] = rho * x[t - 1] + rng.normal(0, sd * np.sqrt(1 - rho ** 2))
        return x
    fm = (theta or {}).get("forced_mean")
    fb = (2.0, 30.0) if fm is None else (fm * FORCED_CONC, (1 - fm) * FORCED_CONC)
    return {"load_level": {z: rng.normal(0, 0.02) for z in info.load},                        # retailers' load expectation error
            "wind_err": {z: ar1(0.04) for z in info.wind_cf},                                  # bidders' wind view against the issue-time forecast
            "solar_err": {z: rng.normal(0, 0.05) for z in info.solar_cf},                      # cloudiness of the day
            "forced": {z: {k: rng.beta(*fb) for k in techs(info)[z] if len(techs(info)[z][k]) > 1} for z in techs(info)},  # forced outage share
            "markup_day": rng.normal(0, 1.0)}


def cost(z, k, info):
    C, hr, ef, oc, sc = techs(info)[z][k]
    fuel = info.gas if k == "gas" else ((COAL_PRICE if info.coal_price is None else info.coal_price) if k == "coal" else 0.0)
    return hr * fuel + ef * info.co2 + oc


def build_book(theta, day, info: Info, storage=True, expected_prices=None):
    """
    Order book of both zones and the capacity accounting. expected_prices (zone -> T) is the storage owners' price
    forecast; if None, storage is left out (used to make that forecast from the rest of the book).
    """
    orders, blocks, acc = [], [], {}
    TECH_ = techs(info); STORAGE_ = info.storage if info.storage is not None else STORAGE
    for z in TECH_:
        load = info.load[z] * (1 + day["load_level"][z])
        netted = theta["netting"] * load                                      # load served inside portfolios
        avail, split, steps = {}, {}, []                                      # steps: [tech, t, q, p] period offers before netting
        for k, spec in TECH_[z].items():
            if k == "wind": avail[k] = spec[0] * np.clip(info.wind_cf[z] + day["wind_err"][z], 0, 1)
            elif k == "solar": avail[k] = spec[0] * np.clip(info.solar_cf[z] * (1 + day["solar_err"][z]), 0, 1)
            else: avail[k] = np.zeros(T) + np.maximum(spec[0] - np.asarray(info.outages.get(z, {}).get(k, 0.0)), 0.0) * (1 - day["forced"][z][k])
            # withholding and blocks apply to AVAILABLE capacity, before any netting (so netting cannot change them)
            if k in ("wind", "solar"):
                offered, blk = avail[k], np.zeros(T)
            else:
                offered = theta["pi"] * avail[k]
                blk = np.where(PEAK, theta["block_share"] * offered.min(), 0.0) if k in ("coal", "gas") else np.zeros(T)
            split[k] = dict(netted=np.zeros(T), period=offered - blk, block=blk, unoffered=avail[k] - offered)
            # the period offers of this technology as price steps
            if k in ("wind", "solar"):
                for t in range(T): steps.append([k, t, float(offered[t]), -5.0 if k == "wind" else -1.0])
                continue
            c = cost(z, k, info) + theta["markup"] + day["markup_day"]
            if k == "hydro": c = 0.8 * cost(z, "gas", info) if "gas" in TECH_[z] else 50.0
            ladder = [(0.0, 0.5), (0.6, 0.5)] if k == "hydro" else [(-0.1, 0.3), (0.0, 0.4), (0.15, 0.3)]
            for t in range(T):
                q = offered[t] - blk[t]
                if k == "lignite":    # must-run share of offered lignite at a negative price
                    steps.append([k, t, float(q * theta["mustrun"]), -20.0]); q = q * (1 - theta["mustrun"])
                for dz, share in ladder: steps.append([k, t, float(q * share), float(c * (1 + dz * theta["spread"] / 0.08 * 0.5))])
            if blk.max() > 1e-6:      # peak block: running cost plus start cost spread over the block's energy
                hours = PEAK.sum() * DT
                blocks.append(Block(f"{z} {k} peak block", z, "sell", float(c + theta["start_cost"] / hours), [float(v) for v in blk], 1.0))
        # netting takes the cheapest period offers in each quarter (the deepest in the money), never block capacity
        short = np.zeros(T)
        for t in range(T):
            need = netted[t]
            for st in sorted([s_ for s_ in steps if s_[1] == t], key=lambda s_: s_[3]):
                take = min(st[2], need); st[2] -= take; need -= take; split[st[0]]["netted"][t] += take; split[st[0]]["period"][t] -= take
                if need <= 1e-9: break
            short[t] = need
        for k_, t, q, p_ in steps:
            if q > 1e-6: orders.append(Order(z, t, "sell", q, p_))
        acc[z] = dict(avail=avail, split=split, load=load, netted=netted, short=short)
        # buy side: price-taking load less netting, flexible demand
        for t in range(T):
            orders.append(Order(z, t, "buy", float(load[t] - netted[t] - theta["flex_share"] * load[t]), 4000.0))
            wtp = theta["flex_wtp"] * ((1.9 * info.gas + 0.37 * info.co2 + 4.0) / (1.9 * 40.0 + 0.37 * 75.0 + 4.0) if info.wtp_scales_with_gas else 1.0)
            for dz, share in ((-0.3, 0.3), (0.0, 0.4), (0.3, 0.3)):
                orders.append(Order(z, t, "buy", float(theta["flex_share"] * load[t] * share), float(wtp * (1 + dz))))
        # exogenous net position (conditional checks only): an export is extra price-taking demand, an import extra
        # price-taking supply at the floor; neither scales with load, so netting and flexible demand do not touch it
        if info.fixed_netpos is not None and z in info.fixed_netpos:
            for t, v in enumerate(info.fixed_netpos[z]):
                if v > 1e-6: orders.append(Order(z, t, "buy", float(v), 4000.0))
                elif v < -1e-6: orders.append(Order(z, t, "sell", float(-v), -500.0))
        # storage: inventory-safe bids from the owners' price forecast
        if storage and expected_prices is not None:
            st = STORAGE_[z]; ep = expected_prices[z]
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


def simulate(theta, day, info, intervals=False, volume_rule="least_flow"):
    """the whole draw: book without storage -> storage owners' price forecast -> full book -> clearing"""
    m0, _ = build_book(theta, day, info, storage=False)
    r0 = clear(m0, intervals=False, volume_rule=volume_rule)
    ep = {z: r0.prices[i] for i, z in enumerate(m0.zones)}
    m, acc = build_book(theta, day, info, storage=True, expected_prices=ep)
    return m, acc, clear(m, intervals=intervals, volume_rule=volume_rule)
