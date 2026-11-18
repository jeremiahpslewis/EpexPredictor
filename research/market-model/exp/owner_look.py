"""By owner: (1) share of hours a block is on and its loading WHEN ON, by margin band, 2025 against 2026 (throttling of online blocks against blocks being off);
(2) reported unavailability (ENTSO-E, latest revisions) of the same plants in the same seasons; (3) total output at a given margin in GW."""
import os, json, numpy as np, pandas as pd
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
D = pd.read_parquet(f"{SP}/out/star_data.parquet").resample("1h").mean(); E = pd.read_parquet(f"{SP}/out/eua_daily.parquet")["eua"]; E.index = pd.to_datetime(E.index); loc = D.index.tz_convert("Europe/Berlin")
ed = E.reindex(pd.date_range(E.index.min(), E.index.max(), freq="D")).ffill().shift(1); eua = pd.Series(ed.reindex(loc.normalize().tz_localize(None)).values, index=D.index); p = D["p_de"]; gas = D["gas"]
W = pd.read_csv(f"{SP}/out/wb_coal.csv", index_col=0, parse_dates=True)["Coal, South African **"] / 6.978 * 0.92; mon = pd.DatetimeIndex(loc.tz_localize(None)).to_period("M").to_timestamp() - pd.DateOffset(months=1); coal = pd.Series(W.reindex(W.index.union(mon.unique())).sort_index().ffill().reindex(mon).values, index=D.index)
COST = {"Braunkohle": 12.0 + eua, "Steinkohle": 4.0 + 2.6 * coal + 0.85 * eua, "Erdgas": 3.0 + 1.85 * gas + 0.37 * eua}; md = loc.strftime("%m-%d")
sel = json.load(open(f"{SP}/ext/smard/blocks_sel.json")); GRP = {"Lausitz Energie Kraftwerke AG": "LEAG", "RWE Power AG": "RWE", "RWE Generation SE": "RWE", "EnBW Energie Baden-Württemberg AG": "EnBW", "Saale Energie GmbH": "Saale (EPH)"}
B = {}
for b in sel:
    fn = f"{SP}/ext/smard/blocks/{b['id']}.parquet"
    if not os.path.exists(fn) or "etrieb" not in b["status"]: continue
    s = pd.read_parquet(fn)["mw"].reindex(D.index)
    if (s > 0).sum() < 500: continue
    B[b["id"]] = (b, s, np.nanpercentile(s[s > 0], 99.5))
bands = [(-1e9, 0), (0, 30), (30, 60), (60, 120), (120, 1e9)]; bl = ["below cost", "0-30", "30-60", "60-120", "above 120"]
for S1, S2 in (("04-01", "07-31"), ("08-01", "11-08")):
    seas = (md >= S1) & (md <= S2); print(f"\n=== {S1} to {S2}. per owner group: output in GW at a given margin (all its blocks summed), share of capacity on, loading of the blocks that are on; 2025 -> 2026")
    for fuel in ("Braunkohle", "Steinkohle", "Erdgas"):
        m = (p - COST[fuel]).values; groups = {}
        for bid, (b, s, cap) in B.items():
            if b["fuel"] == fuel: groups.setdefault(GRP.get(b["company"], "other owners"), []).append((s.values, cap))
        for g, lst in sorted(groups.items()):
            tot = np.nansum([v for v, c in lst], 0); capsum = sum(c for v, c in lst); on = np.nansum([(v > 0.15 * c) * c for v, c in lst], 0); allna = np.all([np.isnan(v) for v, c in lst], 0)
            line = f"  {fuel:10s} {g:13s} {capsum / 1000:4.1f} GW |"
            for (a_, b_), lab in zip(bands, bl):
                out = []
                for y in (2025, 2026):
                    k = seas & (loc.year == y) & (m >= a_) & (m < b_) & ~allna
                    out.append((tot[k].mean() / 1000, on[k].mean() / capsum, tot[k].sum() / max(on[k].sum(), 1), int(k.sum())) if k.sum() >= 40 else None)
                line += f" {lab}: " + (" -> ".join(f"{o[0]:.1f} GW (on {o[1]:.2f}, load {o[2]:.2f})" if o else "." for o in out)) + " |"
            print(line[:330])
# reported unavailability by plant name
U = pd.read_parquet(f"{SP}/entsoe/unavail_gen_DE_LU.parquet").reset_index(); U = U[U["revision"] == U.groupby("mrid")["revision"].transform("max")]; U = U[U["docstatus"].astype(str) != "Cancelled"].copy(); U["un"] = (U["nominal_power"] - pd.to_numeric(U["avail_qty"], errors="coerce")).clip(lower=0)
grid = pd.date_range("2025-01-01", "2026-11-09", freq="1h", tz="UTC"); gl = grid.tz_convert("Europe/Berlin"); gmd = gl.strftime("%m-%d")
print("\nreported unavailable capacity (ENTSO-E, GW, mean over the season) by plant group; planned share in brackets")
for lab, pat, ptype in (("RWE lignite (Neurath, Niederaussem, Weisweiler)", "Neurath|Niedera|Weisweiler", "Fossil Brown coal/Lignite"), ("LEAG lignite (Boxberg, Jänschwalde, Schwarze Pumpe, Lippendorf)", "Boxberg|nschwalde|Schwarze Pumpe|Lippendorf", "Fossil Brown coal/Lignite"), ("Schkopau", "Schkopau", "Fossil Brown coal/Lignite"), ("RWE gas Emsland / Lingen", "Emsland|Lingen", "Fossil Gas"), ("all hard coal", ".", "Fossil Hard coal")):
    x = U[(U["plant_type"] == ptype) & (U["production_resource_name"].astype(str).str.contains(pat, regex=True) | U["production_resource_psr_name"].astype(str).str.contains(pat, regex=True))]; tot = np.zeros(len(grid)); pl = np.zeros(len(grid)); nom = x.groupby("production_resource_psr_name")["nominal_power"].max().sum() / 1000
    for unit, g_ in x.groupby("production_resource_psr_name"):
        u_ = np.zeros(len(grid)); up = np.zeros(len(grid))
        for s_, e_, un, bt in zip(g_["start"], g_["end"], g_["un"], g_["businesstype"]):
            i, j = grid.searchsorted(s_.tz_convert("UTC")), grid.searchsorted(e_.tz_convert("UTC"))
            if j > i and np.isfinite(un):
                u_[i:j] = np.maximum(u_[i:j], un)
                if bt == "Planned maintenance": up[i:j] = np.maximum(up[i:j], un)
        tot += u_; pl += up
    print(f"  {lab:62s} units' nominal {nom:4.1f} GW | " + " | ".join(f"{S1}..{S2}: " + " -> ".join(f"{tot[(gmd >= S1) & (gmd <= S2) & (gl.year == y)].mean() / 1000:.2f} ({pl[(gmd >= S1) & (gmd <= S2) & (gl.year == y)].mean() / max(tot[(gmd >= S1) & (gmd <= S2) & (gl.year == y)].mean(), 1e-9):.0%})" for y in (2025, 2026)) for S1, S2 in (("04-01", "07-31"), ("08-01", "11-08"))))
