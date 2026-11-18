"""SMARD production per plant block (hourly) for the selected lignite, hard coal and large gas blocks, April to mid-November of 2025 and 2026. One parquet per block; resumable."""
import sys, os, json, time, urllib.request
import pandas as pd
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; OUT = f"{SP}/ext/smard/blocks"; os.makedirs(OUT, exist_ok=True); B = "https://www.smard.de/app/chart_data"
WIN = [(pd.Timestamp("2025-03-24", tz="UTC"), pd.Timestamp("2025-11-10", tz="UTC")), (pd.Timestamp("2026-03-23", tz="UTC"), pd.Timestamp("2026-11-10", tz="UTC"))]
def get(url):
    for a in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "research-script"}), timeout=60) as r: return json.loads(r.read().decode())
        except Exception as e:
            if "404" in str(e): return None
            time.sleep(3 + 5 * a)
    return None
sel = json.load(open(f"{SP}/ext/smard/blocks_sel.json")); shard, nsh = int(sys.argv[1]), int(sys.argv[2])
for i, b in enumerate(sel):
    if i % nsh != shard: continue
    fn = f"{OUT}/{b['id']}.parquet"
    if os.path.exists(fn): continue
    idx = get(f"{B}/{b['id']}/{b['region']}/index_hour.json")
    if not idx: print(b["id"], b["name"], "no index", flush=True); continue
    rows = []
    for t in idx["timestamps"]:
        ts = pd.Timestamp(t, unit="ms", tz="UTC")
        if any(lo - pd.Timedelta(days=8) <= ts < hi for lo, hi in WIN):
            d = get(f"{B}/{b['id']}/{b['region']}/{b['id']}_{b['region']}_hour_{t}.json")
            if d: rows += d["series"]
            time.sleep(0.08)
    s = pd.Series({pd.Timestamp(a, unit="ms", tz="UTC"): v for a, v in rows if v is not None}, dtype=float).sort_index(); s.to_frame("mw").to_parquet(fn)
    print(f"{b['id']} {b['fuel']} {b['plant'][:30]} {b['name'][:24]}: {len(s)} hours, max {s.max() if len(s) else float('nan'):.0f} MW (listed {b['power']:.0f})", flush=True)
print("SHARD DONE")
