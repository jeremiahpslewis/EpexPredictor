"""SMARD production per plant block: fill in the rest of the history (July 2024 to mid-November 2026) for the selected blocks; merges into the existing parquet per block."""
import sys, os, json, time, urllib.request
import pandas as pd
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; OUT = f"{SP}/ext/smard/blocks"; B = "https://www.smard.de/app/chart_data"
LO, HI = pd.Timestamp("2024-07-01", tz="UTC"), pd.Timestamp("2026-11-17", tz="UTC")
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
    if i % nsh != shard or "etrieb" not in b["status"]: continue
    fn = f"{OUT}/{b['id']}.parquet"; done = f"{OUT}/{b['id']}.full"
    if os.path.exists(done): continue
    old = pd.read_parquet(fn)["mw"] if os.path.exists(fn) else pd.Series(dtype=float); have = set(old.index.floor("7D")) if len(old) else set()
    idx = get(f"{B}/{b['id']}/{b['region']}/index_hour.json")
    if not idx: continue
    rows = []
    for t in idx["timestamps"]:
        ts = pd.Timestamp(t, unit="ms", tz="UTC")
        if LO - pd.Timedelta(days=8) <= ts < HI and not (len(old) and ((old.index >= ts) & (old.index < ts + pd.Timedelta(days=7))).sum() >= 160):
            d = get(f"{B}/{b['id']}/{b['region']}/{b['id']}_{b['region']}_hour_{t}.json")
            if d: rows += d["series"]
            time.sleep(0.08)
    new = pd.Series({pd.Timestamp(a, unit="ms", tz="UTC"): v for a, v in rows if v is not None}, dtype=float); s = pd.concat([old, new]); s = s[~s.index.duplicated(keep="last")].sort_index(); s.to_frame("mw").to_parquet(fn); open(done, "w").write("ok")
    print(f"{b['id']} {b['fuel']} {b['plant'][:24]}: {len(s)} hours, {s.index.min().date() if len(s) else None} .. {s.index.max() if len(s) else None}", flush=True)
print("SHARD DONE")
