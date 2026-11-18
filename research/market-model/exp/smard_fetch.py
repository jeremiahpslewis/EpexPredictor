"""SMARD chart data: weekly chunks of one series, assembled to a parquet. usage: smard_fetch.py REGION RESOLUTION START ID[:name] ...   (resolution: hour | quarterhour)"""
import sys, os, json, time, urllib.request
import pandas as pd
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; OUT = f"{SP}/ext/smard/series"; os.makedirs(OUT, exist_ok=True)
region, res, start = sys.argv[1], sys.argv[2], pd.Timestamp(sys.argv[3], tz="UTC"); B = "https://www.smard.de/app/chart_data"
def get(url):
    for a in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "research-script"}), timeout=60) as r: return json.loads(r.read().decode())
        except Exception as e:
            if "404" in str(e): return None
            time.sleep(3 + 5 * a)
    return None
for spec in sys.argv[4:]:
    fid, _, name = spec.partition(":"); name = name or fid; fn = f"{OUT}/{name}_{region}_{res}.parquet"
    if os.path.exists(fn): print(name, "exists"); continue
    idx = get(f"{B}/{fid}/{region}/index_{res}.json")
    if not idx: print(name, "no index"); continue
    ts = [t for t in idx["timestamps"] if pd.Timestamp(t, unit="ms", tz="UTC") >= start - pd.Timedelta(days=8)]; rows = []
    for t in ts:
        d = get(f"{B}/{fid}/{region}/{fid}_{region}_{res}_{t}.json")
        if d: rows += d["series"]
        time.sleep(0.12)
    s = pd.Series({pd.Timestamp(a, unit="ms", tz="UTC"): b for a, b in rows if b is not None}, dtype=float).sort_index(); s = s[s.index >= start]; s.to_frame(name).to_parquet(fn)
    print(f"{name}: {len(s)} values, {s.index.min()} .. {s.index.max()}, mean {s.mean():.0f}", flush=True)
