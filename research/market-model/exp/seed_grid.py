import glob, sys, pandas as pd
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"
H = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{SP}/dense/hist_*.parquet"))]).sort_index(); H = H[~H.index.duplicated()]
out = {}
for i in range(58):
    out[f"wind_{i}"] = H[f"w120_{i}"]; out[f"radiation_{i}"] = H[f"sw_{i}"]; out[f"temp_{i}"] = H[f"t2_{i}"]
df = pd.DataFrame(out).astype(float).dropna(); df.index.name = "time"; print(df.shape, df.index.min(), df.index.max(), "rows dropped for missing values:", len(H) - len(df))
df.to_json(sys.argv[1] + "/gridweather_v1_DE_LU.json.gz", compression="gzip")
