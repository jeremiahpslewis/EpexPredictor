"""Compare forecast files of one period against the first one: MAE at night / first 14 h, change with a block-bootstrap interval.
usage: cmp_eval.py PERIOD BLOCK label=file[:key] ..."""
import sys, numpy as np
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; period, blk = sys.argv[1], int(sys.argv[2])
I = np.load(f"{SP}/out/inputs_{period}.npz"); a = I["actual"].astype(float); n = len(a); rng = np.random.default_rng(0)
def ld(spec): f, _, k = spec.partition(":"); return np.load(f"{SP}/out/{f}.npz")[k or "vint"].astype(float)
items = [(s.split("=")[0], ld(s.split("=")[1])) for s in sys.argv[3:]]; ref = items[0][1]
idxs = [np.concatenate([(s + np.arange(blk)) % n for s in rng.integers(0, n, n // blk + 1)])[:n] for _ in range(4000)]
def ci(p1, p0, sl):
    e1, e0 = np.abs(p1 - a)[:, sl].mean(1), np.abs(p0 - a)[:, sl].mean(1); r = [100 * (e1[i].mean() / e0[i].mean() - 1) for i in idxs]
    return f"{100 * (e1.mean() / e0.mean() - 1):+4.0f}% [{np.percentile(r, 2.5):+.0f}..{np.percentile(r, 97.5):+.0f}]"
print(f"{period}: {n} forecasts, MAE ct/kWh{'':<22}night  34-48h  48-58h   vs first: night, 34-48 h")
for lab, p in items:
    e = np.abs(p - a); print(f"  {lab:<44}{e[:, :32].mean():6.2f}{e[:, :56].mean():8.2f}{e[:, 56:96].mean():8.2f}   " + ("" if p is ref else f"{ci(p, ref, slice(0, 32))}   {ci(p, ref, slice(0, 56))}"))
