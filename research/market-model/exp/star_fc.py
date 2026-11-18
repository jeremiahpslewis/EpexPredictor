"""Forecasts from the coupled-market model. For each forecast origin (issued 14:00 on day D for D+2 00:00 onwards): parameters from the latest weekly
refit that uses data to the end of D-1; the market is cleared for the forecast hours with
  arm known: published residual loads of all nine zones (not available at the origin; the ceiling)
  arm ts:    Timestral forecasts of the German residual load and, zone by zone, of the neighbours'
Gas: the price known at the origin. Carbon: the last auction before the origin. Availability: as measured to D-1. Offer premium: last fitted level.
Temperature, wind and sun for the forecast hours come from the weather archive (not a forecast vintage; flagged).
usage: star_fc.py RUN PERIOD SHARD NSHARDS   -> out/starfc_<run>_<period>/<cutoff>.npz ; star_fc.py RUN PERIOD collect"""
import sys, os, time, glob, numpy as np
import star as S
run, period = sys.argv[1], sys.argv[2]; SP = S.lc.SP; OUT = f"{SP}/out/starfc_{run}_{period}"; os.makedirs(OUT, exist_ok=True)
I = np.load(f"{SP}/out/inputs_{period}.npz"); tags = [str(c)[:10] for c in I["cutoffs"]]; n = len(tags); EVERY = int(os.environ.get("EVERY", 7))
if sys.argv[3] == "collect":
    res = {}
    for arm in ("known", "ts"):
        v = np.full((n, 153), np.nan)
        for j, t in enumerate(tags):
            f = f"{OUT}/{t}.npz"
            if os.path.exists(f): v[j] = np.load(f)[arm]
        res[arm] = v; np.savez_compressed(f"{SP}/out/starfc_{run}_{period}_{arm}.npz", vint=v, cutoffs=I["cutoffs"]); a = I["actual"].astype(float); e = np.abs(v - a)
        print(f"{run} {period} {arm}: {np.isfinite(v[:, 0]).sum()} of {n} origins; MAE ct night {np.nanmean(e[:, :32]):.2f}, first 14 h {np.nanmean(e[:, :56]):.2f}, 14-24 h {np.nanmean(e[:, 56:96]):.2f}; bias night {np.nanmean((a - v)[:, :32]):+.2f}")
    sys.exit()
shard, nsh = int(sys.argv[3]), int(sys.argv[4]); G = np.load(f"{SP}/out/gen_ts_{period}.npz"); NBD = f"{SP}/out/tsnbz_{period}"
blocks = [list(range(a, min(a + EVERY, n))) for a in range(0, n, EVERY)]
for b, js in enumerate(blocks):
    if b % nsh != shard or all(os.path.exists(f"{OUT}/{tags[j]}.npz") for j in js): continue
    t0 = time.time(); d0f = int(I["cuts"][js[0]]); q, raw, dfit, Hfit, loss = S.fit(d0f - 96); nw = dfit["nw"]                     # data to the end of D-1 of the block's first origin
    np.savez_compressed(f"{OUT}/fit_{tags[js[0]]}.npz", **{f"raw_{k}": v for k, v in raw.items()}, loss=loss); print(f"block {b}: fit for {tags[js[0]]} in {time.time() - t0:.0f}s, loss {loss:.0f}", flush=True)
    for j in js:
        d0 = int(I["cuts"][j]); H = S.window(d0, d0 + 156); nh = len(H["P"]); d = S.data_of(H, np.full(nh, nw - 1), nw)
        d["gas"] = np.full(nh, float(I["gas"][j])); d["eua"] = np.full(nh, float(S.eua_all[d0 - 192])) * (1.0 if S.USE_EUA else 0.0)
        av = np.zeros((nh, S.Z, S.KB))
        for (z, k), a_ in S.AV.items(): av[:, S.ZN.index(z), k] = a_[d0 - 96]
        d["avail"] = np.nan_to_num(av); d["drl"] = np.zeros(nh); d["ok"] = np.ones((nh, S.Z), bool)
        out = {}
        for arm in ("known", "ts"):
            if arm == "known": rl = np.nan_to_num(H["rl"])
            else:
                zf = f"{NBD}/{tags[j]}.npz"
                if not os.path.exists(zf): continue
                zz = np.load(zf); zl = [str(x) for x in zz["zones"]]; q15 = np.column_stack([G["rl_pes"][j][:152]] + [zz["pes_z"][zl.index(z)][:152] for z in S.ZN[1:]]).astype(float)
                rl = np.vstack([q15.reshape(38, 4, S.Z).mean(1), np.column_stack([G["rl_pes"][j][152:153]] + [zz["pes_z"][zl.index(z)][152:153] for z in S.ZN[1:]]).astype(float)])[:nh]
            pe, _ = S.equilibrium(q, d, rl=rl); out[arm] = np.repeat(pe[:, 0] / 10.0, 4)[:153]
        np.savez_compressed(f"{OUT}/{tags[j]}.npz", **out)
    print(f"block {b} done ({len(js)} origins) in {time.time() - t0:.0f}s", flush=True)
print("SHARD DONE")
