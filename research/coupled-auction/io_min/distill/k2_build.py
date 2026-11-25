"""K2 records: the strict holdout prediction records with Germany's residual-load change against the lag MTU scaled by
KAPPA (a first-order flattening of the German response, fixed from training before the holdout is scored).
Usage: k2_build.py KAPPA OUT.pkl. Then run ../predict11.py on OUT.pkl with VARIANTS=b, NPOST=2 (as oos11s)."""
import sys, pickle
import numpy as np
E11 = "/tmp/claude-0/-home-user-EpexPredictor/1711225d-3f79-42cf-ba0d-e4080065c4b2/scratchpad/e11"
k, OUT = float(sys.argv[1]), sys.argv[2]
d = pickle.load(open(f"{E11}/pred_holdout_strict.pkl", "rb")); out = []
for r in d["recs"]:
    rr = dict(r); D = np.array(r["D"], float); D[0] = r["lag"]["D"][0] + k * (D[0] - r["lag"]["D"][0]); rr["D"] = D; out.append(rr)
pickle.dump(dict(recs=out, skipped=d.get("skipped", {})), open(OUT, "wb")); print(len(out), "records, kappa", k)
