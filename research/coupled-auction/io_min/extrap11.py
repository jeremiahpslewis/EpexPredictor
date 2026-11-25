"""
Inputs-only extrapolation diagnostic for the neighbour curves. Usage: extrap11.py POST.npz TRAIN_RECS.pkl HOLDOUT_PRED_RECS.pkl
For each neighbour: supply (GW) at a fixed price of 150 EUR/MWh at the training-median fuel-cost index and at the
holdout-median index (holdout inputs only, no outcomes), posterior median and 90% interval of the change.
"""
import sys, pickle, numpy as np
P = dict(np.load(sys.argv[1], allow_pickle=True))
Lt = np.median([r["Lc"] for r in pickle.load(open(sys.argv[2], "rb"))["recs"]]); Lh = np.median([r["Lc"] for r in pickle.load(open(sys.argv[3], "rb"))["recs"]])
Z = ["FR", "NL", "BE", "AT", "CZ", "PL", "DK1", "DK2", "SE4", "NO2"]
def S(L, p=150.0):
    g = 1 / (1 + np.exp(-(p - P["aa"] - P["bb"] * L) / (P["sg"] * L))); return P["sbar"][1:] * (P["al"] + P["be"] * g)
d = S(Lh) - S(Lt)
print(f"cost index median: training {Lt:.0f}, holdout {Lh:.0f}. Change in neighbour supply at 150 EUR/MWh (GW), median [5%, 95%]:")
for j, z in enumerate(Z): print(f"  {z:4s} {np.median(d[:, j]):+7.2f} [{np.quantile(d[:, j], .05):+7.2f}, {np.quantile(d[:, j], .95):+7.2f}]   b median {np.median(P['bb'][:, j]):+.2f}")
print(f"  sum  {np.median(d.sum(1)):+7.2f} [{np.quantile(d.sum(1), .05):+7.2f}, {np.quantile(d.sum(1), .95):+7.2f}]")
