"""Outside knowledge for a prior: how much of a change in gas-fired cost L = 1.85 gas + 0.37 carbon reaches the German price, by residual-load range?
2021-2023 (gas 18..227) against the current data. Residual load net of nuclear output, so that the ranges mean the same thing in both periods."""
import numpy as np, pandas as pd
import oc
SP = oc.SP; H = pd.read_parquet(f"{SP}/out/hist2123.parquet").loc["2021-01-01":"2023-12-31"]; H["x"] = H["resload"] - H["nuclear"]; H["L"] = 1.85 * H["gas"] + 0.37 * H["eua"]
ix = pd.DatetimeIndex(oc.S.idx[: oc.n15 // 4 * 4: 4]); C = pd.DataFrame({"p": oc.P, "x": oc.RL[:, 0], "gas": oc.GAS, "eua": oc.EUA}, index=ix); C["L"] = 1.85 * C["gas"] + 0.37 * C["eua"]
SETS = [("2021-2023 (gas 18..227)", H), ("  of which 2021", H.loc["2021"]), ("  of which 2022", H.loc["2022"]), ("  of which 2023", H.loc["2023"]), ("Mar 2024 - Feb 2026 (gas 28..55)", C.loc[:"2026-02-28"]), ("Mar - Nov 2026 (gas 35..95)", C.loc["2026-03-01":])]
bins = ((0, 15), (15, 25), (25, 35), (35, 50), (50, 90))
def slope(D):
    """median regression of the daily-bin median price on the daily L (one point per day and bin): slope and intercept; plus the plain ratio"""
    D = D.dropna(subset=["p", "L", "x"]); day = D.index.tz_convert("Europe/Berlin").normalize(); g = D.groupby(day).agg(p=("p", "median"), L=("L", "median"), n=("p", "size")); g = g[g["n"] >= 3]
    if len(g) < 40 or g["L"].std() < 3: return None
    b = np.polyfit(g["L"], g["p"], 1); r = g["p"] - np.polyval(b, g["L"])
    for _ in range(20): w = 1 / np.maximum(np.abs(r), 2.0); b = np.polyfit(g["L"], g["p"], 1, w=np.sqrt(w)); r = g["p"] - np.polyval(b, g["L"])      # least absolute deviations by reweighting
    return b[0], b[1], np.median(g["p"] / g["L"]), len(g), g["L"].quantile(0.05), g["L"].quantile(0.95)
print("slope of price on gas-fired cost L (least absolute deviations over days), intercept EUR/MWh, median price / L; by residual load net of nuclear")
for lab, D in SETS:
    print(lab)
    for a, b in bins:
        s = slope(D[(D["x"] >= a) & (D["x"] < b)])
        print(f"    [{a:2d},{b:2d}) GW: " + ("too few days or too little cost variation" if s is None else f"slope {s[0]:4.2f}  intercept {s[1]:+6.1f}  price/L {s[2]:4.2f}   ({s[3]} days, L {s[4]:.0f}..{s[5]:.0f})"))
