"""Key metric (user, notes 21.28): forecast made at midnight for 00:00-06:00 of the next day (quarter-hour slots 0-23 of the target day).
The existing forecasts (issued 14:00 two days before the target) use nothing that is not known at that midnight: same prices, older weather and
quantities. So they are valid, slightly conservative entries. usage: key_metric.py [fig]"""
import sys, numpy as np, warnings; warnings.filterwarnings("ignore")
SP = "/tmp/claude-0/-home-user-EpexPredictor/d6125cc0-c431-5b93-a06f-0110cd871ab6/scratchpad"; CR = {"winter": "hw2", "summer": "hs2", "autumn": "ha"}; TS = {"winter": ("win_ts", "median"), "summer": ("ssn_ts", "median"), "autumn": ("tsl_123_8k", "vint")}; SL = slice(0, 24)
def load(per):
    I = np.load(f"{SP}/out/inputs_{per}.npz"); a = I["actual"].astype(float); ld = lambda f, k="vint": np.load(f"{SP}/out/{f}.npz")[k].astype(float); tp = ld(*TS[per]); cc = ld(f"structfc_{CR[per]}_ts_dyn2xxh")
    F = {"candidate (curve + Timestral)": 0.5 * (cc + tp), "candidate's curve + layer": cc, "Timestral price": tp, "merit order + layer": ld(f"structfc_m1g_{per}_price", "vint_ts"), "single curve + layer": ld(f"structfc_pg0e_{per}_price", "vint_ts"), "plain curve + layer": ld(f"structfc_l1g_{per}_ts")}
    F["merit order + Timestral"] = 0.5 * (F["merit order + layer"] + tp); return a, F
if __name__ == "__main__":
    R = {}
    for per in CR:
        a, F = load(per); R[per] = (a, F); print(f"{per} ({len(a)} forecasts), 00:00-06:00, MAE ct (bias): " + " | ".join(f"{k}: {np.abs(v - a)[:, SL].mean():.2f} ({(v - a)[:, SL].mean():+.2f})" for k, v in F.items()))
    if len(sys.argv) > 1:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        BLUE, ORANGE, AQUA, YEL, INK, MUTED, SURF, GRID, NEUT = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#0b0b0b", "#52514e", "#fcfcfb", "#e9e8e3", "#a8a79f"
        plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.family": "DejaVu Sans"})
        ST = (("candidate (curve + Timestral)", AQUA, "-"), ("candidate's curve + layer", ORANGE, "-"), ("merit order + layer", BLUE, "-"), ("merit order + Timestral", BLUE, (0, (4, 2))), ("Timestral price", NEUT, "-")); fig, axs = plt.subplots(2, 3, figsize=(17.5, 8.6), gridspec_kw=dict(hspace=0.42, wspace=0.2)); h = np.arange(24) / 4
        for i, per in enumerate(CR):
            a, F = R[per]; ax = axs[0, i]
            for k, col, ls in ST: ax.plot(h, np.abs(F[k] - a)[:, SL].mean(0), color=col, linestyle=ls, linewidth=1.9, label=f"{k}: {np.abs(F[k] - a)[:, SL].mean():.2f}")
            ax.set_ylim(0, None); ax.set_xlabel("hour of the target night"); ax.set_ylabel("mean absolute error, ct/kWh"); ax.grid(True, color=GRID, linewidth=0.6); ax.legend(frameon=False, fontsize=7.8, loc="upper left", title="MAE 00:00-06:00, ct", title_fontsize=8)
            [ax.spines[s].set_visible(False) for s in ("top", "right")]; b = np.abs(F["merit order + layer"] - a)[:, SL].mean(); c = np.abs(F["candidate's curve + layer"] - a)[:, SL].mean(); ax.set_title(f"{per.capitalize()}: merit order {b:.2f}, candidate's curve {c:.2f} ct;\nerror by hour of the night", loc="left", fontsize=9.5, fontweight="bold", color=INK); ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
            ax = axs[1, i]; em = np.abs(F["merit order + layer"] - a)[:, SL].mean(1); ec = np.abs(F["candidate's curve + layer"] - a)[:, SL].mean(1); m_ = max(em.max(), ec.max()) * 1.05; ax.plot([0, m_], [0, m_], color=INK, linewidth=0.8); ax.scatter(ec, em, s=22, color=BLUE, alpha=0.7, linewidths=0); ax.set_xlim(0, m_); ax.set_ylim(0, m_); ax.set_xlabel("candidate's curve + layer: error of the night, ct"); ax.set_ylabel("merit order + layer: error of the night, ct"); ax.grid(True, color=GRID, linewidth=0.6)
            [ax.spines[s].set_visible(False) for s in ("top", "right")]; w = int((em < ec).sum()); big = int((np.maximum(em, ec) > 2 * np.median(np.r_[em, ec])).sum()); ax.set_title(f"Night by night: merit order better in {w} of {len(em)};\nthe mean is driven by a few bad nights", loc="left", fontsize=9.5, fontweight="bold", color=INK)
        fig.suptitle("Key metric: the price from midnight to 06:00, forecast one day ahead (existing forecasts, which use only what is known at that midnight; Timestral's load)", x=0.125, ha="left", fontsize=11, fontweight="bold", color=INK, y=0.97)
        fig.savefig(f"{SP}/out/key_metric.png", dpi=105, bbox_inches="tight"); print("figure ok")
