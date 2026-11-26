import numpy as np
from panel import champion, MEM
P = champion(); X = P["X"]; a = P["actual"]; eq = X.mean(-1); e = eq - a
de = np.nanmean(e, 1); print("daily mean signed error of blend: mean %.3f sd %.3f; lag1 corr %.2f lag2 %.2f lag7 %.2f" % (np.nanmean(de), np.nanstd(de), *[np.corrcoef(de[:-k], de[k:])[0, 1] for k in (1, 2, 7)]))
for k, m in zip(MEM, range(3)):
    dm = np.nanmean(X[..., m] - a, 1); print(f"  {k}: mean bias {np.nanmean(dm):+.3f}, lag1 corr {np.corrcoef(dm[:-1], dm[1:])[0,1]:.2f}; 61d bias {np.nanmean(dm[:61]):+.3f} 18d bias {np.nanmean(dm[62:]):+.3f}")
# share of MAE that is daily-level bias
lv = np.nanmean(np.abs(e - de[:, None]), 1); print("blend MAE 61d %.3f; after removing each day's own mean error (oracle level) %.3f; 18d %.3f -> %.3f" % (np.nanmean(np.abs(e[:61])), np.nanmean(lv[:61]), np.nanmean(np.abs(e[62:])), np.nanmean(lv[62:])))
med = np.nanmedian(e, 1); lv2 = np.nanmean(np.abs(e - med[:, None]), 1); print("   removing each day's median error: 61d %.3f 18d %.3f" % (np.nanmean(lv2[:61]), np.nanmean(lv2[62:])))
# oracle: per-day scale+level around daily mean (shape vs level)
F = eq.copy()
for d in range(len(a)):
    m = np.isfinite(a[d]); A = np.c_[np.ones(m.sum()), eq[d, m]]; c = np.linalg.lstsq(A, a[d, m], rcond=None)[0]; F[d, m] = A @ c
print("oracle per-day affine (level+scale) of blend: 61d %.3f 18d %.3f" % (np.nanmean(np.abs(F - a)[:61]), np.nanmean(np.abs(F - a)[62:])))
# slot error autocorr same qh lag 1 day, after removing daily mean
r = e - de[:, None]; print("lag-1 corr of within-day shape error (same slot):", np.round(np.corrcoef(r[:-1, :92].ravel()[np.isfinite(r[:-1,:92].ravel()*r[1:,:92].ravel())], r[1:, :92].ravel()[np.isfinite(r[:-1,:92].ravel()*r[1:,:92].ravel())])[0, 1], 2))
