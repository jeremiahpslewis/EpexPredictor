"""
Mixed-regime likelihood checks. Usage: mixed_regime.py SP OUT_DIR toy|recover [N_MTU]

Likelihood used throughout: the JOINT change-of-variables density on each regime's support,
    f(p, n | theta) = f_u(u(p, n; theta)) * |det(Q' diag(S'(p; theta)) Q)|,
where Q is an orthonormal basis of the price subspace V allowed by the observed binding set. The regime is not
conditioned on: the binding set only selects which support (and Q) the observation lies on. Integrated over each
support, this density gives that regime's probability, so no normalising term P(regime | theta) appears.

toy:     two zones, one line of capacity K, linear supply S_i = b_i p, Gaussian shocks. Integrates the density over the
         coupled support and both congested supports, compares with simulated regime frequencies, and shows that
         P(regime | theta) moves with theta (a regime-conditional likelihood would need that theta-dependent normaliser).
recover: eleven zones on real published Core domains (JAO bulk, first MTUs of the sample). Synthetic logistic supply
         per zone with pooled location and scale, Gaussian shocks; cleared by the welfare dual (cvxpy); binding set
         read from the simulated duals (as from published shadow prices); fit by NUTS with the likelihood above.
"""
import sys, json, glob, time
from pathlib import Path
import numpy as np, pandas as pd

SP, OUT, MODE = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]; OUT.mkdir(parents=True, exist_ok=True)

if MODE == "toy":
    from scipy import integrate, stats
    res = {}
    for (b1, b2, K, D1, D2, s) in [(1.0, 2.0, 1.0, 3.0, 0.5, 1.0), (1.0, 2.0, 1.0, 3.0, 0.5, 2.0), (0.5, 3.0, 1.5, 2.0, 2.0, 1.0)]:
        f = lambda u1, u2: stats.norm.pdf(u1, 0, s) * stats.norm.pdf(u2, 0, s)
        # zone balance: b_i p_i - D_i - u_i = n_i, n1 = -n2 = n (export of zone 1), |n| <= K
        # coupled support (p, n), |n| < K: u1 = b1 p - D1 - n, u2 = b2 p - D2 + n, |det| = b1 + b2
        cpl = integrate.dblquad(lambda n, p: f(b1 * p - D1 - n, b2 * p - D2 + n) * (b1 + b2), -30, 30, -K, K)[0]
        # congested, n = +K (zone 1 exports at the limit, p1 <= p2): (p1, p2), |det| = b1 b2
        cg_p = integrate.dblquad(lambda p2, p1: f(b1 * p1 - D1 - K, b2 * p2 - D2 + K) * b1 * b2, -30, 30, lambda p1: p1, 30)[0]
        cg_m = integrate.dblquad(lambda p2, p1: f(b1 * p1 - D1 + K, b2 * p2 - D2 - K) * b1 * b2, -30, 30, -30, lambda p1: p1)[0]
        rng = np.random.default_rng(0); u = rng.normal(0, s, (400000, 2))
        p_unc = (D1 + D2 + u[:, 0] + u[:, 1]) / (b1 + b2); n_unc = b1 * p_unc - D1 - u[:, 0]
        sim = dict(coupled=float(np.mean(np.abs(n_unc) < K)), cong_plus=float(np.mean(n_unc >= K)), cong_minus=float(np.mean(n_unc <= -K)))
        res[f"b=({b1},{b2}) K={K} D=({D1},{D2}) s={s}"] = dict(coupled=cpl, cong_plus=cg_p, cong_minus=cg_m, total=cpl + cg_p + cg_m, simulated=sim)
    print(json.dumps(res, indent=1)); json.dump(res, open(OUT / "toy.json", "w"), indent=1)

if MODE == "recover":
    import cvxpy as cp, jax, jax.numpy as jnp, numpyro, numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS, init_to_median
    from numpyro.infer.util import log_density
    jax.config.update("jax_enable_x64", True); numpyro.set_host_device_count(4)
    NMTU = int(sys.argv[4]) if len(sys.argv) > 4 else 300
    CORE = ["DE", "FR", "NL", "BE", "AT", "CZ", "PL"]; NORD = ["DK_1", "DK_2", "SE_4", "NO_2"]; Z = CORE + NORD; OMIT = ["HR", "HU", "RO", "SI", "SK"]
    LINKS = [("DE", "DK_1"), ("DE", "DK_2"), ("DE", "SE_4"), ("DE", "NO_2"), ("NL", "NO_2"), ("NL", "DK_1"), ("PL", "SE_4")]; LIM = 1500.0
    ny = len(CORE) + 1 + len(LINKS); zi = {z: i for i, z in enumerate(Z)}
    H = np.zeros((len(Z), ny))
    for i, z in enumerate(CORE): H[zi[z], i] = 1.0
    for k, (a, b) in enumerate(LINKS): j = len(CORE) + 1 + k; H[zi[a], j] += 1; H[zi[b], j] -= 1
    e = np.zeros(ny); e[:len(CORE)] = 1.0
    # real domains: presolved rows and omitted-hub net positions for the first hours of the bulk sample
    fcs = sorted(glob.glob(str(SP / "jao/bulk/finalComputation_*.parquet")))[:6]
    FC = pd.concat([pd.read_parquet(f) for f in fcs]); NPJ = pd.concat([pd.read_parquet(f.replace("finalComputation", "netPos")) for f in fcs]).set_index("dateTimeUtc")
    hours = sorted(FC.dateTimeUtc.unique()); rng = np.random.default_rng(1)
    TRUE = dict(mu_a=80.0, tau_a=15.0, mu_ls=np.log(25.0), tau_ls=0.25, sd_u=1.5)
    a_true = rng.normal(TRUE["mu_a"], TRUE["tau_a"], len(Z)); ls_true = rng.normal(TRUE["mu_ls"], TRUE["tau_ls"], len(Z))
    B = np.array([60.0 if z in CORE else 15.0 for z in Z]); B[zi["DE"]] = 50.0; B[zi["FR"]] = 70.0       # GW, known
    Sg = lambda p, a, ls: B * 1000 / (1 + np.exp(-(p - a) / np.exp(ls)))                                   # MW
    rows = []; t0 = time.time()
    for m in range(NMTU):
        h = hours[m % len(hours)]; g = FC[FC.dateTimeUtc == h]; npr = NPJ.loc[h] if not isinstance(NPJ.loc[h], pd.DataFrame) else NPJ.loc[h].iloc[0]
        hubs = [c[5:] for c in g.columns if c.startswith("ptdf_")]
        P = g[[f"ptdf_{x}" for x in hubs]].astype(float).fillna(0).to_numpy(); hc = {x: i for i, x in enumerate(hubs)}
        A = np.zeros((len(g), ny))
        for i, z in enumerate(CORE): A[:, i] = P[:, hc[z]]
        A[:, len(CORE)] = P[:, hc["ALDE"]] - P[:, hc["ALBE"]]
        npo = np.array([float(npr.get(f"hub_{o}", 0) or 0) for o in OMIT]); b = g.ram.astype(float).to_numpy() - P[:, [hc[o] for o in OMIT]] @ npo; e0 = -npo.sum()
        y_obs = np.zeros(ny); y_obs[:len(CORE)] = [float(npr.get(f"hub_{z}")) for z in CORE]; y_obs[len(CORE)] = float(npr.get("hub_ALDE"))
        b = np.maximum(b, A @ y_obs)                                    # published outcome feasible (calibration rule)
        Dbase = Sg(80.0, a_true, ls_true) - H @ y_obs                   # centre: observed net positions at p = 80
        u = rng.normal(0, TRUE["sd_u"] * 1000, len(Z))
        load = Dbase + u
        pz = cp.Variable(len(Z)); mu = cp.Variable(len(g), nonneg=True); lam = cp.Variable(); nl = cp.Variable(ny, nonneg=True); nh = cp.Variable(ny, nonneg=True)
        lo = np.full(ny, -np.inf); hi = np.full(ny, np.inf); lo[len(CORE)], hi[len(CORE)] = -1000, 1000; lo[len(CORE) + 1:], hi[len(CORE) + 1:] = -LIM, LIM
        sc = np.exp(ls_true)
        fb = np.isfinite(hi)
        obj = sum(B[i] * 1000 * sc[i] * cp.logistic((pz[i] - a_true[i]) / sc[i]) - load[i] * pz[i] for i in range(len(Z))) + b @ mu + lam * e0 \
              + hi[fb] @ nh[fb] - lo[fb] @ nl[fb]
        con = [H.T @ pz + A.T @ mu + e * lam + nh - nl == 0, nh[~fb] == 0, nl[~fb] == 0]
        pr = cp.Problem(cp.Minimize(obj), con); pr.solve(solver="CLARABEL")
        if pr.status != "optimal": continue
        p = np.asarray(pz.value); y = -con[0].dual_value; n = H @ y
        # binding set from the simulated duals, exactly as from published shadow prices / links at their limits
        AB = [A[k] for k in np.flatnonzero(mu.value > 1e-4)] + [np.eye(ny)[j] for j in range(len(CORE), ny) if nh.value[j] > 1e-4 or nl.value[j] > 1e-4]
        AB = np.array(AB) if AB else np.zeros((0, ny))
        M = np.hstack([H.T, -AB.T, -e[:, None]]); _, sv, Vt = np.linalg.svd(M); rk = int((sv > 1e-9 * sv[0]).sum())
        U_, s_, _ = np.linalg.svd(Vt[rk:].T[:len(Z)], full_matrices=False); Q = U_[:, s_ > 1e-8]
        resid = float(np.linalg.norm(p - Q @ (Q.T @ p)))
        Qp = np.zeros((len(Z), len(Z))); Qp[:, :Q.shape[1]] = Q
        rows.append(dict(p=p, n=n, D=Dbase, Q=Qp, k=Q.shape[1], resid=resid, nbind=int(AB.shape[0])))
    print(f"cleared {len(rows)} MTUs in {time.time()-t0:.0f}s; price-subspace residual max {max(r['resid'] for r in rows):.2e}")
    k = np.array([r["k"] for r in rows]); print("regimes by dim V:", {int(v): int((k == v).sum()) for v in np.unique(k)})
    pz_ = jnp.asarray(np.array([r["p"] for r in rows])); n_ = jnp.asarray(np.array([r["n"] for r in rows])); D_ = jnp.asarray(np.array([r["D"] for r in rows]))
    Q_ = jnp.asarray(np.array([r["Q"] for r in rows])); pad = jnp.asarray(np.array([[0.0] * r["k"] + [1.0] * (len(Z) - r["k"]) for r in rows]))
    Bj = jnp.asarray(B)
    def model():
        mu_a = numpyro.sample("mu_a", dist.Normal(80, 30)); tau_a = numpyro.sample("tau_a", dist.HalfNormal(30))
        mu_ls = numpyro.sample("mu_ls", dist.Normal(np.log(25), 0.7)); tau_ls = numpyro.sample("tau_ls", dist.HalfNormal(0.5))
        za = numpyro.sample("za", dist.Normal(0, 1).expand([len(Z)])); zl = numpyro.sample("zl", dist.Normal(0, 1).expand([len(Z)]))
        a = numpyro.deterministic("a", mu_a + tau_a * za); ls = numpyro.deterministic("ls", mu_ls + tau_ls * zl)
        sd = numpyro.sample("sd_u", dist.HalfNormal(5.0))
        s = jnp.exp(ls); g = jax.nn.sigmoid((pz_ - a) / s); S = Bj * 1000 * g; dS = Bj * 1000 * g * (1 - g) / s
        u = S - D_ - n_
        Mt = jnp.einsum("tzk,tz,tzl->tkl", Q_, dS / 1000, Q_) + jax.vmap(jnp.diag)(pad)
        numpyro.factor("lik", dist.Normal(0, sd * 1000).log_prob(u).sum() + jnp.linalg.slogdet(Mt)[1].sum())
    mc = MCMC(NUTS(model, target_accept_prob=0.9, init_strategy=init_to_median(num_samples=50)), num_warmup=800, num_samples=800, num_chains=4, chain_method="parallel", progress_bar=False)
    mc.run(jax.random.PRNGKey(0)); Sm = mc.get_samples(); sm = numpyro.diagnostics.summary(mc.get_samples(group_by_chain=True))
    rh = max(float(np.max(v["r_hat"])) for v in sm.values()); div = int(np.asarray(mc.get_extra_fields()["diverging"]).sum())
    G = mc.get_samples(group_by_chain=True)
    print("per-chain median mu_a, mu_ls, sd_u:", [(round(float(np.median(G['mu_a'][c])), 1), round(float(np.median(G['mu_ls'][c])), 2), round(float(np.median(G['sd_u'][c])), 2)) for c in range(4)])
    tr = {"mu_a": TRUE["mu_a"], "tau_a": TRUE["tau_a"], "mu_ls": TRUE["mu_ls"], "tau_ls": TRUE["tau_ls"], "sd_u": TRUE["sd_u"],
          "za": (a_true - TRUE["mu_a"]) / TRUE["tau_a"], "zl": (ls_true - TRUE["mu_ls"]) / TRUE["tau_ls"]}
    lp_true = float(log_density(model, (), {}, tr)[0])
    lp_ch = [float(log_density(model, (), {}, {k: np.median(np.asarray(G[k][c]), axis=0) for k in tr})[0]) for c in range(4)]
    print("log density at truth:", round(lp_true, 1), " at chain medians:", [round(v, 1) for v in lp_ch])
    out = {"n_mtu": len(rows), "logdens_truth": lp_true, "logdens_chains": lp_ch, "regimes_by_dimV": {int(v): int((k == v).sum()) for v in np.unique(k)}, "max_rhat": rh, "divergences": div}
    ins = []
    for name, tv in [("a", a_true), ("ls", ls_true)]:
        q = np.quantile(np.asarray(Sm[name]), [0.05, 0.95], axis=0); ins += list((q[0] <= tv) & (tv <= q[1]))
        out[name] = dict(true=np.round(tv, 3).tolist(), q05=np.round(q[0], 3).tolist(), q95=np.round(q[1], 3).tolist())
    for name in ("mu_a", "tau_a", "mu_ls", "tau_ls", "sd_u"):
        q = np.quantile(np.asarray(Sm[name]), [0.05, 0.5, 0.95]); out[name] = dict(true=float(TRUE[name]), q05=float(q[0]), med=float(q[1]), q95=float(q[2]))
    out["zone_params_inside_90"] = f"{int(np.sum(ins))} of {len(ins)}"
    # same fit without the Jacobian term, to show it matters
    print(json.dumps(out, indent=1)); json.dump(out, open(OUT / "recover.json", "w"), indent=1)
