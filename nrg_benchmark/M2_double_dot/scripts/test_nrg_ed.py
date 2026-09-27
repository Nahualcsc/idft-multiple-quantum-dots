"""Exact-diagonalisation check of nrg_multidot on a short Wilson chain.

With no truncation the NRG iteration must reproduce, to machine precision,
  (i)   the many-body spectrum of dots + L chain sites per channel,
  (ii)  ground-state occupations and <n_1 n_2>,
  (iii) the Lehmann spectral weights of d_{i,up} and of the F-operator
        [d_{i,up}, H_int], at T -> 0 (FDM reduces to Lehmann).
Run:  python test_nrg_ed.py
"""
import numpy as np
from scipy.linalg import eigh
import nrg_multidot as nm


def ed_full(model, L, Lam, z):
    t, epsc, norm = nm.wilson_chain(Lam, z, L + 1)
    V = model.V() * norm
    M, F = model.M, model.F
    nmodes = F * (L + 1)                 # dots first, then chain site 0, 1, ...
    dim = 1 << nmodes

    def ann(mode):
        # Jordan-Wigner annihilator for 'mode' (bit index)
        A = np.zeros((dim, dim))
        for s in range(dim):
            if (s >> mode) & 1:
                sign = (-1) ** bin(s & ((1 << mode) - 1)).count("1")
                A[s ^ (1 << mode), s] = sign
        return A

    c = [ann(k) for k in range(nmodes)]
    n = [ci.T @ ci for ci in c]
    H = np.zeros((dim, dim))
    nd = [n[2 * i] + n[2 * i + 1] for i in range(M)]
    for i in range(M):
        H += model.eps[i] * nd[i] + model.U[i] * n[2 * i] @ n[2 * i + 1]
        for j in range(i + 1, M):
            H += model.Uij[i, j] * nd[i] @ nd[j]
    for f in range(F):
        i = f // 2
        m0 = F + f                        # site 0, flavour f
        H += V[i] * (c[f].T @ c[m0] + c[m0].T @ c[f])
        for site in range(L):
            H += epsc[site] * n[F * (site + 1) + f]
        for site in range(L - 1):
            a, b = F * (site + 1) + f, F * (site + 2) + f
            H += t[site] * (c[a].T @ c[b] + c[b].T @ c[a])
    E, U = eigh(H)
    return E, U, c, nd


def main():
    rng = np.random.default_rng(1)
    for (M, L) in [(1, 4), (2, 2)]:
        eps = rng.uniform(-0.6, 0.1, M)
        Uon = rng.uniform(0.3, 1.0, M)
        Uij = 0.25 if M == 2 else 0.0
        Gam = rng.uniform(0.1, 0.4, M)
        model = nm.MultiDot(eps, Uon, Uij, Gam)
        Lam, z = 2.5, 0.7
        E, U, c, nd = ed_full(model, L, Lam, z)
        run = nm.NRGRun(model, Lam=Lam, z=z, nshells=L, Ecut=1e9, Nkeep=10 ** 9, T=1e-9).run()
        Enrg = np.sort(np.concatenate([b["E"] for b in run.shells[-1]["blocks"].values()])) + run.shells[-1]["offset"]
        print(f"M={M} L={L}: dim={len(E)}  max|E_nrg - E_ed| = {np.max(np.abs(Enrg - E)):.2e}")

        # ground-state expectation values (T -> 0; average over ground multiplet)
        g = np.abs(E - E[0]) < 1e-9
        rho = U[:, g] @ U[:, g].T / g.sum()
        ex = run.expectation()
        for i in range(M):
            print(f"   <n_{i}>  ED={np.trace(rho @ nd[i]):.12f}  NRG={ex[('n', i)]:.12f}")
        if M == 2:
            print(f"   <n0n1> ED={np.trace(rho @ nd[0] @ nd[1]):.12f}  NRG={ex[('nn', 0, 1)]:.12f}")

        # Lehmann weights of d_{i,up} and O_i, binned like the NRG
        bins = nm.LogBins(per_decade=50)
        hist = run.fdm_spectral(bins)
        for i in range(M):
            f = 2 * i
            d = c[f]
            others = sum(model.Uij[i, j] * nd[j] for j in range(M) if j != i) if M > 1 else 0
            O = (model.U[i] * (c[f + 1].T @ c[f + 1]) + others) @ d
            dE = U.T @ d @ U
            OE = U.T @ O @ U
            p = np.exp(-(E - E[0]) / 1e-9); p /= p.sum()
            om = E[None, :] - E[:, None]              # E_s - E_r for <r|d|s>
            Wd = dE * dE * (p[:, None] + p[None, :])
            WO = OE * dE * (p[:, None] + p[None, :])
            hd = [np.zeros(bins.nb), np.zeros(bins.nb), 0.0]
            hO = [np.zeros(bins.nb), np.zeros(bins.nb), 0.0]
            bins.add(hd, om, Wd)
            bins.add(hO, om, WO)
            a, bO = hist[("d", i)], hist[("O", i)]
            print(f"   dot {i}: sum rule ED={hd[2]:.12f} NRG={a[2]:.12f} | "
                  f"max|hist diff| d: {max(np.abs(a[0]-hd[0]).max(), np.abs(a[1]-hd[1]).max()):.2e}  "
                  f"O: {max(np.abs(bO[0]-hO[0]).max(), np.abs(bO[1]-hO[1]).max()):.2e}")


if __name__ == "__main__":
    main()
