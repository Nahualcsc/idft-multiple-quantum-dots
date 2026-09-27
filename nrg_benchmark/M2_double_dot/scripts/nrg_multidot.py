"""
nrg_multidot.py -- full-density-matrix NRG for M capacitively coupled quantum
dots, each dot coupled to its OWN conduction channel (diagonal Gamma matrix).

    H = sum_i eps_i n_i + sum_i U_i n_i,up n_i,dn + sum_{i<j} U_ij n_i n_j
        + sum_{i,s} V_i (d^+_{i s} f_{0,i s} + h.c.) + sum_i H_lead,i

Leads: flat band of half-width D (= 1 internally), wide-band hybridisation
Gamma_i = 2 pi V_i^2 rho_0 (FULL width, identical to the i-DFT gamma; the
half-width is Delta_i = Gamma_i/2).

Symmetry used: the particle number of every flavour (i, sigma) is conserved,
because the dots only talk to each other through density-density terms and
each dot has its own channel -> U(1)^(2M), all abelian.

Spectral functions: full-density-matrix NRG (Weichselbaum & von Delft, PRL 99,
076402 (2007)); Campo-Oliveira discretisation with z-averaging; log-Gaussian
broadening; self-energy trick (Bulla, Hewson, Pruschke, JPCM 10, 8365 (1998))
generalised to inter-dot interactions:
    F_i = << [d_i, H_int] ; d_i^+ >>,  [d_i,up, H_int] = (U_i n_i,dn + sum_j U_ij n_j) d_i,up
    Sigma_i = F_i / G_i,  G_i = 1/(w - eps_i - Delta_i(w) - Sigma_i(w)).

Conventions of the returned spectra: A_i(w) = -Im G_i,sigma(w)/pi, per spin,
normalised to 1.  The i-DFT quantity gamma*A_i/4 of the manuscript equals
(Gamma_i/2) * A_i(w) here (both are 1/pi at the unitary limit).

Only numpy/scipy/mpmath/joblib are needed.
"""

import os
import math
import time
import pickle
import tempfile
from collections import defaultdict

import numpy as np
from scipy.linalg import eigh
from scipy.special import logsumexp


# ----------------------------------------------------------------------------
# Wilson chain (Campo-Oliveira discretisation of a flat band, z-shifted)
# ----------------------------------------------------------------------------
_CHAIN_CACHE = {}


def wilson_chain(Lam, z, nsites):
    """Hoppings t_n (n=0..nsites-2) and on-site energies eps_n of the Wilson
    chain for a flat band on [-1, 1] with total hybridisation weight 1.

    Campo & Oliveira, PRB 72, 104432 (2005): intervals
    [Lam^-(j+z), Lam^-(j-1+z)] (first one [Lam^-z, 1]) represented by the
    energy (b-a)/ln(b/a).  Tridiagonalised with Lanczos in multiprecision,
    because the hoppings span ~Lam^(-nsites/2)."""
    key = (float(Lam), float(z), int(nsites))
    if key in _CHAIN_CACHE:
        return _CHAIN_CACHE[key]
    import mpmath as mp
    J = nsites + 12
    mp.mp.dps = int(40 + 1.2 * nsites * math.log10(Lam))
    L, zz = mp.mpf(Lam), mp.mpf(z)
    x = [mp.mpf(1)] + [L ** (-(j - 1 + zz)) for j in range(1, J + 1)]
    E, W = [], []
    for j in range(J):
        a, b = x[j + 1], x[j]
        E.append((b - a) / mp.log(b / a))
        W.append((b - a) / 2)
    ener = E + [-e for e in E]
    wts = W + W
    norm = mp.sqrt(mp.fsum(wts))
    v = [mp.sqrt(w) / norm for w in wts]
    basis = [v]
    eps, t = [], []
    v_prev, beta_prev = None, mp.mpf(0)
    for n in range(nsites):
        w = [ener[k] * v[k] for k in range(len(v))]
        alpha = mp.fsum(w[k] * v[k] for k in range(len(v)))
        w = [w[k] - alpha * v[k] - (beta_prev * v_prev[k] if v_prev else 0)
             for k in range(len(v))]
        for u in basis:  # full re-orthogonalisation
            c = mp.fsum(w[k] * u[k] for k in range(len(v)))
            w = [w[k] - c * u[k] for k in range(len(v))]
        beta = mp.sqrt(mp.fsum(wk * wk for wk in w))
        eps.append(float(alpha))
        if n < nsites - 1:
            t.append(float(beta))
            v_prev, beta_prev = v, beta
            v = [wk / beta for wk in w]
            basis.append(v)
    out = (np.array(t), np.array(eps), float(norm))
    _CHAIN_CACHE[key] = out
    return out


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class MultiDot:
    """M dots, flavour f = 2*i + sigma (sigma = 0 up, 1 down)."""

    def __init__(self, eps, U, Uij, Gamma, D=1.0):
        self.eps = np.atleast_1d(np.asarray(eps, float))
        self.M = len(self.eps)
        self.F = 2 * self.M
        self.U = np.asarray(U, float) * np.ones(self.M)
        Uij = np.asarray(Uij, float)
        if Uij.ndim == 0:
            Uij = float(Uij) * (1.0 - np.eye(self.M))
        self.Uij = np.array(Uij, float)
        np.fill_diagonal(self.Uij, 0.0)
        assert np.allclose(self.Uij, self.Uij.T)
        self.Gamma = np.asarray(Gamma, float) * np.ones(self.M)
        self.D = float(D)

    def V(self):
        # Gamma = 2 pi V^2 rho0 with rho0 = 1/(2D)  ->  V^2 = Gamma D / pi
        return np.sqrt(self.Gamma * self.D / np.pi)


def _popcount(x):
    return bin(x).count("1")


def _local_tables(F):
    S = 1 << F
    occ = np.array([[(s >> f) & 1 for f in range(F)] for s in range(S)], dtype=np.int64)
    nsum = occ.sum(axis=1)
    # sign of moving an operator of flavour f past the creators of flavours < f
    sgn = np.array([[(-1) ** _popcount(s & ((1 << f) - 1)) for s in range(S)]
                    for f in range(F)], dtype=float)
    return S, occ, nsum, sgn


# ----------------------------------------------------------------------------
# One NRG run (single z)
# ----------------------------------------------------------------------------
class NRGRun:
    """Forward NRG iteration; optionally stores everything needed for FDM."""

    def __init__(self, model, Lam=3.0, z=1.0, nshells=45, Ecut=8.0, Nkeep=3000,
                 T=1e-10, spec_dots=None, store=True, degtol=1e-6, verbose=False,
                 Nkeep_early=None, n_early=0, disk_store_dir=None):
        self.m = model
        self.Lam, self.z = float(Lam), float(z)
        self.nshells = int(nshells)
        self.Ecut, self.Nkeep = float(Ecut), int(Nkeep)
        # larger truncation at the first shells, where the Wilson chain is not
        # yet scale separated (matters for multi-channel problems)
        self.Nkeep_early = int(Nkeep if Nkeep_early is None else Nkeep_early)
        self.n_early = int(n_early)
        self.T = float(T)
        self.spec_dots = list(range(model.M)) if spec_dots is None else list(spec_dots)
        self.store = store
        self.disk_store_dir = disk_store_dir
        self.degtol = degtol
        self.verbose = verbose
        F = model.F
        self.S, self.occ, self.nsum, self.sgn = _local_tables(F)
        self.t_chain, self.eps_chain, self.chain_norm = wilson_chain(self.Lam, self.z, self.nshells + 1)

    # ------------------------------------------------------------------
    def _dot_basis(self):
        m, F, S, occ = self.m, self.m.F, self.S, self.occ
        up, dn = occ[:, 0::2], occ[:, 1::2]
        nd = up + dn
        E = nd @ m.eps + (up * dn) @ m.U + 0.5 * np.einsum("si,ij,sj->s", nd, m.Uij, nd)
        E0 = E.min()
        KB = {"E": {}, "L": [dict() for _ in range(F)], "ops": {}, "nops": {}}
        q_of = [tuple(occ[s]) for s in range(S)]
        for s in range(S):
            KB["E"][q_of[s]] = np.array([E[s] - E0])
        for f in range(F):
            for s in range(S):
                if occ[s, f]:
                    KB["L"][f][q_of[s]] = np.array([[self.sgn[f, s]]])
        for i in self.spec_dots:
            f = 2 * i
            dmat, Omat = {}, {}
            for s in range(S):
                if occ[s, f]:
                    # O = [d_{i,up}, H_int] = (U_i n_{i,dn} + sum_j U_ij n_j) d_{i,up}
                    fac = m.U[i] * occ[s, f + 1] + sum(m.Uij[i, j] * nd[s, j] for j in range(m.M) if j != i)
                    dmat[q_of[s]] = np.array([[self.sgn[f, s]]])
                    Omat[q_of[s]] = np.array([[self.sgn[f, s] * fac]])
            KB["ops"][("d", i)] = (f, dmat)
            KB["ops"][("O", i)] = (f, Omat)
        for i in range(m.M):
            KB["nops"][("n", i)] = {q_of[s]: np.array([[float(nd[s, i])]]) for s in range(S)}
        for i in range(m.M):
            for j in range(i + 1, m.M):
                KB["nops"][("nn", i, j)] = {q_of[s]: np.array([[float(nd[s, i] * nd[s, j])]]) for s in range(S)}
        for i in range(m.M):
            KB["nops"][("docc", i)] = {q_of[s]: np.array([[float(up[s, i] * dn[s, i])]]) for s in range(S)}
        return KB, E0

    # ------------------------------------------------------------------
    def _build(self, KB, t_f, eps_site, scale):
        F, S, occ, nsum, sgn = self.m.F, self.S, self.occ, self.nsum, self.sgn
        new = defaultdict(list)
        for q in KB["E"]:
            qa = np.array(q)
            for s in range(S):
                new[tuple(qa + occ[s])].append((q, s))
        blocks = {}
        for qn, segs in new.items():
            sizes = [len(KB["E"][q]) for q, s in segs]
            offs = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
            dim = int(offs[-1])
            index = {seg: k for k, seg in enumerate(segs)}
            H = np.zeros((dim, dim))
            H[np.diag_indices(dim)] = np.concatenate(
                [KB["E"][q] + eps_site * nsum[s] for q, s in segs]) / scale
            for k, (q, s) in enumerate(segs):
                for f in range(F):
                    if (s >> f) & 1:
                        continue
                    Lm = KB["L"][f].get(q)
                    if Lm is None:
                        continue
                    q2 = list(q); q2[f] -= 1
                    k2 = index.get((tuple(q2), s | (1 << f)))
                    if k2 is None:
                        continue
                    amp = t_f[f] * (-1.0) ** nsum[s] * sgn[f, s] / scale
                    H[offs[k2]:offs[k2 + 1], offs[k]:offs[k + 1]] += amp * Lm
                    H[offs[k]:offs[k + 1], offs[k2]:offs[k2 + 1]] += amp * Lm.T
            E, U = eigh(H, overwrite_a=True, check_finite=False)
            blocks[qn] = {"segs": segs, "offs": offs, "index": index, "E": E * scale, "U": U}
        return blocks

    def _truncate(self, blocks, scale, last, Nkeep):
        allE = np.concatenate([b["E"] for b in blocks.values()])
        E0 = allE.min()
        if last:
            for b in blocks.values():
                b["nK"] = 0
                b["E"] = b["E"] - E0
            return E0, 0, len(allE)
        Es = np.sort((allE - E0) / scale)
        nk = int(np.searchsorted(Es, self.Ecut, side="right"))
        nk = max(1, min(nk, Nkeep))
        while nk < len(Es) and Es[nk] - Es[nk - 1] < self.degtol:
            nk += 1
        thr = np.inf if nk >= len(Es) else 0.5 * (Es[nk - 1] + Es[nk])
        for b in blocks.values():
            b["E"] = b["E"] - E0
            b["nK"] = int(np.count_nonzero(b["E"] / scale < thr))
        return E0, nk, len(allE)

    # ------------------------------------------------------------------
    # operator transformations
    def _transform_fermion(self, op_old, f, blocks, rsel="K", csel="K"):
        """op_old maps old block q -> q - e_f. Returns new matrices keyed by
        the column block qn, shape (rows of qn-e_f) x (cols of qn)."""
        nsum = self.nsum
        out = {}
        for qn, b in blocks.items():
            qt = list(qn); qt[f] -= 1; qt = tuple(qt)
            bt = blocks.get(qt)
            if bt is None:
                continue
            cs = _sel(b, csel); rs = _sel(bt, rsel)
            if cs.stop - cs.start == 0 or rs.stop - rs.start == 0:
                continue
            Ub = b["U"][:, cs]
            Y = np.zeros((bt["U"].shape[0], cs.stop - cs.start))
            hit = False
            for k, (q, s) in enumerate(b["segs"]):
                mo = op_old.get(q)
                if mo is None:
                    continue
                qo = list(q); qo[f] -= 1
                k2 = bt["index"].get((tuple(qo), s))
                if k2 is None:
                    continue
                Y[bt["offs"][k2]:bt["offs"][k2 + 1]] += ((-1.0) ** nsum[s]) * (mo @ Ub[b["offs"][k]:b["offs"][k + 1]])
                hit = True
            if hit:
                out[qn] = bt["U"][:, rs].T @ Y
        return out

    def _transform_boson(self, op_old, blocks, rsel="K", csel="K", diag_only=False):
        out = {}
        for qn, b in blocks.items():
            cs = _sel(b, csel)
            if cs.stop - cs.start == 0:
                continue
            Ub = b["U"][:, cs]
            Y = np.zeros_like(Ub)
            for k, (q, s) in enumerate(b["segs"]):
                mo = op_old.get(q)
                if mo is None:
                    continue
                sl = slice(b["offs"][k], b["offs"][k + 1])
                Y[sl] = mo @ Ub[sl]
            if diag_only:
                out[qn] = np.einsum("ij,ij->j", Ub, Y)
            else:
                rs = _sel(b, rsel)
                out[qn] = b["U"][:, rs].T @ Y
        return out

    def _new_last_site(self, blocks, f):
        sgn = self.sgn
        out = {}
        for qn, b in blocks.items():
            if b["nK"] == 0:
                continue
            qt = list(qn); qt[f] -= 1; qt = tuple(qt)
            bt = blocks.get(qt)
            if bt is None or bt["nK"] == 0:
                continue
            UK = b["U"][:, :b["nK"]]
            Y = np.zeros((bt["U"].shape[0], b["nK"]))
            hit = False
            for k, (q, s) in enumerate(b["segs"]):
                if not (s >> f) & 1:
                    continue
                k2 = bt["index"].get((q, s ^ (1 << f)))
                if k2 is None:
                    continue
                Y[bt["offs"][k2]:bt["offs"][k2 + 1]] = sgn[f, s] * UK[b["offs"][k]:b["offs"][k + 1]]
                hit = True
            if hit:
                out[qn] = bt["U"][:, :bt["nK"]].T @ Y
        return out

    # ------------------------------------------------------------------
    def run(self):
        m, F = self.m, self.m.F
        KB, E0 = self._dot_basis()
        offset = E0
        V = m.V() * self.chain_norm
        self.shells = []
        self.flow = []
        tstart = time.time()
        for n in range(self.nshells):
            if n == 0:
                t_f = np.repeat(V, 2)
                scale = max(self.t_chain[0] * math.sqrt(self.Lam), float(np.max(V)))
            else:
                t_f = np.full(F, self.t_chain[n - 1])
                scale = self.t_chain[n - 1]
            last = (n == self.nshells - 1)
            blocks = self._build(KB, t_f, self.eps_chain[n], scale)
            Nk_n = self.Nkeep_early if n < self.n_early else self.Nkeep
            E0, nk, ntot = self._truncate(blocks, scale, last, Nk_n)
            offset_n = offset + E0          # absolute energy of the ground state of shell n
            # flow diagnostics: lowest rescaled levels
            lev = np.sort(np.concatenate([b["E"] for b in blocks.values()]))[:12] / scale
            self.flow.append(lev)
            # diagonal of number-type operators on the discarded states (FDM thermodynamics)
            ndiag = {name: self._transform_boson(op, blocks, csel="D", diag_only=True)
                     for name, op in KB["nops"].items()}
            sh = {"n": n, "scale": scale, "offset": offset_n, "ndiag": ndiag,
                  "nK": nk, "ntot": ntot,
                  "ED": {qn: b["E"][b["nK"]:] for qn, b in blocks.items() if len(b["E"]) > b["nK"]}}
            if self.store:
                sh["blocks"] = blocks
                sh["ops_prev"] = KB["ops"]
            if not last:
                KBn = {"E": {qn: b["E"][:b["nK"]] for qn, b in blocks.items() if b["nK"] > 0}}
                KBn["L"] = [self._new_last_site(blocks, f) for f in range(F)]
                KBn["ops"] = {name: (f, self._transform_fermion(op, f, blocks))
                              for name, (f, op) in KB["ops"].items()}
                KBn["nops"] = {name: self._transform_boson(op, blocks) for name, op in KB["nops"].items()}
                KB = KBn
                offset = offset_n
            if not self.store:
                for b in blocks.values():
                    b.pop("U", None)
            elif self.disk_store_dir is not None:
                path = os.path.join(self.disk_store_dir, f"shell_{n:03d}.pkl")
                with open(path, "wb") as fh:
                    pickle.dump((sh.pop("blocks"), sh.pop("ops_prev")), fh,
                                protocol=pickle.HIGHEST_PROTOCOL)
                sh["disk_path"] = path
            self.shells.append(sh)
            if self.verbose:
                dmax = max(len(b["E"]) for b in blocks.values())
                print(f"  z={self.z:.3f} n={n:2d} scale={scale:.3e} dim={ntot:6d} kept={nk:5d} "
                      f"blocks={len(blocks):4d} maxblock={dmax:5d} E1={lev[1]:.4f} t={time.time()-tstart:6.1f}s",
                      flush=True)
        return self

    # ------------------------------------------------------------------
    def fdm_weights(self):
        """w_n and per-state Boltzmann factors of the discarded states."""
        beta = 1.0 / self.T
        S = self.S
        last = len(self.shells) - 1
        Eref = self.shells[-1]["offset"]
        logw = np.full(len(self.shells), -np.inf)
        for sh in self.shells:
            if not sh["ED"]:
                continue
            Eall = np.concatenate(list(sh["ED"].values())) + sh["offset"] - Eref
            sh["logZD"] = logsumexp(-beta * Eall)
            logw[sh["n"]] = (last - sh["n"]) * math.log(S) + sh["logZD"]
        logw -= logsumexp(logw[np.isfinite(logw)])
        self.logw = logw
        for sh in self.shells:
            w = math.exp(logw[sh["n"]]) if np.isfinite(logw[sh["n"]]) else 0.0
            sh["w"] = w
            sh["rhoD"] = {}
            if w > 1e-300:
                for qn, ED in sh["ED"].items():
                    sh["rhoD"][qn] = w * np.exp(-beta * (ED + sh["offset"] - Eref) - sh["logZD"])
        return logw

    def expectation(self):
        """FDM thermal expectation values of the number-type operators."""
        if not hasattr(self, "logw"):
            self.fdm_weights()
        out = defaultdict(float)
        for sh in self.shells:
            for qn, rho in sh["rhoD"].items():
                for name, d in sh["ndiag"].items():
                    if qn in d:
                        out[name] += float(np.dot(rho, d[qn]))
        return dict(out)

    # ------------------------------------------------------------------
    def fdm_spectral(self, bins):
        """Backward FDM pass. Returns raw binned spectral weights for each
        spectral operator:  {name: (hist_pos, hist_neg, total_weight)}."""
        assert self.store
        if not hasattr(self, "logw"):
            self.fdm_weights()
        names = []
        for i in self.spec_dots:
            names += [("d", i), ("O", i)]
        hist = {nm: bins.empty_hist() for nm in names}
        rhoKK = None
        first_trunc = min([sh["n"] for sh in self.shells if sh["ED"]])
        for n in range(len(self.shells) - 1, first_trunc - 1, -1):
            sh = self.shells[n]
            if "disk_path" in sh:
                with open(sh["disk_path"], "rb") as fh:
                    blocks, ops_prev = pickle.load(fh)
            else:
                blocks, ops_prev = sh["blocks"], sh["ops_prev"]
            rhoD = sh["rhoD"]
            if rhoKK is None:
                rhoKK = {}
            use_DD = len(rhoD) > 0
            for i in self.spec_dots:
                f, dprev = ops_prev[("d", i)]
                _, Oprev = ops_prev[("O", i)]
                parts = [("D", "K"), ("K", "D")] + ([("D", "D")] if use_DD else [])
                for (rsel, csel) in parts:
                    dm = self._transform_fermion(dprev, f, blocks, rsel, csel)
                    Om = self._transform_fermion(Oprev, f, blocks, rsel, csel)
                    for qs, dmat in dm.items():
                        qr = list(qs); qr[f] -= 1; qr = tuple(qr)
                        br, bs = blocks[qr], blocks[qs]
                        # X = R_r d + d R_s, with R block-diagonal in (K, D):
                        # rows in K_r only see rhoKK[qr], rows in D_r only rhoD[qr], etc.
                        X = np.zeros_like(dmat)
                        if rsel == "K":
                            if qr in rhoKK:
                                X += rhoKK[qr] @ dmat
                        elif qr in rhoD:
                            X += rhoD[qr][:, None] * dmat
                        if csel == "K":
                            if qs in rhoKK:
                                X += dmat @ rhoKK[qs]
                        elif qs in rhoD:
                            X += dmat * rhoD[qs][None, :]
                        Er = br["E"][_sel(br, rsel)]
                        Es = bs["E"][_sel(bs, csel)]
                        om = Es[None, :] - Er[:, None]
                        bins.add(hist[("d", i)], om, dmat * X)
                        if qs in Om:
                            bins.add(hist[("O", i)], om, Om[qs] * X)
            # propagate the reduced density matrix to the kept space of shell n-1
            if n > first_trunc:
                new = {}
                for qn, b in blocks.items():
                    nK = b["nK"]
                    RK = rhoKK.get(qn)
                    RD = rhoD.get(qn)
                    if RK is None and RD is None:
                        continue
                    U = b["U"]
                    for k, (q, s) in enumerate(b["segs"]):
                        sl = slice(b["offs"][k], b["offs"][k + 1])
                        P = 0.0
                        if RK is not None and nK > 0:
                            UK = U[sl, :nK]
                            P = UK @ RK @ UK.T
                        if RD is not None:
                            UD = U[sl, nK:]
                            P = P + (UD * RD[None, :]) @ UD.T
                        if q in new:
                            new[q] += P
                        else:
                            new[q] = np.array(P, copy=True)
                rhoKK = new
            # free memory of this shell
            sh.pop("blocks", None)
            if "disk_path" in sh:
                os.remove(sh.pop("disk_path"))
            del blocks, ops_prev
        return hist


def _sel(b, which):
    if which == "K":
        return slice(0, b["nK"])
    if which == "D":
        return slice(b["nK"], len(b["E"]))
    return slice(0, len(b["E"]))


# ----------------------------------------------------------------------------
# Binning and broadening
# ----------------------------------------------------------------------------
class LogBins:
    def __init__(self, wmin=1e-18, wmax=10.0, per_decade=200,
                 wlin=0.0, nlin=401):
        self.lmin, self.lmax = math.log(wmin), math.log(wmax)
        self.nb = int(round((math.log10(wmax) - math.log10(wmin)) * per_decade))
        self.dl = (self.lmax - self.lmin) / self.nb
        self.centers = np.exp(self.lmin + (np.arange(self.nb) + 0.5) * self.dl)
        self.wlin = float(wlin)
        self.linear_centers = (np.linspace(-self.wlin, self.wlin, nlin)
                               if self.wlin > 0 else np.array([]))

    def empty_hist(self):
        h = [np.zeros(self.nb), np.zeros(self.nb), 0.0]
        if self.wlin > 0:
            h.append(np.zeros(len(self.linear_centers)))
        return h

    def add(self, h, om, W):
        om = om.ravel(); W = W.ravel()
        h[2] += float(W.sum())
        if self.wlin > 0:
            low = np.abs(om) < self.wlin
            if np.any(low):
                step = self.linear_centers[1] - self.linear_centers[0]
                idx = np.rint((om[low] + self.wlin) / step).astype(np.int64)
                np.clip(idx, 0, len(self.linear_centers) - 1, out=idx)
                h[3] += np.bincount(idx, weights=W[low],
                                    minlength=len(self.linear_centers))
            om, W = om[~low], W[~low]
            if len(om) == 0:
                return
        pos = om > 0
        for sign, mask in ((0, pos), (1, ~pos)):
            if not mask.any():
                continue
            a = np.abs(om[mask])
            idx = np.floor((np.log(np.maximum(a, 1e-300)) - self.lmin) / self.dl).astype(np.int64)
            np.clip(idx, 0, self.nb - 1, out=idx)
            h[sign] += np.bincount(idx, weights=W[mask], minlength=self.nb)


def log_gauss_broaden(hist_pos, hist_neg, centers, w_out, b=0.5):
    """Bulla's log-Gaussian kernel, normalised in w."""
    A = np.zeros_like(w_out)
    for h, sgn in ((hist_pos, 1.0), (hist_neg, -1.0)):
        nz = np.nonzero(h)[0]
        if len(nz) == 0:
            continue
        wc, hw = centers[nz], h[nz]
        mask = (np.sign(w_out) == sgn) & (w_out != 0)
        x = np.abs(w_out[mask])[:, None]
        K = np.exp(-b * b / 4.0) / (b * wc[None, :] * math.sqrt(math.pi)) * \
            np.exp(-(np.log(x / wc[None, :]) / b) ** 2)
        A[mask] += K @ hw
    return A


def mixed_broaden(hist, centers, w_out, b, linear_centers=None, linear_eta=None):
    """Log broadening outside the thermal window; linear Gaussian within it.

    Every line is assigned to exactly one kernel. Both kernels integrate to one
    on the real axis. The linear grid has a center at zero, so zero-energy
    transitions are treated symmetrically instead of entering a signed log bin.
    """
    A = log_gauss_broaden(hist[0], hist[1], centers, w_out, b)
    if len(hist) > 3:
        if linear_centers is None or linear_eta is None or linear_eta <= 0:
            raise ValueError("linear histogram requires centers and positive width")
        nz = np.flatnonzero(hist[3])
        if len(nz):
            y = (w_out[:, None] - linear_centers[nz][None, :]) / linear_eta
            A += np.exp(-y * y) @ hist[3][nz] / (math.sqrt(math.pi) * linear_eta)
    return A


def kramers_kronig(w, A):
    """Re G(w) = P int A(x)/(w-x) dx for piecewise-linear A on grid w."""
    x = w
    dx = np.diff(x)
    s = np.diff(A) / dx                              # slopes
    ReG = np.zeros_like(w)
    for i, wi in enumerate(w):
        alpha = A[:-1] + s * (wi - x[:-1])           # linear extrapolation to wi
        d1 = wi - x[1:]
        d0 = wi - x[:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            lg = np.log(np.abs(d1)) - np.log(np.abs(d0))
        # the two segments adjacent to wi: combine their singular logs
        with np.errstate(invalid="ignore"):
            term = -alpha * lg
        if 0 < i:
            k = i - 1   # segment [x_{i-1}, x_i]: d1 = 0
            term[k] = -A[i] * (0.0 - np.log(abs(d0[k])))
        if i < len(w) - 1:
            k = i       # segment [x_i, x_{i+1}]: d0 = 0
            term[k] = -A[i] * (np.log(abs(d1[k])) - 0.0)
        ReG[i] = np.sum(term) - np.sum(s * dx)
    return ReG


def default_grid(wmin=1e-12, wmax=0.999, per_decade=60):
    pos = np.logspace(math.log10(wmin), math.log10(wmax), int((math.log10(wmax) - math.log10(wmin)) * per_decade) + 1)
    return np.concatenate([-pos[::-1], pos])


# ----------------------------------------------------------------------------
# One z-run and z-averaged driver
# ----------------------------------------------------------------------------
def run_single_z(model, z, Lam=3.0, nshells=45, Ecut=8.0, Nkeep=3000, T=1e-10,
                 bins_per_decade=200, verbose=False, spectral=True, Nkeep_early=None, n_early=0,
                 linear_factor=4.0, linear_eta_factor=1.0, linear_nbins=401,
                 disk_backed=False):
    """One NRG run at a single discretisation offset z.  With spectral=True the
    eigenvectors of every shell are kept for the backward FDM pass (memory
    heavy); with spectral=False only thermodynamics is returned (light)."""
    t0 = time.time()
    tmp = tempfile.TemporaryDirectory(prefix="nrg2_shells_") if disk_backed and spectral else None
    try:
        run = NRGRun(model, Lam=Lam, z=z, nshells=nshells, Ecut=Ecut, Nkeep=Nkeep, T=T,
                     store=spectral, verbose=verbose, Nkeep_early=Nkeep_early, n_early=n_early,
                     disk_store_dir=tmp.name if tmp is not None else None).run()
        run.fdm_weights()
        expv = run.expectation()
        out = {"z": z, "expect": expv,
               "flow": np.array([np.pad(f[:8], (0, max(0, 8 - len(f[:8]))), constant_values=np.nan) for f in run.flow]),
               "nkept": np.array([sh["nK"] for sh in run.shells]),
               "scale": np.array([sh["scale"] for sh in run.shells])}
        if spectral:
            bins = LogBins(per_decade=bins_per_decade, wlin=linear_factor * T,
                           nlin=linear_nbins)
            hist = run.fdm_spectral(bins)
            out["bins"] = bins.centers
            out["lin_bins"] = bins.linear_centers
            out["lin_eta"] = linear_eta_factor * T
            out["hist"] = {k: tuple(v) for k, v in hist.items()}
        out["time"] = time.time() - t0
        return out
    finally:
        if tmp is not None:
            tmp.cleanup()


def physical_memory_gb():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        return 16.0


def resolve_n_jobs(n_jobs, n_tasks, mem_per_job_gb=None, mem_fraction=0.75):
    """joblib-style n_jobs (-2 = all cores but one), capped by the number of
    tasks and, if mem_per_job_gb is given, by the physical memory."""
    ncpu = os.cpu_count() or 1
    nj = ncpu + 1 + n_jobs if n_jobs < 0 else n_jobs
    nj = max(1, min(nj, n_tasks))
    if mem_per_job_gb:
        nj = max(1, min(nj, int(mem_fraction * physical_memory_gb() / mem_per_job_gb)))
    return nj


def broaden_and_sigma(model, res, w_out, b, sigma_trick=True):
    """z-average already-computed runs: broaden, Kramers-Kronig, self-energy trick."""
    Nz = len(res)
    out = {"w": w_out}
    for i in range(model.M):
        AG = np.zeros_like(w_out); AF = np.zeros_like(w_out)
        tot = []
        for r in res:
            hd = r["hist"][("d", i)]
            AG += mixed_broaden(hd, r["bins"], w_out, b,
                                r.get("lin_bins"), r.get("lin_eta")) / Nz
            tot.append(hd[2])
            hO = r["hist"][("O", i)]
            AF += mixed_broaden(hO, r["bins"], w_out, b,
                                r.get("lin_bins"), r.get("lin_eta")) / Nz
        out[("A_raw", i)] = AG
        out[("A_F", i)] = AF
        out[("sumrule", i)] = np.array(tot)
        if sigma_trick:
            G = kramers_kronig(w_out, AG) - 1j * np.pi * AG
            Fc = kramers_kronig(w_out, AF) - 1j * np.pi * AF
            Sig = Fc / G
            Dl, Gam = model.D, model.Gamma[i]
            with np.errstate(divide="ignore", invalid="ignore"):
                ReHyb = Gam / (2 * np.pi) * np.log(np.abs((Dl + w_out) / (Dl - w_out)))
            Hyb = ReHyb - 1j * Gam / 2 * (np.abs(w_out) < Dl)
            G_unclipped = 1.0 / (w_out - model.eps[i] - Hyb - Sig)
            Sig_c = Sig.real + 1j * np.minimum(Sig.imag, 0.0)   # causality clip of broadening noise
            Gimp = 1.0 / (w_out - model.eps[i] - Hyb - Sig_c)
            out[("Sigma", i)] = Sig
            out[("A_unclipped", i)] = -G_unclipped.imag / np.pi
            out[("sigma_clipped", i)] = Sig.imag > 0
            out[("A_sigma", i)] = -Gimp.imag / np.pi
    ex = defaultdict(list)
    for r in res:
        for k, v in r["expect"].items():
            ex[k].append(v)
    out["expect"] = {k: float(np.mean(v)) for k, v in ex.items()}
    out["expect_z"] = {k: np.array(v) for k, v in ex.items()}
    return out


def z_average(model, Nz=8, b=0.4, n_jobs=-2, w_out=None, mem_per_job_gb=None,
              sigma_trick=True, verbose=False, **nrg_kwargs):
    """z-averaged FDM spectral functions of all dots (parallel over z).
    nrg_kwargs: Lam, nshells, Ecut, Nkeep, Nkeep_early, n_early, T.
    Returns dict with grid w (units of D), A_raw[i], A_sigma[i] (per spin),
    Sigma[i] (complex), occupations and diagnostics."""
    from joblib import Parallel, delayed
    zs = [(k + 1) / Nz for k in range(Nz)]
    nj = resolve_n_jobs(n_jobs, Nz, mem_per_job_gb)
    if verbose:
        print(f"z_average: {Nz} z-values on {nj} parallel workers", flush=True)
    res = Parallel(n_jobs=nj, verbose=0)(
        delayed(run_single_z)(model, z, spectral=True, verbose=False, **nrg_kwargs) for z in zs)
    if w_out is None:
        w_out = default_grid()
    out = broaden_and_sigma(model, res, w_out, b, sigma_trick)
    out.update({"zs": zs, "runs": res, "b": b, "nrg_kwargs": nrg_kwargs})
    return out
