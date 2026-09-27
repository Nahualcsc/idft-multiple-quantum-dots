from itertools import product
import numpy as np


class ImpurityN(object):
    """
    N-orbital Anderson impurity (spinful) with:
      - onsite energies: (v_i + Eimp_i)
      - onsite Hubbard: U_i n_{i↑} n_{i↓}
      - density-density interorbital: sum_{i<j} Uij_{ij} n_i n_j
      - Hund (Ising-like as in your 2-orbital code): -sum_{i<j} JH_{ij} (n_{i↑} n_{j↑} + n_{i↓} n_{j↓})
      - hopping (spin-conserving): sum_{σ} sum_{i≠j} t_{ij} c†_{iσ} c_{jσ}  (you typically provide Hermitian t)

    Mode ordering in the Fock basis:
      (1↑, 1↓, 2↑, 2↓, ..., N↑, N↓)
    """

    def __init__(self, U_list, Uij, JH, Eimp, t):
        self.U = np.asarray(U_list, dtype=float)
        self.N = int(self.U.size)

        self.Eimp = np.asarray(Eimp, dtype=float)
        if self.Eimp.size != self.N:
            raise ValueError("Eimp must have length N.")

        # Uij and JH can be scalars (same for all pairs) or NxN matrices
        self.Uij = Uij
        self.JH = JH

        # hopping matrix NxN (real or complex), assumed spin-conserving
        self.t = np.asarray(t, dtype=float)
        if self.t.shape != (self.N, self.N):
            raise ValueError("t must be an (N,N) matrix.")

        self.v = np.zeros(self.N, dtype=float)

    def set_gates(self, v_list):
        v = np.asarray(v_list, dtype=float)
        if v.size != self.N:
            raise ValueError("v_list must have length N.")
        self.v = v

    def _pair_matrix_value(self, X, i, j):
        # X can be scalar or (N,N)
        if np.isscalar(X):
            return float(X)
        X = np.asarray(X, dtype=float)
        if X.shape != (self.N, self.N):
            raise ValueError("Pair interaction must be scalar or (N,N).")
        return float(X[i, j])

    def energy_from_state(self, st):
        """
        st: tuple/list/array of length 2N with entries 0/1 in mode order:
            (n1u,n1d,n2u,n2d,...)
        """
        st = np.asarray(st, dtype=int)
        if st.size != 2 * self.N:
            raise ValueError("State must have length 2N.")

        # onsite terms
        E = 0.0
        n_up = st[0::2]
        n_dn = st[1::2]
        n = n_up + n_dn

        vtot = self.v + self.Eimp
        E += float(np.dot(vtot, n))
        E += float(np.dot(self.U, n_up * n_dn))

        # interorbital density-density and Hund (Ising-like)
        for i in range(self.N):
            for j in range(i + 1, self.N):
                Uij_ = self._pair_matrix_value(self.Uij, i, j)
                JH_  = self._pair_matrix_value(self.JH,  i, j)
                E += Uij_ * n[i] * n[j]
                E -= JH_ * (n_up[i] * n_up[j] + n_dn[i] * n_dn[j])

        return E


# ---------- fermionic utilities (occupation basis) ----------

def ann_sign(st, mode):
    """
    Fermionic sign for applying annihilation operator c_mode on a Fock basis state st
    with the given mode ordering:
      c_mode |n0 n1 ...> = (-1)^(sum_{k<mode} nk) |... n_mode-1 ...>
    """
    return -1.0 if (sum(st[:mode]) % 2 == 1) else 1.0


def build_basis_2N(N):
    # list of tuples of length 2N, each entry 0/1
    return list(product([0, 1], repeat=2 * N))


def build_annihilation_matrices(basis):
    """
    Build annihilation matrices C[mode] in the given basis.
    Returns:
      C_list: list of length n_modes, each a (dim,dim) float64 matrix
      state_indices: dict mapping state tuple -> index
    """
    dim = len(basis)
    n_modes = len(basis[0])
    state_indices = {st: i for i, st in enumerate(basis)}

    C_list = []
    for mode in range(n_modes):
        C = np.zeros((dim, dim), dtype=np.float64)
        for i, st in enumerate(basis):
            if st[mode] == 1:
                stj = list(st)
                stj[mode] = 0
                j = state_indices[tuple(stj)]
                C[j, i] = ann_sign(st, mode)
        C_list.append(C)

    return C_list, state_indices


def build_hamiltonian(imp, basis, state_indices):
    """
    H = H_diag + sum_{σ} sum_{i≠j} t_{ij} c†_{iσ} c_{jσ}
    with correct fermionic signs handled by explicit state updates in occupation basis.
    """
    N = imp.N
    dim = len(basis)
    H = np.zeros((dim, dim), dtype=np.float64)

    for idx_i, st in enumerate(basis):
        # diagonal
        H[idx_i, idx_i] = float(imp.energy_from_state(st))

        # hopping (spin-conserving). Mode indices: (i↑)->2*i, (i↓)->2*i+1
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                tij = float(imp.t[i, j])
                if tij == 0.0:
                    continue

                # up spin hop j -> i : c†_{i↑} c_{j↑}
                mi = 2 * i
                mj = 2 * j
                if st[mi] == 0 and st[mj] == 1:
                    stj = list(st)
                    # apply annihilation at mj then creation at mi; total sign from moving fermions in occupation basis
                    # Using state-update with occupation ordering, the sign is (-1)^(#occupied between modes).
                    # Here we implement it by the same convention as in your 2-orbital code:
                    # perform the move and include the canonical fermionic sign for the operator string.
                    sign = ann_sign(st, mj) * ann_sign(tuple(stj[:mj] + [0] + stj[mj+1:]), mi)
                    stj[mj] = 0
                    stj[mi] = 1
                    idx_j = state_indices[tuple(stj)]
                    H[idx_i, idx_j] += sign * tij

                # down spin hop j -> i : c†_{i↓} c_{j↓}
                mi = 2 * i + 1
                mj = 2 * j + 1
                if st[mi] == 0 and st[mj] == 1:
                    stj = list(st)
                    sign = ann_sign(st, mj) * ann_sign(tuple(stj[:mj] + [0] + stj[mj+1:]), mi)
                    stj[mj] = 0
                    stj[mi] = 1
                    idx_j = state_indices[tuple(stj)]
                    H[idx_i, idx_j] += sign * tij

    # Numerical safety: symmetrise (your “smart fix” step)
    H = 0.5 * (H + H.T)
    return H


# ---------- GCE observables ----------

def comp_dens_N(imp, beta):
    """
    Returns <n_i> for i=0..N-1 in the grand-canonical ensemble of the isolated impurity.
    """
    beta = float(beta)
    basis = build_basis_2N(imp.N)
    dim = len(basis)
    state_indices = {st: i for i, st in enumerate(basis)}

    H = build_hamiltonian(imp, basis, state_indices)
    E, V = np.linalg.eigh(H)

    # weights (canonical for isolated impurity; no mu here)
    x = -beta * E
    x -= np.max(x)
    W = np.exp(x)
    Z = np.sum(W)
    W /= Z

    # number operators in occupation basis
    n_avg = np.zeros(imp.N, dtype=float)
    for m in range(dim):
        psi = V[:, m]
        wm = W[m]
        # expectation over basis components
        probs = np.abs(psi) ** 2
        for idx_b, p in enumerate(probs):
            st = basis[idx_b]
            for i in range(imp.N):
                n_i = st[2 * i] + st[2 * i + 1]
                n_avg[i] += wm * n_i * p

    return n_avg.tolist()


import numpy as np

def compute_coherence_N(imp, beta, i=0, j=1, average_over_spin=True):
    """
    Computes the one-body coherence between orbitals i and j in the GCE of the isolated impurity.

    Observable:
      O = sum_sigma c†_{iσ} c_{jσ}
    Returns:
      <O>/2  if average_over_spin=True  (i.e. ( <c†_{i↑}c_{j↑}> + <c†_{i↓}c_{j↓}> ) / 2 )
      <O>    otherwise

    Notes:
    - This implementation is robust: it constructs O from the annihilation matrices you already build,
      hence fermionic signs are handled consistently by construction.
    - It is complex-safe (works also if later you use complex hoppings / phases), provided your
      build_hamiltonian returns a Hermitian matrix and uses complex dtype when needed.
    """
    beta = float(beta)
    i = int(i); j = int(j)
    if i == j:
        raise ValueError("Choose i != j.")
    if i < 0 or j < 0 or i >= imp.N or j >= imp.N:
        raise ValueError("i,j out of range.")

    # Basis and operators
    basis = build_basis_2N(imp.N)
    state_indices = {st: k for k, st in enumerate(basis)}

    # Hamiltonian (your current build_hamiltonian is fine as long as it is correct/Hermitian)
    H = build_hamiltonian(imp, basis, state_indices).astype(np.complex128)

    # Annihilation matrices (fermionic signs handled internally here)
    C_list, _ = build_annihilation_matrices(basis)

    # Mode indices: (i↑)->2*i, (i↓)->2*i+1
    mi_up, mj_up = 2 * i, 2 * j
    mi_dn, mj_dn = 2 * i + 1, 2 * j + 1

    c_iup = C_list[mi_up].astype(np.complex128)
    c_jup = C_list[mj_up].astype(np.complex128)
    c_idn = C_list[mi_dn].astype(np.complex128)
    c_jdn = C_list[mj_dn].astype(np.complex128)

    # Build O = c†_{i↑} c_{j↑} + c†_{i↓} c_{j↓}
    O = (c_iup.conj().T @ c_jup) + (c_idn.conj().T @ c_jdn)

    # Diagonalize and thermal weights (grand-canonical with mu=0 for isolated impurity in your convention)
    E, V = np.linalg.eigh(H)  # V columns are eigenvectors
    x = -beta * E
    x -= np.max(x)           # stabilize
    W = np.exp(x)
    Z = np.sum(W)
    W /= Z

    # Expectation value: sum_m W_m <m|O|m>
    # Use conjugate transpose, complex-safe.
    expval = 0.0 + 0.0j
    for m in range(V.shape[1]):
        psi = V[:, m]
        expval += W[m] * (psi.conj().T @ (O @ psi))

    expval = expval / 2.0 if average_over_spin else expval

    # For time-reversal symmetric real Hamiltonians this should be real; keep robust return.
    # If tiny imaginary part remains from numerics, drop it.
    if abs(expval.imag) < 1e-12:
        return float(expval.real)
    return expval


def compute_spectral_N(imp, beta, w_grid, gamma, mu=0.0, which=0, spin_resolved=False):
    """
    Grand-canonical many-body spectral function for orbital `which` (0..N-1),
    broadened by a Lorentzian of width gamma (same conventions as your code).

    Returns:
      spin_resolved=False: A(ω) = A_{↑}(ω) + A_{↓}(ω)
      spin_resolved=True : A_{↑}(ω)

    Basis ordering: (1↑,1↓,2↑,2↓,...).
    """
    w_grid = np.asarray(w_grid, dtype=np.float64)
    beta   = float(beta)
    gamma  = float(gamma)
    mu     = float(mu)
    which  = int(which)

    if gamma <= 0.0:
        raise ValueError("gamma must be > 0.")
    if which < 0 or which >= imp.N:
        raise ValueError("which out of range (0..N-1).")

    basis = build_basis_2N(imp.N)
    dim = len(basis)
    state_indices = {st: i for i, st in enumerate(basis)}

    # annihilation operators for the selected orbital
    mode_up = 2 * which
    mode_dn = 2 * which + 1

    # Build all annihilation matrices (robust and simple; dim grows as 2^(2N))
    C_list, _ = build_annihilation_matrices(basis)
    c_up = C_list[mode_up]
    c_dn = C_list[mode_dn]

    # Hamiltonian
    H = build_hamiltonian(imp, basis, state_indices)

    # Your stabilisation: shift + scale before diagonalisation
    shift = np.trace(H) / dim
    Hs = H - shift * np.eye(dim)
    scale = np.max(np.abs(Hs))
    if scale == 0.0:
        scale = 1.0
    Hn = Hs / scale

    En, V = np.linalg.eigh(Hn)
    E = scale * En + shift

    if (not np.isfinite(E).all()) or (not np.isfinite(V).all()):
        raise FloatingPointError("Non-finite eigenpairs from diagonalisation; check parameters.")

    # Particle number in the eigenstates (as in your code)
    Nvals = np.array([sum(st) for st in basis], dtype=np.float64)
    N_eig = np.rint((np.abs(V) ** 2).T @ Nvals).astype(int)

    # Grand-canonical weights
    x = -beta * (E - mu * N_eig)
    x -= np.max(x)
    W = np.exp(x)
    Z = np.sum(W)
    if (Z == 0.0) or (not np.isfinite(Z)) or (not np.isfinite(W).all()):
        raise FloatingPointError("Partition function under/overflow; adjust beta/mu or energy scales.")
    W /= Z

    def L(x):
        return (gamma / 2.0 / np.pi) / (x * x + (gamma * gamma) / 4.0)

    # Operator matrix elements in eigenbasis
    CV_up = c_up @ V
    CV_dn = c_dn @ V
    C_up_e = V.T @ CV_up
    C_dn_e = V.T @ CV_dn

    A = np.zeros_like(w_grid, dtype=np.float64)
    for m in range(dim):
        wm = W[m]
        dE = E - E[m]

        if spin_resolved:
            add_w = np.abs(C_up_e[m, :]) ** 2
            rem_w = np.abs(C_up_e[:, m]) ** 2
        else:
            add_w = np.abs(C_up_e[m, :]) ** 2 + np.abs(C_dn_e[m, :]) ** 2
            rem_w = np.abs(C_up_e[:, m]) ** 2 + np.abs(C_dn_e[:, m]) ** 2

        A += wm * (
            add_w[None, :] * L(w_grid[:, None] - dE[None, :]) +
            rem_w[None, :] * L(w_grid[:, None] + dE[None, :])
        ).sum(axis=1)

    return A 