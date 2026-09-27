import numpy as np
from scipy.special import psi
from scipy.optimize import root
from scipy.linalg import eig


class i_DFT_M_t0(object):

    def __init__(self, v_list, V, TL, TR, U_list, Uij,
                 gammaL, gammaR, gamma_off,t,Vxc_approach,Kondo_factor):
        self.v = np.array(v_list, dtype=float)
        self.M = len(self.v)

        self.T = 0.5 * (TL + TR)
        self.V = float(V)

        self.U = np.array(U_list, dtype=float)

        self.Uij = float(Uij)
        self.gammatip = float(gammaL)

        self.gammaR = float(gammaR)
        self.gamma  = self.gammaR

        self.gamma_off = float(gamma_off)
        gamma_pattern="nn"
        self.gamma_pattern = str(gamma_pattern).lower().strip()

        self.ww = 0.16
        self.ww2 = 0.16
        self.ww3 = 0.16

        self.t = t
        self.Hhop = self._build_hopping_matrix(t)

        self.Gamma = self._build_Gamma_matrix()

        self.Kondo_factor = Kondo_factor
        self.Vxc_approach= Vxc_approach

        self.n = None
        self.vHxc = None

    # -------------------------
    # Matrices
    # -------------------------
    def _build_hopping_matrix(self, t):
        M = self.M
        if np.isscalar(t):
            tt = float(t)
            H = np.zeros((M, M), dtype=float)
            for i in range(M - 1):
                H[i, i + 1] = tt
                H[i + 1, i] = tt
            return H
        Tm = np.array(t, dtype=float)
        if Tm.shape != (M, M):
            raise ValueError("If t is a matrix, it must have shape (M,M).")
        return Tm

    def _build_Gamma_matrix(self):
        """
        Substrate broadening matrix Gamma.

        Default old case: Gamma = gamma * I.

        With gamma_off != 0:
          - gamma_pattern="nn": nearest-neighbor off-diagonals (Gamma_{i,i+1}=Gamma_{i+1,i}=gamma_off)
          - gamma_pattern="all": all-to-all off-diagonals (Gamma_{ij}=gamma_off for i!=j)
          - gamma_pattern="pair01": only (0,1) and (1,0)

        For M=2, "nn" and "pair01" coincide.
        """
        M = self.M
        G = self.gamma * np.eye(M, dtype=float)

        if abs(self.gamma_off) < 1e-30:
            return G

        patt = self.gamma_pattern
        if patt == "nn":
            for i in range(M - 1):
                G[i, i + 1] = self.gamma_off
                G[i + 1, i] = self.gamma_off
        elif patt == "all":
            for i in range(M):
                for j in range(M):
                    if i != j:
                        G[i, j] = self.gamma_off
        elif patt == "pair01":
            if M < 2:
                raise ValueError("pair01 requires M>=2.")
            G[0, 1] = self.gamma_off
            G[1, 0] = self.gamma_off
        else:
            raise ValueError("gamma_pattern must be 'nn', 'all', or 'pair01'.")

        return G


    def fermi(self,x,beta):
        fermifunc = 0
        if (x < 0):
            fermifunc = 1/(1 + np.exp(beta*x))
        elif (x > 0):
            fermifunc = np.exp(-beta*x)/(1 + np.exp(-beta*x))
        return fermifunc
    # -------------------------
    # Hxc model (unchanged)
    # -------------------------
    def v_Hxc(self, n_list):
        n_list = np.array(n_list, dtype=float)

        def SIAM(U, n):
            return U * 0.5 * (1.0 - (2.0 / np.pi) * np.arctan(
                U * (1.0 - n) / (self.ww * self.gamma)))

        def CIM(U, N):
            return sum(SIAM(U, N - k) for k in range(2 * self.M - 1))

        N = float(np.sum(n_list))
        cim = CIM(self.Uij, N)

        vhxc = np.zeros(self.M, dtype=float)
        for i in range(self.M):
            vhxc[i] = cim + SIAM(self.U[i] - self.Uij, n_list[i])
        return vhxc
    def v_Hxc_for_Vxc(self, n_list):
        n_list = np.array(n_list, dtype=float)

        def SIAM(U, n):
            return U * 0.5 * (1.0 - (2.0 / np.pi) * np.arctan(
                U * (1.0 - n) / (2*self.ww * self.gamma)))

        def CIM(U, N):
            return sum(SIAM(U, N - k) for k in range(2 * self.M - 1))

        N = float(np.sum(n_list))
        cim = CIM(self.Uij, N)

        vhxc = np.zeros(self.M, dtype=float)
        for i in range(self.M):
            vhxc[i] = cim + SIAM(self.U[i] - self.Uij, n_list[i])
        return vhxc

    def Vxc_i(self, n_list, I, i):
        n_list = np.array(n_list, dtype=float)
        M, N = len(n_list), np.sum(n_list)
        Ui = self.U[i]
        ni = n_list[i]

        if self.Vxc_approach == "Vxc_option1": ## OLD

            def a_Kondo(U, I):
                if self.Kondo_factor: return (1.0 - 2.0 / np.pi * np.arctan((U * I / self.gamma / 4.0) ** 2 / self.ww2))
                else: return 0

            v0 = self.v_Hxc(n_list)[i]
            n_shift = n_list.copy()
            n_shift[i] = n_shift[i] + I / 2.0
            vI = self.v_Hxc(n_shift)[i]
            return (1.0 - a_Kondo(Ui, I)) * (v0 - vI)

        elif self.Vxc_approach == "Vxc_option2": ### NL+SIAM
            def SIAM(U, n):
                return U / np.pi * np.arctan(U*(n-1)/(self.ww * self.gamma))
            Vxc_CIM = 0.0
            bet_eff = 10.0  # smoothness around I = 0
            def rescaling_S2(n1, n2, eps=1e-12):
                den = n1*n1 + n2*n2
                if den < eps:
                    return 2.0
            def rescaling_S(n1, n2, n0=1e-3):
                den = n1*n1 + n2*n2 + n0*n0
                return 1.0 + 2.0*n1*n2 / den

                return (n1 + n2)**2 / (n1**2+n2**2)
            S = rescaling_S(n_list[0],n_list[1]) 
            #print('S = {}'.format(S))
            Ieff = S * I
            for K in range(1, 2*M):
                ap = 4.0 / (2*M - K + 1.0)
                am = 4.0 / (K + 1.0)
                Dp = N - K
                # Smooth version of:
                # alpha = ap if I >= 0 else am
                alpha_smooth = ap * self.fermi(-Ieff, bet_eff) + am * self.fermi(Ieff, bet_eff)
                Dm = N - K + Ieff/4.0 * alpha_smooth
                Vxc_CIM += (self.Uij / np.pi) * (
                    np.arctan(Dp * self.Uij / (2.0 * self.ww3 * self.gamma))
                    - np.arctan(Dm * self.Uij / (2.0 * self.ww3 * self.gamma)) )

            if self.Kondo_factor:
                aCIM = 1.0 - (2.0 / np.pi) * np.arctan(
                    self.ww2 * (I * self.Uij / (4.0 * self.ww2 * self.gamma))**2 )
            else: aCIM = 0.0
            Vxc_CIM *= (1.0 - aCIM)

            VxcSIAM = SIAM(Ui-self.Uij, ni)- SIAM(Ui-self.Uij, ni + I/2)
            aS = 1.0  - (2.0/np.pi)*np.arctan(1/self.ww2*(I*(Ui-self.Uij)/(4.0* self.gamma))**2) if self.Kondo_factor else 0.0
            Vxc_SIAM = (1.0 - aS) * VxcSIAM

            return Vxc_CIM + Vxc_SIAM

    def dVxc_dI(self, n_list, I,i):
        I_values = np.array([I - 1e-6, I, I + 1e-6], dtype=float)
        V_values = np.array([self.Vxc_i(n_list, x,i) for x in I_values], dtype=float)
        return np.gradient(V_values, I_values)[1]

    # -------------------------
    # KS effective Hamiltonian (only change here)
    # -------------------------
    def _Heff(self, vs_vec):
        vs_vec = np.array(vs_vec, dtype=float)
        # Old: - i (gamma/2) I
        # New: - i/2 * Gamma (possibly non-diagonal)
        return np.diag(vs_vec) + self.Hhop - 1j * 0.5 * self.Gamma

    # -------------------------
    # Poles/residues (local, unchanged)
    # -------------------------
    def _poles_residues_local(self, index, vs_vec):
        Heff = self._Heff(vs_vec)
        w, vl, vr = eig(Heff, left=True, right=True)

        denom = np.einsum("ij,ij->j", vl.conj(), vr)     # <L_k|R_k>
        num = vl[index, :].conj() * vr[index, :]         # (L_{k,i})^* (R_{k,i})
        r = num / denom
        return w, r

    def _Phi(self, mu, Tmu, poles, residues):
        a = 0.5 + 1j * ((poles - mu) / (2.0 * np.pi * Tmu))
        return np.sum((residues * psi(a)).imag)

    # -------------------------
    # Densities (unchanged)
    # -------------------------
    def densities(self, n_guess_list):
        n_guess_list = np.array(n_guess_list, dtype=float)
        if len(n_guess_list) != self.M:
            raise ValueError("n_guess_list must have length M.")

        def residuals(n_vec):
            vHxc = self.v_Hxc(n_vec)
            vs = self.v + vHxc
            out = np.zeros(self.M, dtype=float)
            for i in range(self.M):
                poles, res = self._poles_residues_local(i, vs)
                im = self._Phi(0.0, self.T, poles, res)
                out[i] = 2.0 * (0.5 - im / np.pi)
            return (out - n_vec).tolist()

        sol = root(residuals, n_guess_list, method="hybr",
                   options={"xtol": 1e-12, "maxfev": 5000, "factor": 0.1})
        return sol.x

    def set_n(self, n_list):
        n_list = np.array(n_list, dtype=float)
        if len(n_list) != self.M:
            raise ValueError("n_list must have length M.")
        self.n = n_list
        self.vHxc = self.v_Hxc(self.n)

    # -------------------------
    # Probe current (unchanged; reduced current convention)
    # -------------------------
    def _I_local(self, V, index, T_tip=1e-6):
        if self.n is None or self.vHxc is None:
            raise RuntimeError("Call set_n(...) before computing currents.")
        index = int(index)
        if index < 0 or index >= self.M:
            raise ValueError("index out of range.")

        vs = self.v + self.vHxc
        poles, res = self._poles_residues_local(index, vs)
        Phi_sub = self._Phi(0.0, self.T, poles, res)

        Ui = self.U[index]

        def eq(I):
            Ival = float(I)
            Vxc = self.Vxc_i(self.n, Ival,index)
            Vs = V + Vxc
            Phi_tip = self._Phi(Vs, T_tip, poles, res)
            return (2.0 / np.pi) * (Phi_sub - Phi_tip) - Ival

        sol = root(lambda x: np.array([eq(x[0])]), np.array([0.5]), method="hybr")
        return float(sol.x[0])

    def I(self, V, index):
        return self._I_local(V, index)

    # -------------------------
    # KS local spectral (unchanged; via matrix inversion)
    # -------------------------
    def _As_local(self, w, vs_vec, index):
        Heff = self._Heff(vs_vec)
        G = np.linalg.inv((w * np.eye(self.M, dtype=complex)) - Heff)
        return -2*np.imag(G[index, index])

    # -------------------------
    # i-DFT reconstruction (unchanged)
    # -------------------------
    def A(self, w):
        self.gamma_off = float(0)
        self.Gamma = self._build_Gamma_matrix()
        if self.n is None or self.vHxc is None:
            raise RuntimeError("Call set_n(...) before computing A(w).")

        vs = self.v + self.vHxc
        A_out = np.zeros(self.M, dtype=float)

        for i in range(self.M):
            Iw = self.I(w, i)
            Ui = self.U[i]

            Vxc = self.Vxc_i(self.n, Iw,i)
            dV = self.dVxc_dI(self.n, Iw,i)

            As_i = self._As_local(w + Vxc, vs, i)
            A_out[i] = As_i / (1.0 - dV * As_i/np.pi)/np.pi


        return A_out

    # -------------------------
    # Coherences (unchanged, optional)
    # -------------------------
    def _poles_residues_ij(self, i, j, vs_vec):
        Heff = self._Heff(vs_vec)
        w, vl, vr = eig(Heff, left=True, right=True)

        denom = np.einsum("ij,ij->j", vl.conj(), vr)
        num = vl[i, :].conj() * vr[j, :]
        r_ij = num / denom
        return w, r_ij

    def _Phi_ij(self, mu, Tmu, poles, residues_ij):
        a = 0.5 + 1j * ((poles - mu) / (2.0 * np.pi * Tmu))
        return np.sum((residues_ij * psi(a)).imag)

    def coherence_KS(self, i=0, j=1, mu=0.0, average_over_spin=True):
        if self.n is None or self.vHxc is None:
            raise RuntimeError("Call set_n(...) before computing coherences.")
        i = int(i)
        j = int(j)
        if i < 0 or j < 0 or i >= self.M or j >= self.M:
            raise ValueError("i,j out of range.")

        vs = self.v + self.vHxc
        poles, res_ij = self._poles_residues_ij(i, j, vs)
        Phi_ij = self._Phi_ij(mu, self.T, poles, res_ij)

        alpha = (0.5 if i == j else 0.0) - Phi_ij / np.pi
        return float(alpha) if average_over_spin else float(2.0 * alpha)