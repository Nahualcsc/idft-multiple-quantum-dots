# Frozen engine for Figure 4 (reproduces the published Zenodo data of 2026-05-13).
# Identical to iDFT_Transmission.py except for three lines that were changed there on 2026-08-26:
#   wlambdavHxc = 0.08 (not 0.16), Iloc = I/M (not I), T_KS = -sum(Im G) (not trace(G Gamma G^+)).
# Do not edit; use iDFT_Transmission.py for new work.
import numpy as np
from scipy.special import psi
from scipy.optimize import root
from scipy.linalg import eig
import scipy.integrate as integrate

class i_DFT_M_t0:
    def __init__(self, v_list, TL, TR, U_list, Uij, gammaL, gammaR, t,Vxc_approach, Kondo_factor):
        self.Vxc_approach= Vxc_approach
        self.Kondo_factor = Kondo_factor
        self.v = np.asarray(v_list)
        self.M = len(self.v)
        self.T = TL
        self.T_tip = TR
        self.U = np.asarray(U_list)
        self.Uij = Uij
        self.gamma = gammaR
        self.ww = 0.16
        self.ww2 = 3*self.ww
        #self.lambda_SMS = 08
        #self.lambda_NL = 0.16#3
        self.wlambdavHxc = 0.08
        self.t = t
        self.Gamma = self.gamma * np.ones((self.M, self.M))
        #self.Gamma = np.array([[self.gamma,0],[0,self.gamma]])

    # ---- Hxc potential ----
    def _SIAM(self, U, n):
        if U==0.: return 0
        else:   return U * 0.5 * (1.0 - (2.0 / np.pi) * np.arctan(U * (1.0 - n) / (self.wlambdavHxc * self.gamma)))
        #Gamma=0.5*self.gamma
        #alpha = U/(U+5.68*Gamma)
        #GammaoverU = Gamma/U
        #sigma = 0.811*GammaoverU - 0.390*pow(GammaoverU,2) - 0.168*pow(GammaoverU,3)    
        #return U*n*0.5 + alpha*0.5*U*(1 - n - (2./np.pi)*np.arctan((1-n)/sigma))

    def fermi(self, x, beta):
        if x < 0: return 1.0 / (1.0 + np.exp(beta*x))
        elif x > 0:  return np.exp(-beta*x) / (1.0 + np.exp(-beta*x))
        else: return 0.5

    def v_Hxc(self, n_list):
        n_list = np.asarray(n_list)
        N = np.sum(n_list)
        cim = sum(self._SIAM(self.Uij, N - k) for k in range(2 * self.M - 1))
        return np.array([cim + self._SIAM(self.U[i] - self.Uij, n_list[i])
                         for i in range(self.M)])

    # ---- Xc bias ----
    def Vxc_i(self, n_list, I):
        n = np.asarray(n_list, dtype=float)
        M, N = len(n), np.sum(n)

        if self.Vxc_approach == "Vxc_option1":

            Vxc_CIM = 0.0
            bet_eff = 10.0  # smoothness around I = 0
            for K in range(1, 2*M):
                ap = 4.0 / (2*M - K + 1.0)
                am = 4.0 / (K + 1.0)
                Dp = N - K
                # Smooth version of: alpha = ap if I >= 0 else am
                alpha_smooth = ap * self.fermi(-I, bet_eff) + am * self.fermi(I, bet_eff)
                Dm = N - K + I/4.0 * alpha_smooth
                Vxc_CIM += (self.Uij / np.pi) * (
                    np.arctan(Dp * self.Uij / (2.0 * self.ww * self.gamma))
                    - np.arctan(Dm * self.Uij / (2.0 * self.ww * self.gamma)) )

            aCIM = 1.0 - (2.0 / np.pi) * np.arctan(1/self.ww2*(I * (self.Uij) / (4* self.gamma))**2 ) if self.Kondo_factor else 0.0
            #aCIM = 1.0 - (4.0 / np.pi**2) * np.arctan(1/(self.lambda_SMS*self.gamma)*I * (self.U[0])) **2  if self.Kondo_factor else 0.0
            #aCIM = 1.0 - (2.0 / np.pi) * np.arctan(
            #        1/self.wKondo * (I * (2/3*self.U[0]) / (4.0  * self.gamma))**2 )
            Vxc_CIM = (1.0 - aCIM)*Vxc_CIM

            Iloc, Vxc_SIAM = I/M, 0.0
            S = [1,1]
        
            for ni, Uloc, M in zip(n, self.U, S):
                SIAM0 = (Uloc-self.Uij) * 0.5 * (1.0 - (2.0 / np.pi) * np.arctan((Uloc-self.Uij) * (1.0 - ni) / (self.ww * self.gamma)))
                SIAMI = (Uloc-self.Uij) * 0.5 * (1.0 - (2.0 / np.pi) * np.arctan((Uloc-self.Uij) * (1.0 - (ni + Iloc/2)) / (self.ww * self.gamma)))
                Vxc_SIAMi = SIAM0- SIAMI
                aS = 1.0  - (2.0/np.pi)*np.arctan(1/self.ww2*(Iloc*(Uloc - self.Uij)/(4.0* self.gamma))**2) if self.Kondo_factor else 0.0
                Vxc_SIAM += (1.0 - aS) * Vxc_SIAMi

            return  Vxc_CIM+ Vxc_SIAM/2
 
        #elif self.Vxc_approach == "Vxc_option2":
            #aS = 1.0  - (2.0/np.pi)*np.arctan(1/self.lambda_NL*(I*(self.U[0])/(4.0* self.gamma))**2) if self.Kondo_factor else 0.0
            #return (1.0 - aS) * (self.v_Hxc( n_list)[0]- self.v_Hxc( n_list+I/2)[0])
    

    def dVxc_dI(self, n_list, I, dI=1e-6):
        return (self.Vxc_i(n_list, I + dI) - self.Vxc_i(n_list, I - dI)) / (2.0 * dI)

    # ---- KS Hamiltonian ----
    def _HKS(self, vs):
        H = np.diag(vs.astype(complex)) - 0.5j * self.Gamma
        for i in range(self.M - 1):
            H[i, i + 1] += self.t
            H[i + 1, i] += self.t
        return H

    # ---- Spectral sums ----
    def _Phi(self, mu, T, vs, site=None):
        """ Digamma spectral sum.
        site = int  ->  residues of G_{ii}  (for densities)
        site = None ->  residues of gamma * sum_ij G_{ij}  (for current)
        """
        w, vl, vr = eig(self._HKS(vs), left=True, right=True)
        #denom = np.einsum("ij,ij->j", vl.conj(), vr)
        denom = np.array([vl[:, j].conj() @ vr[:, j] for j in range(self.M)])

        if site is not None: res = (vl[site, :].conj() * vr[site, :]) / denom
        else: res = np.sum(vl.conj(), axis=0) * np.sum(vr, axis=0) / denom
        a = 0.5 + 1j * (w - mu) / (2.0 * np.pi * T)
        return np.sum((res * psi(a)).imag)

    # ---- Densities ----
    def densities(self, n_guess):
        def residuals(n_vec):
            vs = self.v + self.v_Hxc(n_vec)
            return np.array([2.0 * (0.5 - self._Phi(0.0, self.T, vs, site=i) / np.pi)
                             for i in range(self.M)]) - n_vec
        sol = root(residuals, np.asarray(n_guess, dtype=float), method="hybr",
                   options={"xtol": 1e-12, "maxfev": 500000})
        return sol.x

    def set_n(self, n_list):
        self.n = np.asarray(n_list)
        self.vHxc = self.v_Hxc(self.n)
        self.vs = self.v + self.vHxc

    # ---- Transmission / current ----
    def T_KS(self, w):
        Gr = np.linalg.inv((w + 1.e-10j) * np.eye(self.M, dtype=complex) - self._HKS(self.vs))
        #T_num = -2*np.sum(np.imag(Gr))
        T_num = -np.sum(np.imag(Gr))
        return T_num

    def A_KS(self, w):
        Gr = np.linalg.inv((w + 1.e-10j) * np.eye(self.M, dtype=complex) - self._HKS(self.vs))
        #T_num = -2*np.sum(np.imag(Gr))
        Tnum2 = np.trace(Gr @ self.Gamma @ Gr.conj().T)
        return Tnum2.real


    def _I_total(self, V):
        Phi_sub = self._Phi(0.0, self.T, self.vs)
        def eq(I_val):
            Vxc = self.Vxc_i(self.n, I_val)
            Phi_tip = self._Phi(V + Vxc, self.T_tip, self.vs)
            return (Phi_sub - Phi_tip)*2 / np.pi - I_val
        return root(eq, 0.5, method="hybr").x

    def T_iDFT(self, w):
        Iw = self._I_total(w)
        Vxc = self.Vxc_i(self.n, Iw)
        dVxc = self.dVxc_dI(self.n, Iw)
        TS = self.T_KS(w + Vxc)
        return self.gamma*TS / (1.0 - TS*dVxc/np.pi)


    def A_iDFT(self, w):
        Iw = self._I_total(w)
        Vxc = self.Vxc_i(self.n, Iw)
        dVxc = self.dVxc_dI(self.n, Iw)
        AS = self.A_KS(w + Vxc)
        return AS / (1.0 - AS*dVxc/np.pi)
    
    def T_iDFT_sweep(self, w_grid):
        return np.array([self.T_iDFT(w) for w in np.asarray(w_grid)])

    def A_iDFT_sweep(self, w_grid):
        return np.array([self.A_iDFT(w) for w in np.asarray(w_grid)])
