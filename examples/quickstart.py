"""Quickstart: Kondo spectral functions of a double quantum dot with i-DFT.

System of Fig. 3(a) of the paper: two dots without interdot hopping, each
coupled to its own reservoir, U1 = 4, U2 = 6, U12 = 2, common gate v = -5,
T = 1e-5 (all in units of gamma).  Runs in a few seconds on a laptop.

    python examples/quickstart.py
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "paper_figures"))
from i_DFT_NQD import i_DFT_M_t0  # noqa: E402

gamma = 0.1                        # substrate broadening; energies below are in units of gamma
U_list = [4 * gamma, 6 * gamma]    # on-site interactions U_l
U12 = 2 * gamma                    # interdot interaction U_lm
v = -5 * gamma                     # common gate voltage v_l = v
T = 1e-5 * gamma

dft = i_DFT_M_t0(
    v_list=[v, v], V=0.0, TL=T, TR=T, U_list=U_list, Uij=U12,
    gammaL=1e-4, gammaR=gamma,     # weak STM-like probe, substrate
    gamma_off=0, t=0.0,            # diagonal coupling matrix, no interdot hopping
    Vxc_approach="Vxc_option1", Kondo_factor=True)
dft.ww, dft.ww2, dft.ww3 = 0.16, 0.16, 0.16   # functional widths w0, w1 used in Fig. 3

# 1) self-consistent KS densities (M coupled equations)
n = dft.densities([1.0, 1.0])
dft.set_n(n)
print("occupations n_l =", np.round(n, 4))

# 2) spectral functions: one independent non-linear equation per frequency
w = 1.5 * np.sinh(np.linspace(-7, 7, 801)) / np.sinh(7)   # dense near omega = 0
A = np.array([dft.A(wi) for wi in w]).T      # shape (M, len(w))

fig, ax = plt.subplots(figsize=(5, 3.2))
for l, A_l in enumerate(A):
    ax.plot(w / gamma, gamma * A_l / 4, label=f"dot {l + 1}")
ax.set_xlabel(r"$\omega/\gamma$")
ax.set_ylabel(r"$\gamma A_l(\omega)/4$")
ax.set_xlim(w[0] / gamma, w[-1] / gamma)
ax.legend(frameon=False)
fig.tight_layout()
out = os.path.join(HERE, "quickstart.png")
fig.savefig(out, dpi=200)
print("saved", out)
