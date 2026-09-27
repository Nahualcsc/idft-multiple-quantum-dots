"""Parameters, paths and diagnostics shared by the M = 2 scripts.

All parameters are read from ../input/dqd_parameters.json; outputs go to
../output.  The NRG engine is nrg_multidot.py in this folder (the copy that
produced the archived spectra).
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SYSTEM = HERE.parent
INPUT = SYSTEM / "input" / "dqd_parameters.json"
OUTPUT = SYSTEM / "output"
sys.path.insert(0, str(HERE))
import nrg_multidot as nm  # noqa: E402

P = json.loads(INPUT.read_text())
GAMMA = P["model"]["gamma_over_D"]  # D = 1; gamma is the full level width
T = P["model"]["T_over_gamma"] * GAMMA
U_LIST = tuple(P["model"]["U_over_gamma"])
U12 = P["model"]["U12_over_gamma"]
V_LINE = P["model"]["v_over_gamma"]


def model(v=V_LINE):
    return nm.MultiDot(eps=[v * GAMMA, v * GAMMA],
                       U=[u * GAMMA for u in U_LIST], Uij=U12 * GAMMA,
                       Gamma=[GAMMA, GAMMA])


def settings(nkeep, nkeep_early, lam=3.0, ecut=10.0, n_early=7, temperature=None):
    temperature = T if temperature is None else float(temperature)
    nshells = int(np.ceil(2 * np.log(0.5 / (1e-3 * temperature)) / np.log(lam))) + 2
    return dict(Lam=float(lam), nshells=nshells, Ecut=float(ecut),
                Nkeep=int(nkeep), Nkeep_early=int(nkeep_early),
                n_early=int(n_early), T=temperature)


def settings_from(block, temperature=None):
    """NRG settings from one block of the parameter file."""
    return settings(block["Nkeep"], block["Nkeep_early"], block["Lambda"],
                    block["Ecut"], block["n_early"], temperature)


def spectrum_metrics(w, A, raw, sigma, n):
    """w in omega/gamma; A and raw in gamma*A_per_spin/2."""
    neg = w < 0
    low = np.abs(w) < 1e-3
    clip = np.imag(sigma) > 1e-10
    return {
        "n_fdm": float(n),
        "n_from_A": float(4 * np.trapezoid(A[neg], w[neg])),
        "integral_A": float(2 * np.trapezoid(A, w)),
        "integral_raw": float(2 * np.trapezoid(raw, w)),
        "raw_max": float(np.max(raw)),
        "raw_max_omega_over_gamma": float(w[np.argmax(raw)]),
        "A_near_zero": float(A[np.argmin(np.abs(w - 3e-4))]),
        "friedel_zero_T": float(np.sin(np.pi * n / 2) ** 2 / np.pi),
        "positive_ImSigma_fraction_below_1e-3_gamma": float(np.mean(clip[low])),
        "positive_ImSigma_max_below_1e-3_gamma": float(np.max(np.imag(sigma[low]))),
    }
