"""Figure 3 with the NRG benchmark: i-DFT (solid) and NRG (dashed / circles).

Same panels as figure3_kondo_2,3,4.py -- gamma*A_i(omega)/4 at v_line for
M = 2, 3, 4 with the occupations n_i(v) as insets -- plus the NRG reference
for M = 2 and M = 3 (spectra and occupations).  M = 4 has no NRG.

i-DFT: recomputed here with the published Fig. 3 functional (Vxc_option1 +
Kondo factor, w0 = w1 = 0.16).  The widths are fixed explicitly below, so the
result does not depend on the default ww in i_DFT_NQD.py.

NRG (full-density-matrix NRG, see ../nrg_benchmark):
  M = 2: Lambda = 3, Nz = 8, Nkeep = 6000 (24000 in the first 7 shells), b = 0.35;
         occupations: Lambda = 3, z = 1, Nkeep = 2500 (8000 in the first 7 shells)
  M = 3: Lambda = 16, Nz = 8, Nkeep = 4000 (48000 in the first 6 shells), b = 0.3;
         occupations: same Lambda, Nkeep and early shells, z = 1.

Writes figure3/figure3_L16.pdf (and .png) and figure3/figure3_L16_data.npz, and
the Zenodo data of Figure 3 (zenodo_repository/data/figure_3: i-DFT and NRG curves,
via zenodo_data_saver_fig3_idft_kondo.save_fig3_dataset).

Usage:  python figure3_kondo_nrg.py
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
from joblib import Parallel, delayed

from i_DFT_NQD import i_DFT_M_t0
from zenodo_data_saver_fig3_idft_kondo import save_fig3_dataset

HERE = os.path.dirname(os.path.abspath(__file__))

# ==========================================================
# NRG input files
# ==========================================================
NRG_DIR = os.path.join(HERE, "..")  # repository root, which holds nrg_benchmark/
NRG_M2_SPECTRA = os.path.join(NRG_DIR, "nrg_benchmark", "M2_double_dot", "output", "dqd_nrg_spectra.npz")
NRG_M2_SWEEP = os.path.join(NRG_DIR, "nrg_benchmark", "M2_double_dot", "output", "dqd_nrg_occupation_sweep.npz")
NRG_M3_SPECTRA = os.path.join(NRG_DIR, "nrg_benchmark", "M3_triple_dot", "output", "tqd_nrg.npz")
NRG_M3_SWEEP = os.path.join(NRG_DIR, "nrg_benchmark", "M3_triple_dot", "output", "tqd_nrg_sweep.npz")

# ==========================================================
# Parameters — plot in units of gamma = gamma_rest
# ==========================================================
gamma_rest = 0.1
gamma_tip  = 0.0001
gamma      = gamma_tip + gamma_rest
g          = gamma_rest

# T = 1e-5 gamma, as in the caption and the NRG.  (The published figure3.pdf was
# computed with T = 1e-4 = 1e-3 gamma; spectra and occupations differ by < 3e-6.)
T_OVER_GAMMA = 1e-5
T      = T_OVER_GAMMA * g
t_hop  = 0.
V      = 0.0
mu     = 0.0

# published Fig. 3 functional
W0, W1 = 0.16, 0.16

# ── N = 2 ──
N2       = 2
U_list_2 = [.4, .6]
Uij_2    = 0.2
v_line_2 = -.5

# ── N = 3 ──
N3       = 3
U_list_3 = [.4, .6, .8]
Uij_3    = 0.2
v_line_3 = -1.

# ── N = 4 ──
N4       = 4
U_list_4 = [.4, .6, .8, 1.0]
Uij_4    = 0.2
v_line_4 = -1.5

# Grids
w_range2  = np.linspace(-1.50, 1.50, 2000)
v_scan    = np.linspace(-2.5, .5, 200)

delta_v   = 0.

outdir = os.path.join(HERE, "figure3")
os.makedirs(outdir, exist_ok=True)


# ==========================================================
# i-DFT
# ==========================================================
def make_objects(N, U_list, Uij, v_common, delta_v):
    idx = np.arange(N, dtype=float)
    shifts = (0.5 * (N - 1) - idx) * float(delta_v)
    v_list = (float(v_common) + shifts).tolist()

    DFT_obj = i_DFT_M_t0(
        v_list=v_list, V=0.0, TL=T, TR=T, U_list=U_list, Uij=Uij, gammaL=gamma_tip,
        gammaR=gamma_rest, gamma_off=0, t=t_hop, Vxc_approach="Vxc_option1", Kondo_factor=True)
    DFT_obj.ww, DFT_obj.ww2, DFT_obj.ww3 = W0, W1, W0
    return v_list, DFT_obj


def spectrum_point(dft, w):
    return np.array(dft.A(w), dtype=float)


def compute_all(N, U_list, Uij, v_line):
    """Return i-DFT densities along v_scan and the line spectra at v_line."""
    n_DFT = np.zeros((len(v_scan), N))
    n_guess = np.ones(N)

    for iv, v in enumerate(tqdm(v_scan, desc=f"N={N} densities", unit="v")):
        _, dft = make_objects(N, U_list, Uij, v, delta_v)
        nD = np.array(dft.densities(n_guess.tolist()), dtype=float)
        dft.set_n(nD.tolist())
        n_DFT[iv] = nD
        n_guess = nD.copy()

    # --- Line spectra at v_line ---
    iv0 = int(np.argmin(np.abs(v_scan - v_line)))
    _, dft_L = make_objects(N, U_list, Uij, v_line, delta_v)
    nD_L = np.array(dft_L.densities(n_DFT[iv0].tolist()), dtype=float)
    dft_L.set_n(nD_L.tolist())

    A_DFT_cols = Parallel(n_jobs=-2, prefer="processes")(
        delayed(spectrum_point)(dft_L, w) for w in tqdm(w_range2, desc=f"N={N} line", unit="w"))
    A_DFT_line = np.stack(A_DFT_cols, axis=1)

    print(f"\n=== N={N}: n(v_line) = {np.round(nD_L, 4)}, normalization ===")
    for i in range(N):
        print(f"  Dot {i+1}:  i-DFT = {np.trapezoid(A_DFT_line[i], w_range2):.4f}")
    return n_DFT, A_DFT_line, nD_L


# ==========================================================
# NRG
# ==========================================================
def load_nrg(spectra, sweep, U_list, Uij, v_line):
    """NRG gamma*A_i/4 on |omega| <= 15 gamma and n_i(v); checks the parameters."""
    d = np.load(spectra)
    assert np.allclose(d["U_list"], np.array(U_list) / g) and np.isclose(float(d["U12"]), Uij / g)
    assert np.isclose(float(d["v_line"]), v_line / g), f"{spectra}: v_line = {float(d['v_line'])}"
    w = d["w_over_gamma"]
    keep = (np.abs(w) <= w_range2[-1] / g) & (np.abs(w) > 0.5 * float(d["T_over_gamma"]))
    s = np.load(sweep)
    info = (f"Lambda={float(d['Lam']):g}, Nz={int(d['Nz'])}, Nkeep={int(d['Nkeep'])} "
            f"({int(d['Nkeep_early'])} in the first {int(d['n_early'])} shells), b={float(d['b']):g}, "
            f"T/gamma={float(d['T_over_gamma']):g}")
    print(f"NRG M={len(U_list)}: {info}; n(v_line) = {np.round(d['n_nrg'], 4)}")
    return dict(w=w[keep], A=d["gA4_nrg"][:, keep], n_line=d["n_nrg"],
                v=s["v_sweep"], n=s["n_sweep"], info=info)


# ==========================================================
# Run
# ==========================================================
n_DFT_2, A_DFT_line_2, nL_2 = compute_all(N2, U_list_2, Uij_2, v_line_2)
n_DFT_3, A_DFT_line_3, nL_3 = compute_all(N3, U_list_3, Uij_3, v_line_3)
n_DFT_4, A_DFT_line_4, nL_4 = compute_all(N4, U_list_4, Uij_4, v_line_4)

nrg_2 = load_nrg(NRG_M2_SPECTRA, NRG_M2_SWEEP, U_list_2, Uij_2, v_line_2)
nrg_3 = load_nrg(NRG_M3_SPECTRA, NRG_M3_SWEEP, U_list_3, Uij_3, v_line_3)

np.savez_compressed(
    os.path.join(outdir, "figure3_L16_data.npz"),
    gamma_rest=gamma_rest, gamma_tip=gamma_tip, T_over_gamma=T_OVER_GAMMA, w0=W0, w1=W1,
    w_over_gamma=w_range2 / g, v_over_gamma=v_scan / g,
    idft_gA4_M2=A_DFT_line_2 * g / 4.0, idft_n_M2=n_DFT_2, idft_n_line_M2=nL_2,
    idft_gA4_M3=A_DFT_line_3 * g / 4.0, idft_n_M3=n_DFT_3, idft_n_line_M3=nL_3,
    idft_gA4_M4=A_DFT_line_4 * g / 4.0, idft_n_M4=n_DFT_4, idft_n_line_M4=nL_4,
    nrg_w_M2=nrg_2["w"], nrg_gA4_M2=nrg_2["A"], nrg_v_M2=nrg_2["v"], nrg_n_M2=nrg_2["n"], nrg_info_M2=nrg_2["info"],
    nrg_w_M3=nrg_3["w"], nrg_gA4_M3=nrg_3["A"], nrg_v_M3=nrg_3["v"], nrg_n_M3=nrg_3["n"], nrg_info_M3=nrg_3["info"])

# Zenodo data of Figure 3 (i-DFT for M = 2, 3, 4 and NRG for M = 2, 3)
save_fig3_dataset(
    output_dir=os.path.join(HERE, "zenodo_repository", "data", "figure_3"),
    gamma_rest=gamma_rest, gamma_tip=gamma_tip, gamma=gamma, T=T, t_hop=t_hop, V=V, mu=mu,
    delta_v=delta_v, w_range2=w_range2, v_scan=v_scan,
    N2=N2, U_list_2=U_list_2, Uij_2=Uij_2, v_line_2=v_line_2, n_DFT_2=n_DFT_2, A_DFT_line_2=A_DFT_line_2,
    N3=N3, U_list_3=U_list_3, Uij_3=Uij_3, v_line_3=v_line_3, n_DFT_3=n_DFT_3, A_DFT_line_3=A_DFT_line_3,
    N4=N4, U_list_4=U_list_4, Uij_4=Uij_4, v_line_4=v_line_4, n_DFT_4=n_DFT_4, A_DFT_line_4=A_DFT_line_4,
    extra_metadata={
        "figure": "Figure 3",
        "description": "Local spectral functions of multiple quantum dots without interdot hopping in the Kondo regime: "
                       "i-DFT (M=2,3,4) and NRG (M=2,3).",
        "model": "i-DFT in the STM limit with Kondo correction; NRG for the same model",
        "paper": "Spectral and transmission properties of multiple correlated quantum dots made simple",
        "script": "figure3_kondo_nrg.py",
        "functional": {"w0": W0, "w1": W1, "Vxc_approach": "Vxc_option1", "Kondo_factor": True},
    },
    nrg={"M2": dict(w_red=nrg_2["w"], A_plot=nrg_2["A"], v_red=nrg_2["v"], n=nrg_2["n"], info=nrg_2["info"],
                    source="nrg_benchmark/M2_double_dot/output/dqd_nrg_spectra.npz, dqd_nrg_occupation_sweep.npz"),
         "M3": dict(w_red=nrg_3["w"], A_plot=nrg_3["A"], v_red=nrg_3["v"], n=nrg_3["n"], info=nrg_3["info"],
                    source="nrg_benchmark/M3_triple_dot/output/tqd_nrg.npz, tqd_nrg_sweep.npz")},
)

# ==========================================================
# Style
# ==========================================================
plt.rcParams.update({
    "text.usetex":         True,
    "font.family":         "serif",
    "font.serif":          ["Computer Modern Roman"],
    "axes.labelsize":      11,
    "xtick.labelsize":     10,
    "ytick.labelsize":     10,
    "legend.fontsize":     9,
    "legend.frameon":      False,
    "axes.linewidth":      0.6,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
})


# ==========================================================
# FIGURE: Spectral functions + occupation insets
#   Left: N=2  |  Center: N=3  |  Right: N=4
#   i-DFT: solid lines; NRG: dashed lines (spectra) and circles (insets)
# ==========================================================
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
NRG_LS = dict(ls=(0, (3.5, 1.5)), lw=0.9)


def panel(ax, N, A_DFT_line, n_DFT, v_line, title, tag, nrg=None, style_key=False):
    for i in range(N):
        c = colors[i]
        ax.plot(w_range2 / g, A_DFT_line[i] * g / 4.0, "-", lw=1.0, color=c)
        if nrg is not None:
            ax.plot(nrg["w"], nrg["A"][i], color=c, **NRG_LS)
    ax.set_xlabel(r"$\omega/\gamma$")
    ax.set_xlim(w_range2[0] / g, w_range2[-1] / g)
    ax.set_xticks([-10, -5, 0, 5, 10])   # no labels at +-15, where neighbouring panels meet
    ax.set_ylim(bottom=0)
    ax.set_title(title, pad=4)
    handles = [Line2D([], [], color=colors[i], lw=1.0, label=rf"$A_{{{i+1}}}$") for i in range(N)]
    if style_key:          # line-style key (i-DFT / NRG), shown once
        handles += [Line2D([], [], color="k", lw=1.0, label=r"i-DFT"),
                    Line2D([], [], color="k", marker="o", ms=2.5, mfc="none", mew=0.6,
                           label=r"NRG", **NRG_LS)]
    ax.legend(handles=handles, loc="upper right", fontsize=8, ncols=1, handlelength=1.6)
    ax.text(0.05, 0.95, rf"\textbf{{{tag}}}", fontsize=11,
            transform=ax.transAxes, va="top", ha="left")

    ins = ax.inset_axes([0.65, 0.3, 0.3, 0.4])
    for i in range(N):
        c = colors[i]
        ins.plot(v_scan / g, n_DFT[:, i], "-", lw=0.8, color=c, label=rf"$n_{{{i+1}}}$")
        if nrg is not None:
            ins.plot(nrg["v"], nrg["n"][:, i], "o", ms=1.8, mfc="none", mew=0.5, color=c)
    ins.axvline(v_line / g, color="gray", ls=":", lw=0.5)
    ins.set_xlim(v_scan[0] / g, v_scan[-1] / g)
    ins.set_xlabel(r"$v/\gamma$", fontsize=7)
    ins.set_ylabel(r"$n_i$", fontsize=7)
    ins.tick_params(labelsize=6)
    ins.legend(loc="best", fontsize=6, handlelength=0.6)


fig1, (ax_l, ax_c, ax_r) = plt.subplots(1, 3, figsize=(10.8, 3.5), sharey=True,
                                          gridspec_kw={"wspace": 0.05})
panel(ax_l, N2, A_DFT_line_2, n_DFT_2, v_line_2, r"$M=2$", "(a)", nrg_2, style_key=True)
panel(ax_c, N3, A_DFT_line_3, n_DFT_3, v_line_3, r"$M=3$", "(b)", nrg_3)
panel(ax_r, N4, A_DFT_line_4, n_DFT_4, v_line_4, r"$M=4$", "(c)")
ax_l.set_ylabel(r"$\gamma A_i(\omega)/4$")

for ext in ("pdf", "png"):
    fig1.savefig(os.path.join(outdir, f"figure3_L16.{ext}"), dpi=300, bbox_inches="tight")
print("[OK] Saved: figure3/figure3_L16.pdf, figure3_L16.png, figure3_L16_data.npz")
plt.close(fig1)
