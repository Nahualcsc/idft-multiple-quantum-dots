import os
import numpy as np

# np.trapz was removed in NumPy 2.0; np.trapezoid is the new name.
trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
from joblib import Parallel, delayed

from i_DFT_NQD import i_DFT_M_t0
from GCE_NQD import ImpurityN, comp_dens_N, compute_spectral_N

from zenodo_data_saver_fig1_idft_spectra import save_fig1_dataset

# =========================
# Parameters (N=4) — all in units of gamma = gamma_rest
# =========================

N = 4

T          = 0.04
t_hop      = 0.

gamma_tip  = 0.0001
gamma_rest = 0.08
gamma      = gamma_tip + gamma_rest

Uij        = 0.4
U_list     = [0.4, 0.5, 0.6, 0.7]

V  = 0.0
mu = 0.0

w_range  = np.linspace(-1.3, 1.3, 300)
w_range2 = np.linspace(-1.3, 1.3, 300)
v_scan   = np.linspace(-3.8, .9, 300)

delta_v_map  = 0.
delta_v_line = delta_v_map
v_line       = -1

outdir = "./figure1"
os.makedirs(outdir, exist_ok=True)

beta = 1.0 / T

# For plotting: rescale axes to units of gamma
g = gamma_rest

# =========================
# Coherences to compute (all pairs i<j)
# =========================
pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
npairs = len(pairs)

# =========================
# Helpers
# =========================
def detuning_pattern(delta_v: float, N: int):
    idx = np.arange(N, dtype=float)
    return (0.5 * (N - 1) - idx) * float(delta_v)


def make_objects(v_common: float, delta_v: float):
    shifts = detuning_pattern(delta_v, N)
    v_list = (float(v_common) + shifts).tolist()

    DFT_obj = i_DFT_M_t0(
        v_list=v_list, V=0.0, TL=T, TR=T,
        U_list=U_list, Uij=Uij, gammaL=gamma_tip, gammaR=gamma_rest,
        gamma_off=0, t=t_hop, Vxc_approach ="Vxc_option1",Kondo_factor=False)

    tmat = np.zeros((N, N), dtype=np.complex128)
    for i in range(N - 1):
        tmat[i, i + 1] = t_hop
        tmat[i + 1, i] = np.conjugate(t_hop)

    imp = ImpurityN(U_list=U_list, Uij=Uij, JH=0.0, Eimp=[0.0] * N, t=tmat)
    imp.set_gates(v_list)
    return v_list, DFT_obj, imp


# =========================
# Pass 1: DFT densities by continuation in v (SEQUENTIAL)
# =========================
n_DFT_vs_v = np.zeros((len(v_scan), N), dtype=float)
n_GCE_vs_v = np.zeros((len(v_scan), N), dtype=float)

n_guess = np.ones(N, dtype=float)

for iv, v in enumerate(tqdm(v_scan, desc="DFT densities (continuation)", unit="v")):
    _, DFT_obj, imp = make_objects(v, delta_v_map)

    nG = np.array(comp_dens_N(imp, beta), dtype=float)
    n_GCE_vs_v[iv, :] = nG

    nD = np.array(DFT_obj.densities(n_guess.tolist()), dtype=float)
    DFT_obj.set_n(nD.tolist())

    n_DFT_vs_v[iv, :] = nD
    n_guess = nD.copy()



# =========================
# Pass 2: DFT maps A_i(w) (PARALLEL over v)
# =========================
def compute_map_at_v(iv: int):
    v = float(v_scan[iv])
    _, DFT_obj, _ = make_objects(v, delta_v_map)

    nD = n_DFT_vs_v[iv, :]
    DFT_obj.set_n(nD.tolist())

    Avals = np.zeros((N, len(w_range)), dtype=float)
    for iw, w in enumerate(w_range):
        Avals[:, iw] = np.array(DFT_obj.A(w), dtype=float)
    return iv, Avals


results_maps = Parallel(n_jobs=-1)(
    delayed(compute_map_at_v)(iv)
    for iv in tqdm(range(len(v_scan)), desc="DFT maps over v", unit="v")
)

A_map = np.zeros((N, len(v_scan), len(w_range)), dtype=float)
for iv, Avals in results_maps:
    A_map[:, iv, :] = Avals


# =========================
# Line spectra at v_line
# =========================
iv0 = int(np.argmin(np.abs(v_scan - v_line)))
v_listL, DFT_L, impL = make_objects(v_line, delta_v_line)

n_guess_line = n_DFT_vs_v[iv0, :].copy()
nD_L = np.array(DFT_L.densities(n_guess_line.tolist()), dtype=float)
DFT_L.set_n(nD_L.tolist())

A_GCE_line = np.zeros((N, len(w_range2)), dtype=float)
for i in range(N):
    A_GCE_line[i, :] = compute_spectral_N(
        impL, beta, w_range2, gamma, mu=mu, which=i, spin_resolved=False
    )

A_DFT_cols = Parallel(n_jobs=-1, prefer="processes")(
    delayed(lambda ww: np.array(DFT_L.A(ww), dtype=float))(w)
    for w in tqdm(w_range2, desc="DFT line over ω", unit="ω")
)
A_DFT_line = np.stack(A_DFT_cols, axis=1)

save_fig1_dataset(
    output_dir="zenodo_repository/data/figure_1",
    N=N,
    T=T,
    t_hop=t_hop,
    gamma_tip=gamma_tip,
    gamma_rest=gamma_rest,
    gamma=gamma,
    Uij=Uij,
    U_list=U_list,
    V=V,
    mu=mu,
    w_range=w_range,
    w_range2=w_range2,
    v_scan=v_scan,
    delta_v_map=delta_v_map,
    delta_v_line=delta_v_line,
    v_line=v_line,
    A_map=A_map,
    A_DFT_line=A_DFT_line,
    A_GCE_line=A_GCE_line,
    n_DFT_vs_v=n_DFT_vs_v,
    n_GCE_vs_v=n_GCE_vs_v,
    extra_metadata={
        "figure": "Figure 1",
        "description": "Local spectral functions of a quadruple quantum dot in the Coulomb blockade regime.",
        "models": ["i-DFT in the STM limit", "GCE Lehmann benchmark"],
        "paper": "Spectral and transmission properties of multiple correlated quantum dots made simple",
        "script": "main script used for Figure 1"
    }
)

print("\n=== Normalization check (integral of A_i / 4 over omega) ===")
for i in range(N):
    norm_dft = trapezoid(A_DFT_line[i, :], w_range2)
    norm_gce = trapezoid(A_GCE_line[i, :], w_range2)
    print(f"  Dot {i+1}:  i-DFT = {norm_dft:.4f},  GCE = {norm_gce:.4f}")


# ==========================================================
# Colourmap (shared by all figures)
# ==========================================================
inferno = plt.get_cmap("inferno")
inferno_colors = inferno(np.linspace(0, 1, 256))
transition_colors = np.linspace(np.array([1, 1, 1, 1]), inferno_colors[0], 50)
cmap = mcolors.LinearSegmentedColormap.from_list(
    "white_to_inferno", np.vstack((transition_colors, inferno_colors))
)


# ==========================================================
# DIAGNOSTIC FIGURE (3 rows: maps, spectra, occupations)
# ==========================================================
fig, axs = plt.subplots(3, N, figsize=(4.2 * N, 11.0), sharex=False)

for i in range(N):
    im = axs[0, i].imshow(
        A_map[i, :, :] * g / 4.0,
        origin="lower", aspect="auto", cmap=cmap,
        extent=[w_range[0]/g, w_range[-1]/g, v_scan[0]/g, v_scan[-1]/g],
    )
    axs[0, i].set_title(rf"$\gamma A_{{{i+1}}}(\omega)/4$", fontsize=13)
    axs[0, i].set_xlabel(r"$\omega/\gamma$", fontsize=13)
    axs[0, i].set_ylabel(r"$v/\gamma$", fontsize=13)
    axs[0, i].axhline(v_line/g, color="red", linestyle="--", linewidth=2.0, alpha=0.9)
    cb = fig.colorbar(im, ax=axs[0, i], fraction=0.046, pad=0.04)
    cb.set_label(rf"$\gamma A_{{{i+1}}}(\omega)/4$", fontsize=12)

    axs[1, i].plot(w_range2/g, A_DFT_line[i, :] * g / 4.0, "k-", lw=1.8, label="i-DFT", zorder=1)
    axs[1, i].plot(w_range2/g, A_GCE_line[i, :] * g / 4.0, "r--", lw=1.8, label="GCE (Lehmann)", zorder=2)
    axs[1, i].set_xlabel(r"$\omega/\gamma$", fontsize=13)
    axs[1, i].set_ylabel(rf"$\gamma A_{{{i+1}}}(\omega)/4$", fontsize=13)
    axs[1, i].legend(loc="best", fontsize=10)

    axs[2, i].plot(v_scan/g, n_DFT_vs_v[:, i], "k-", lw=1.8, label=rf"$n_{{{i+1}}}$ i-DFT", zorder=1)
    axs[2, i].plot(v_scan/g, n_GCE_vs_v[:, i], "r--", lw=1.8, label=rf"$n_{{{i+1}}}$ GCE", zorder=2)
    axs[2, i].axvline(v_line/g, color="red", linestyle="--", linewidth=2.0, alpha=0.9)
    axs[2, i].set_xlabel(r"$v/\gamma$", fontsize=13)
    axs[2, i].set_ylabel(rf"$n_{{{i+1}}}$", fontsize=13)
    axs[2, i].set_title(rf"Densities vs $v/\gamma$ (dot {i+1})", fontsize=13)
    axs[2, i].legend(loc="best", fontsize=10)

Ustr = "[" + ", ".join(f"{u/g:.3g}" for u in U_list) + "]"
fig.suptitle(
    r"$T/\gamma={:.3g}$, $U/\gamma={}$, $U_{{ij}}/\gamma={:.3g}$, $v_{{\rm line}}/\gamma={:.3g}$".format(
        T/g, Ustr, Uij/g, v_line/g
    ),
    fontsize=13,
)
fig.tight_layout(rect=[0, 0, 1, 0.96])
outfile = os.path.join(outdir, "figure1_with_ni.png")
fig.savefig(outfile, dpi=300)
print(f"[OK] Saved: {outfile}")
plt.close(fig)



# ==========================================================
# PUBLICATION FIGURE (2 rows: maps + line cuts)
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

# Rescaled arrays
w_plot  = w_range  / g
w2_plot = w_range2 / g
v_plot  = v_scan   / g
vl_plot = v_line   / g

A_map_plot = A_map       *g / 4.0
A_DFT_plot = A_DFT_line  *g / 4.0
A_GCE_plot = A_GCE_line  *g / 4.0

vmax_global = A_map_plot.max()

fig, axs = plt.subplots(2, N, figsize=(7.2, 4.4),
                         gridspec_kw={"height_ratios": [1.0, 0.7],
                                      "hspace": 0.35, "wspace": 0.05})

labels = list("abcdefgh")

for i in range(N):
    # ── top row: colour maps ──
    im = axs[0, i].imshow(
        A_map_plot[i], origin="lower", aspect="auto",
        extent=[w_plot[0], w_plot[-1], v_plot[0], v_plot[-1]],
        cmap=cmap, vmin=0, vmax=vmax_global, rasterized=True,
    )
    axs[0, i].axhline(vl_plot, color="#d62728", ls="--", lw=0.8, alpha=0.85)
    axs[0, i].set_title(rf"$ \gamma A_{{{i+1}}}(\omega)/4$", pad=4)
    axs[0, i].set_xlabel(r"$\omega/\gamma$")
    axs[0, i].text(0.2, 1.16, rf"\textbf{{({labels[i]})}}", fontsize=11,
                   transform=axs[0, i].transAxes, va="top", ha="right")
    if i == 0:
        axs[0, i].set_ylabel(r"$v/\gamma$")
    else:
        axs[0, i].set_yticklabels([])

    # ── bottom row: line cuts ──
    axs[1, i].plot(w2_plot, A_DFT_plot[i], "k-", lw=1.0, label="i-DFT")
    axs[1, i].plot(w2_plot, A_GCE_plot[i], color="#d62728", ls="--", lw=1.0, label="GCE")
    axs[1, i].set_xlabel(r"$\omega/\gamma$")
    axs[1, i].set_xlim(w2_plot[0], w2_plot[-1])
    axs[1, i].set_ylim(bottom=0)
    axs[1, i].text(0.2, 1.16, rf"\textbf{{({labels[N + i]})}}", fontsize=11,
                   transform=axs[1, i].transAxes, va="top", ha="right")
    if i == 0:
        axs[1, i].set_ylabel(r"$ \gamma A_i(\omega)/4$")
        axs[1, i].legend(loc="upper left", handlelength=0.8,borderaxespad=0.25)
    else:
        axs[1, i].set_yticklabels([])
    # shared y-scale for bottom row
    ymax_line = max(axs[1, i].get_ylim()[1] for i in range(N))
    for i in range(N):
        axs[1, i].set_ylim(0, 0.175)

# shared colourbar
fig.subplots_adjust(right=0.91)
cax = fig.add_axes([0.92, axs[0, -1].get_position().y0,
                     0.015, axs[0, -1].get_position().y1 - axs[0, -1].get_position().y0])
fig.colorbar(im, cax=cax)

outfile_pub = os.path.join(outdir, "figure1.pdf")
fig.savefig(outfile_pub, dpi=300, bbox_inches="tight")
print(f"[OK] Saved: {outfile_pub}")
plt.close(fig)