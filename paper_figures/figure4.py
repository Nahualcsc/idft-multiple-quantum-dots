import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
from iDFT_Transmission_fig4 import i_DFT_M_t0
import matplotlib.colors as mcolors


from zenodo_data_saver_fig4_idft_transmission import save_fig4_dataset

# ============================================================
# Custom colormap: white -> inferno
# ============================================================
inferno = plt.get_cmap('inferno')
i_col = inferno(np.linspace(0, 1, 256))
white = np.array([1, 1, 1, 1])
tc = np.linspace(white, i_col[0], 50)
w_to_i = mcolors.LinearSegmentedColormap.from_list(
    'white_to_inferno', np.vstack((tc, i_col))
)

out_dir = "./figure4"
os.makedirs(out_dir, exist_ok=True)


Vxc_approach = 'Vxc_option1'
Kondo_factor = True


U = 1.0
 
# ============================================================
# Dimensionless control parameters
# Keep these fixed to test scaling invariance
# ============================================================
DeltaU_over_U = 1/3
U12_over_U = 1-DeltaU_over_U
Gamma_over_U = 1.0 / 15
T_over_U = 3e-6          # temperature of the NRG reference (Kleeorin & Meir 2018, Fig. 1)

# left coupling relative to Gamma
gammaL_over_Gamma = 1e-8

# dimensionless grids
w_over_U_grid = np.linspace(-0.0175, 0.0175, 350)
de_over_DeltaU_grid = np.linspace(0.6, 1.3, 350)
#de_over_DeltaU_grid = np.linspace(-0.1, 0.1, 300)

# linecuts in dimensionless DeltaE / DeltaU
target_cut_1 = 1.04
target_cut_2 = 0.92
target_cut_3 = 1.0
target_cut_4 = 0.8

# ============================================================
# Dimensional parameters derived from U
# ============================================================
DeltaU = DeltaU_over_U * U
U12 = U12_over_U * U
Gamma = Gamma_over_U * U
T = T_over_U * U
gammaL = gammaL_over_Gamma * Gamma
gammaR = Gamma

# physical grids
w_grid = w_over_U_grid * U
De_grid = de_over_DeltaU_grid * DeltaU

# ============================================================
# Helper functions
# ============================================================
def ph_energies(DeltaE, U, U12):
    """
    Particle-hole symmetric onsite energies for a 2-level DQD.
    """
    eps1 = -0.5 * (U + 2.0 * U12 + DeltaE)
    eps2 = eps1 + DeltaE
    return eps1, eps2

def robust_densities(iDFT, DeltaE_over_DU, sum_tol=1e-3):

    guesses = [
        [1.1, 0.9],                                   # default
        [1.0 + 0.3 * DeltaE_over_DU, 1.0 - 0.3 * DeltaE_over_DU],
        [1.5, 0.5],                                   # AFM-phase-like
        [1.9, 0.1],                                   # deep AFM
        [1.001, 0.999],                               # almost symmetric
    ]
    best_n, best_err = None, np.inf
    for g in guesses:
        try:
            n = iDFT.densities(np.asarray(g, dtype=float))
        except Exception:
            continue
        if not np.all(np.isfinite(n)):
            continue
        err = abs(np.sum(n) - 2.0)
        if err < best_err:
            best_err, best_n = err, n
        if err < sum_tol:
            return n, err              # good enough, stop early
    return best_n, best_err


def compute_row(DeltaE, DeltaE_over_DU):
    eps1, eps2 = ph_energies(DeltaE, U, U12)

    # TL = temperature of the dots (substrate), TR = temperature of the STM tip
    iDFT = i_DFT_M_t0(
        v_list=[eps1, eps2],
        TL=T,
        TR=T*1e-6,
        U_list=[U, U],
        Uij=U12,
        gammaL=gammaL,
        gammaR=gammaR,
        t=0.0,Vxc_approach = Vxc_approach, Kondo_factor = Kondo_factor
    )

    n, sum_err = robust_densities(iDFT, DeltaE_over_DU)

    iDFT.set_n(n)
    t_idft = iDFT.T_iDFT_sweep(w_grid)
    t_idft_rescaled = (1 / 2.0) * t_idft
    return t_idft_rescaled, n, sum_err


# ============================================================
# Compute transmission map
# ============================================================
results = Parallel(n_jobs=-1)(
    delayed(compute_row)(De, x)
    for De, x in tqdm(list(zip(De_grid, de_over_DeltaU_grid)), desc="Δε sweep")
)

Tmap_idft = np.stack([r[0] for r in results], axis=0)
Tmap_idft = np.squeeze(Tmap_idft)

densities = np.stack([r[1] for r in results], axis=0)
sum_errs  = np.array([r[2] for r in results])

idx_cut_1 = np.argmin(np.abs(de_over_DeltaU_grid - target_cut_1))
idx_cut_2 = np.argmin(np.abs(de_over_DeltaU_grid - target_cut_2))
idx_cut_3 = np.argmin(np.abs(de_over_DeltaU_grid - target_cut_3))
idx_cut_4 = np.argmin(np.abs(de_over_DeltaU_grid - target_cut_4))

linecut_1 = np.asarray(Tmap_idft[idx_cut_1, :]).squeeze()
linecut_2 = np.asarray(Tmap_idft[idx_cut_2, :]).squeeze()
linecut_3 = np.asarray(Tmap_idft[idx_cut_3, :]).squeeze()
linecut_4 = np.asarray(Tmap_idft[idx_cut_4, :]).squeeze()

# ============================================================
# Load digitized NRG linecuts
# ============================================================
nrg_b_path = "panel_b_digitized.txt"  # Δε/ΔU = 1.04
nrg_c_path = "panel_c_digitized.txt"  # Δε/ΔU = 0.92

nrg_b = np.loadtxt(nrg_b_path)
nrg_c = np.loadtxt(nrg_c_path)

w_nrg_b, T_nrg_b = nrg_b[:, 0], nrg_b[:, 1]
w_nrg_c, T_nrg_c = nrg_c[:, 0], nrg_c[:, 1]


# functional widths, read from the engine
_widths = i_DFT_M_t0(v_list=[0.0, 0.0], TL=T, TR=T*1e-6, U_list=[U, U], Uij=U12, gammaL=gammaL,
                     gammaR=gammaR, t=0.0, Vxc_approach=Vxc_approach, Kondo_factor=Kondo_factor)

save_fig4_dataset(
    output_dir="zenodo_repository/data/figure_4",
    U=U,
    DeltaU_over_U=DeltaU_over_U,
    U12_over_U=U12_over_U,
    Gamma_over_U=Gamma_over_U,
    T_over_U=T_over_U,
    gammaL_over_Gamma=gammaL_over_Gamma,
    Vxc_approach=Vxc_approach,
    Kondo_factor=Kondo_factor,
    DeltaU=DeltaU,
    U12=U12,
    Gamma=Gamma,
    T=T,
    gammaL=gammaL,
    gammaR=gammaR,
    w_over_U_grid=w_over_U_grid,
    de_over_DeltaU_grid=de_over_DeltaU_grid,
    w_grid=w_grid,
    De_grid=De_grid,
    Tmap_idft=Tmap_idft,
    densities=densities,
    sum_errs=sum_errs,
    target_cut_1=target_cut_1,
    target_cut_2=target_cut_2,
    target_cut_3=target_cut_3,
    target_cut_4=target_cut_4,
    idx_cut_1=idx_cut_1,
    idx_cut_2=idx_cut_2,
    idx_cut_3=idx_cut_3,
    idx_cut_4=idx_cut_4,
    linecut_1=linecut_1,
    linecut_2=linecut_2,
    linecut_3=linecut_3,
    linecut_4=linecut_4,
    w_nrg_b=w_nrg_b,
    T_nrg_b=T_nrg_b,
    w_nrg_c=w_nrg_c,
    T_nrg_c=T_nrg_c,
    nrg_b_path=nrg_b_path,
    nrg_c_path=nrg_c_path,
    extra_metadata={
        "figure": "Figure 4",
        "description": "Transmission spectral function of a double quantum dot as a function of frequency and level detuning, computed within i-DFT and compared with digitized NRG line cuts.",
        "model": "i-DFT transmission spectral function in the STM limit",
        "paper": "Spectral and transmission properties of multiple correlated quantum dots made simple",
        "script": "main script used for Figure 4",
        "functional": {"w0_Hxc_potential": round(_widths.wlambdavHxc, 12), "w0_xc_bias": round(_widths.ww, 12),
                       "w1_Kondo_factor": round(_widths.ww2, 12)},
        "temperatures": {"T_dots_over_U": T_over_U, "T_tip_over_U": float(f"{T_over_U * 1e-6:.6g}")},
    }
)

# ============================================================
# Plot
# ============================================================
extent = [
    w_over_U_grid.min(),
    w_over_U_grid.max(),
    de_over_DeltaU_grid.min(),
    de_over_DeltaU_grid.max()
]

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# ------------------------------------------------------------
# Left: i-DFT transmission map
# ------------------------------------------------------------
ax = axes[0]
im = ax.imshow(
    Tmap_idft,
    origin="lower",
    aspect="auto",
    extent=extent,
    interpolation="nearest",
    cmap=w_to_i
)

ax.axhline(
    de_over_DeltaU_grid[idx_cut_1],
    color='blue',
    ls='--',
    lw=1.4,
    label=rf'$\Delta\varepsilon/\Delta U={de_over_DeltaU_grid[idx_cut_1]:.2f}$'
)
ax.axhline(
    de_over_DeltaU_grid[idx_cut_2],
    color='red',
    ls='--',
    lw=1.4,
    label=rf'$\Delta\varepsilon/\Delta U={de_over_DeltaU_grid[idx_cut_2]:.2f}$'
)

ax.set_xlabel(r"$\omega/U$", fontsize=14)
ax.set_ylabel(r"$\Delta\varepsilon / \Delta U$", fontsize=14)
# Only show these x-axis ticks
ax.set_xticks([-0.01, 0.0, 0.01])
ax.set_xticklabels([r"$-0.01$", r"$0$", r"$0.01$"])
# Increase tick-label size on both axes
ax.tick_params(axis="both", labelsize=12)
cbar = fig.colorbar(im, ax=ax, shrink=0.85)
cbar.ax.set_title(r"$T(\omega)$", pad=10, fontsize=14)
cbar.ax.tick_params(labelsize=12)
# ------------------------------------------------------------
# Right: linecuts + NRG comparison
# ------------------------------------------------------------
ax = axes[1]

# i-DFT linecuts
ax.plot(
    w_over_U_grid,
    linecut_1,
    color='blue',
    lw=1.5,
    label=rf'i-DFT, $\Delta\varepsilon/\Delta U={de_over_DeltaU_grid[idx_cut_1]:.2f}$'
)
ax.plot(
    w_over_U_grid,
    linecut_2,
    color='red',
    lw=1.5,
    label=rf'i-DFT, $\Delta\varepsilon/\Delta U={de_over_DeltaU_grid[idx_cut_2]:.2f}$'
)
'''
ax.plot(
    w_over_U_grid,
    linecut_3,
    color='red',
    lw=1.5,
    label=rf'i-DFT, $\Delta\varepsilon/\Delta U={de_over_DeltaU_grid[idx_cut_3]:.2f}$'
)
ax.plot(
    w_over_U_grid,
    linecut_4,
    color='black',
    lw=1.5,
    label=rf'i-DFT, $\Delta\varepsilon/\Delta U={de_over_DeltaU_grid[idx_cut_4]:.2f}$'
)
'''
# NRG digitized data
ax.plot(
    w_nrg_b,
    T_nrg_b,
    color='blue',
    ls='--',
    lw=1.2,
    label=r'NRG, $\Delta\varepsilon/\Delta U=1.04$'
)
ax.plot(
    w_nrg_c,
    T_nrg_c,
    color='red',
    ls='--',
    lw=1.2,
    label=r'NRG, $\Delta\varepsilon/\Delta U=0.92$'
)

ax.set_xlabel(r"$\omega/U$", fontsize=14)
ax.set_ylabel(r"$T(\omega)$", fontsize=14)
#ax.set_title("Linecuts", fontsize=13)
ax.set_xticks([-0.01, 0.0, 0.01])
ax.tick_params(axis="both", labelsize=12)
ax.legend(fontsize=8.5)
ax.set_xlim(w_over_U_grid.min(), w_over_U_grid.max())
# ============================================================
# Save figure
# ============================================================
png_path = os.path.join(out_dir, f"transmission_maps_U_{U:.3f}.png")
fig.savefig(png_path, dpi=200)
plt.close(fig)

print(f"Saved figure: {png_path}")
