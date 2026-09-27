"""
Auxiliary data-saving utilities for Figure 4 of
"Spectral and transmission properties of multiple correlated quantum dots made simple".

Recommended use:
    1. Import this file in the script that computes Figure 4.
    2. Call save_fig4_dataset(...) immediately after Tmap_idft, densities,
       sum_errs, linecuts, and the digitized NRG data are computed/loaded.
    3. Use load_fig4_dataset(...) in a plotting-only script to regenerate the figure
       without repeating the expensive i-DFT transmission calculation.

The saved dataset contains:
    - One compressed NumPy archive (.npz) with all numerical arrays.
    - CSV files for the transmission map, line cuts, densities, and digitized NRG data.
    - A JSON metadata file with parameters, units, grid definitions, and array descriptions.
    - A README file describing how to load the data.

Array conventions follow the plotting script:
    Tmap_idft[i_de, iw] = T(omega=w_over_U_grid[iw], DeltaE/DeltaU=de_over_DeltaU_grid[i_de])
    densities[i_de, i] = occupation of dot i at DeltaE/DeltaU grid index i_de
    sum_errs[i_de] = |sum_i n_i - 2| for the density solution
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Sequence

import numpy as np


def _to_jsonable(obj: Any) -> Any:
    """Convert NumPy scalars/arrays and Paths into JSON-serializable objects."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")


def _save_map_csv(path: Path, x_grid: np.ndarray, y_grid: np.ndarray, z_map: np.ndarray) -> None:
    """
    Save a 2D map as a long-form CSV table with columns:
        omega_over_U, DeltaE_over_DeltaU, value

    z_map must have shape (len(y_grid), len(x_grid)).
    """
    x_grid = np.asarray(x_grid)
    y_grid = np.asarray(y_grid)
    z_map = np.asarray(z_map)

    expected_shape = (y_grid.size, x_grid.size)
    if z_map.shape != expected_shape:
        raise ValueError(f"Map shape {z_map.shape} is inconsistent with expected {expected_shape}.")

    xx, yy = np.meshgrid(x_grid, y_grid)
    table = np.column_stack([xx.ravel(), yy.ravel(), z_map.ravel()])
    np.savetxt(
        path,
        table,
        delimiter=",",
        header="omega_over_U,DeltaE_over_DeltaU,value",
        comments="",
    )


def _save_linecuts_csv(
    path: Path,
    w_over_U_grid: np.ndarray,
    linecuts: Dict[str, np.ndarray],
) -> None:
    """Save several i-DFT line cuts on the same omega/U grid."""
    w_over_U_grid = np.asarray(w_over_U_grid)
    columns = ["omega_over_U"]
    arrays = [w_over_U_grid]

    for name, arr in linecuts.items():
        arr = np.asarray(arr)
        if arr.shape != w_over_U_grid.shape:
            raise ValueError(f"Linecut {name} has shape {arr.shape}, expected {w_over_U_grid.shape}.")
        columns.append(name)
        arrays.append(arr)

    table = np.column_stack(arrays)
    np.savetxt(path, table, delimiter=",", header=",".join(columns), comments="")


def _save_densities_csv(path: Path, de_over_DeltaU_grid: np.ndarray, densities: np.ndarray, sum_errs: np.ndarray) -> None:
    """Save self-consistent densities and sum-rule errors."""
    de_over_DeltaU_grid = np.asarray(de_over_DeltaU_grid)
    densities = np.asarray(densities)
    sum_errs = np.asarray(sum_errs)

    if densities.shape[0] != de_over_DeltaU_grid.size:
        raise ValueError("densities must have shape (len(de_over_DeltaU_grid), N).")
    if sum_errs.shape != de_over_DeltaU_grid.shape:
        raise ValueError("sum_errs must have shape (len(de_over_DeltaU_grid),).")

    N = densities.shape[1]
    columns = ["DeltaE_over_DeltaU"] + [f"n{i + 1}" for i in range(N)] + ["sum_error_abs_sum_n_minus_2"]
    table = np.column_stack([de_over_DeltaU_grid] + [densities[:, i] for i in range(N)] + [sum_errs])
    np.savetxt(path, table, delimiter=",", header=",".join(columns), comments="")


def _save_xy_csv(path: Path, x: np.ndarray, y: np.ndarray, x_name: str, y_name: str) -> None:
    """Save a two-column CSV table."""
    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} and {y.shape}.")
    table = np.column_stack([x, y])
    np.savetxt(path, table, delimiter=",", header=f"{x_name},{y_name}", comments="")


def save_fig4_dataset(
    *,
    output_dir: str | Path = "zenodo_repository/data/figure_4",
    U: float,
    DeltaU_over_U: float,
    U12_over_U: float,
    Gamma_over_U: float,
    T_over_U: float,
    gammaL_over_Gamma: float,
    Vxc_approach: str,
    Kondo_factor: bool,
    DeltaU: float,
    U12: float,
    Gamma: float,
    T: float,
    gammaL: float,
    gammaR: float,
    w_over_U_grid: np.ndarray,
    de_over_DeltaU_grid: np.ndarray,
    w_grid: np.ndarray,
    De_grid: np.ndarray,
    Tmap_idft: np.ndarray,
    densities: np.ndarray,
    sum_errs: np.ndarray,
    target_cut_1: float,
    target_cut_2: float,
    target_cut_3: float,
    target_cut_4: float,
    idx_cut_1: int,
    idx_cut_2: int,
    idx_cut_3: int,
    idx_cut_4: int,
    linecut_1: np.ndarray,
    linecut_2: np.ndarray,
    linecut_3: np.ndarray | None = None,
    linecut_4: np.ndarray | None = None,
    w_nrg_b: np.ndarray | None = None,
    T_nrg_b: np.ndarray | None = None,
    w_nrg_c: np.ndarray | None = None,
    T_nrg_c: np.ndarray | None = None,
    nrg_b_path: str | None = None,
    nrg_c_path: str | None = None,
    extra_metadata: Dict[str, Any] | None = None,
) -> Path:
    """
    Save all numerical data needed to reproduce Figure 4.

    Parameters
    ----------
    output_dir:
        Directory where the files will be written.
    U, DeltaU_over_U, U12_over_U, Gamma_over_U, T_over_U, gammaL_over_Gamma:
        Dimensionless and reference parameters used to construct the calculation.
    Vxc_approach, Kondo_factor:
        i-DFT functional options used in the calculation.
    DeltaU, U12, Gamma, T, gammaL, gammaR:
        Dimensional parameters derived from U.
    w_over_U_grid:
        Dimensionless frequency grid omega/U used in the plot.
    de_over_DeltaU_grid:
        Dimensionless detuning grid DeltaE/DeltaU used in the plot.
    w_grid:
        Physical frequency grid omega.
    De_grid:
        Physical detuning grid DeltaE.
    Tmap_idft:
        i-DFT transmission map with shape (len(de_over_DeltaU_grid), len(w_over_U_grid)).
    densities:
        Self-consistent occupations with shape (len(de_over_DeltaU_grid), 2).
    sum_errs:
        Density sum-rule errors |n1+n2-2|.
    target_cut_*, idx_cut_*, linecut_*:
        Linecut information used in the plotted comparison.
    w_nrg_b, T_nrg_b, w_nrg_c, T_nrg_c:
        Optional digitized NRG data used for comparison.
    nrg_b_path, nrg_c_path:
        Original filenames for digitized NRG data, if available.
    extra_metadata:
        Optional dictionary where you can add model notes, paper title, commit hash,
        software versions, or journal information.

    Returns
    -------
    Path
        Path to the main .npz file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / "figure4_data.npz"
    metadata_path = output_dir / "metadata_figure4.json"
    readme_path = output_dir / "README_figure4.md"
    csv_dir = output_dir / "csv"
    csv_dir.mkdir(exist_ok=True)

    w_over_U_grid = np.asarray(w_over_U_grid, dtype=float)
    de_over_DeltaU_grid = np.asarray(de_over_DeltaU_grid, dtype=float)
    w_grid = np.asarray(w_grid, dtype=float)
    De_grid = np.asarray(De_grid, dtype=float)
    Tmap_idft = np.asarray(Tmap_idft, dtype=float)
    densities = np.asarray(densities, dtype=float)
    sum_errs = np.asarray(sum_errs, dtype=float)

    expected_map_shape = (de_over_DeltaU_grid.size, w_over_U_grid.size)
    if Tmap_idft.shape != expected_map_shape:
        raise ValueError(f"Tmap_idft has shape {Tmap_idft.shape}, expected {expected_map_shape}.")
    if densities.shape != (de_over_DeltaU_grid.size, 2):
        raise ValueError(f"densities has shape {densities.shape}, expected {(de_over_DeltaU_grid.size, 2)}.")
    if sum_errs.shape != (de_over_DeltaU_grid.size,):
        raise ValueError(f"sum_errs has shape {sum_errs.shape}, expected {(de_over_DeltaU_grid.size,)}.")
    if w_grid.shape != w_over_U_grid.shape:
        raise ValueError("w_grid and w_over_U_grid must have the same shape.")
    if De_grid.shape != de_over_DeltaU_grid.shape:
        raise ValueError("De_grid and de_over_DeltaU_grid must have the same shape.")

    linecut_1 = np.asarray(linecut_1, dtype=float)
    linecut_2 = np.asarray(linecut_2, dtype=float)
    linecut_3_arr = np.full_like(w_over_U_grid, np.nan, dtype=float) if linecut_3 is None else np.asarray(linecut_3, dtype=float)
    linecut_4_arr = np.full_like(w_over_U_grid, np.nan, dtype=float) if linecut_4 is None else np.asarray(linecut_4, dtype=float)
    for name, arr in {
        "linecut_1": linecut_1,
        "linecut_2": linecut_2,
        "linecut_3": linecut_3_arr,
        "linecut_4": linecut_4_arr,
    }.items():
        if arr.shape != w_over_U_grid.shape:
            raise ValueError(f"{name} has shape {arr.shape}, expected {w_over_U_grid.shape}.")

    # Optional NRG arrays: stable schema using empty arrays if not supplied.
    w_nrg_b_arr = np.array([], dtype=float) if w_nrg_b is None else np.asarray(w_nrg_b, dtype=float)
    T_nrg_b_arr = np.array([], dtype=float) if T_nrg_b is None else np.asarray(T_nrg_b, dtype=float)
    w_nrg_c_arr = np.array([], dtype=float) if w_nrg_c is None else np.asarray(w_nrg_c, dtype=float)
    T_nrg_c_arr = np.array([], dtype=float) if T_nrg_c is None else np.asarray(T_nrg_c, dtype=float)

    if w_nrg_b_arr.size or T_nrg_b_arr.size:
        if w_nrg_b_arr.shape != T_nrg_b_arr.shape:
            raise ValueError("w_nrg_b and T_nrg_b must have the same shape.")
    if w_nrg_c_arr.size or T_nrg_c_arr.size:
        if w_nrg_c_arr.shape != T_nrg_c_arr.shape:
            raise ValueError("w_nrg_c and T_nrg_c must have the same shape.")

    actual_cut_1 = float(de_over_DeltaU_grid[idx_cut_1])
    actual_cut_2 = float(de_over_DeltaU_grid[idx_cut_2])
    actual_cut_3 = float(de_over_DeltaU_grid[idx_cut_3])
    actual_cut_4 = float(de_over_DeltaU_grid[idx_cut_4])

    np.savez_compressed(
        npz_path,
        U=U,
        DeltaU_over_U=DeltaU_over_U,
        U12_over_U=U12_over_U,
        Gamma_over_U=Gamma_over_U,
        T_over_U=T_over_U,
        gammaL_over_Gamma=gammaL_over_Gamma,
        Vxc_approach=np.array(Vxc_approach),
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
        actual_cut_1=actual_cut_1,
        actual_cut_2=actual_cut_2,
        actual_cut_3=actual_cut_3,
        actual_cut_4=actual_cut_4,
        linecut_1=linecut_1,
        linecut_2=linecut_2,
        linecut_3=linecut_3_arr,
        linecut_4=linecut_4_arr,
        w_nrg_b=w_nrg_b_arr,
        T_nrg_b=T_nrg_b_arr,
        w_nrg_c=w_nrg_c_arr,
        T_nrg_c=T_nrg_c_arr,
    )

    # CSV exports.
    _save_map_csv(csv_dir / "figure4_transmission_map_iDFT.csv", w_over_U_grid, de_over_DeltaU_grid, Tmap_idft)
    _save_densities_csv(csv_dir / "figure4_densities_and_sum_errors.csv", de_over_DeltaU_grid, densities, sum_errs)

    _save_linecuts_csv(
        csv_dir / "figure4_linecuts_iDFT.csv",
        w_over_U_grid,
        {
            f"T_iDFT_DeltaE_over_DeltaU_{actual_cut_1:.6g}": linecut_1,
            f"T_iDFT_DeltaE_over_DeltaU_{actual_cut_2:.6g}": linecut_2,
            f"T_iDFT_DeltaE_over_DeltaU_{actual_cut_3:.6g}": linecut_3_arr,
            f"T_iDFT_DeltaE_over_DeltaU_{actual_cut_4:.6g}": linecut_4_arr,
        },
    )

    if w_nrg_b_arr.size:
        _save_xy_csv(csv_dir / "figure4_NRG_digitized_panel_b_DeltaE_over_DeltaU_1p04.csv", w_nrg_b_arr, T_nrg_b_arr, "omega_over_U", "T_NRG")
    if w_nrg_c_arr.size:
        _save_xy_csv(csv_dir / "figure4_NRG_digitized_panel_c_DeltaE_over_DeltaU_0p92.csv", w_nrg_c_arr, T_nrg_c_arr, "omega_over_U", "T_NRG")

    finite_T = Tmap_idft[np.isfinite(Tmap_idft)]
    Tmin = float(np.nanmin(finite_T)) if finite_T.size else 0.0
    Tmax = float(np.nanmax(finite_T)) if finite_T.size else 1.0

    metadata = {
        "dataset": "Numerical data for Figure 4",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "format_version": "1.0",
        "paper": {
            "title": "Spectral and transmission properties of multiple correlated quantum dots made simple",
            "figure_description": "Transmission spectral function of a double quantum dot as a function of frequency and level detuning, computed within i-DFT and compared with digitized NRG line cuts.",
        },
        "parameters": {
            "U": U,
            "DeltaU_over_U": DeltaU_over_U,
            "U12_over_U": U12_over_U,
            "Gamma_over_U": Gamma_over_U,
            "T_over_U": T_over_U,
            "gammaL_over_Gamma": gammaL_over_Gamma,
            "DeltaU": DeltaU,
            "U12": U12,
            "Gamma": Gamma,
            "T": T,
            "gammaL": gammaL,
            "gammaR": gammaR,
            "Vxc_approach": Vxc_approach,
            "Kondo_factor": Kondo_factor,
            "rescaling": "The stored Tmap_idft corresponds to (1/2)*iDFT.T_iDFT_sweep(w_grid), as in the plotting script.",
            "notes": "The map uses dimensionless axes omega/U and DeltaE/DeltaU. The physical grids w_grid and De_grid are also stored.",
        },
        "linecuts": {
            "target_cut_1": target_cut_1,
            "target_cut_2": target_cut_2,
            "target_cut_3": target_cut_3,
            "target_cut_4": target_cut_4,
            "idx_cut_1": int(idx_cut_1),
            "idx_cut_2": int(idx_cut_2),
            "idx_cut_3": int(idx_cut_3),
            "idx_cut_4": int(idx_cut_4),
            "actual_cut_1": actual_cut_1,
            "actual_cut_2": actual_cut_2,
            "actual_cut_3": actual_cut_3,
            "actual_cut_4": actual_cut_4,
            "nrg_b_path": nrg_b_path,
            "nrg_c_path": nrg_c_path,
        },
        "units": {
            "w_over_U_grid": "omega/U",
            "de_over_DeltaU_grid": "DeltaE/DeltaU",
            "w_grid": "omega in physical energy units",
            "De_grid": "DeltaE in physical energy units",
            "Tmap_idft": "dimensionless transmission T(omega)",
            "densities": "self-consistent occupations n1,n2",
            "sum_errs": "absolute error |n1+n2-2|",
            "linecut_*": "dimensionless transmission T(omega) along selected DeltaE/DeltaU cuts",
            "w_nrg_*": "digitized NRG omega/U values",
            "T_nrg_*": "digitized NRG transmission values",
        },
        "array_orientation": {
            "Tmap_idft": "Shape (len(de_over_DeltaU_grid), len(w_over_U_grid)); Tmap_idft[i_de, iw] is the transmission at DeltaE/DeltaU index i_de and omega/U index iw.",
            "densities": "Shape (len(de_over_DeltaU_grid), 2); densities[i_de, :] = [n1, n2].",
            "sum_errs": "Shape (len(de_over_DeltaU_grid),).",
            "linecuts": "Each linecut has shape (len(w_over_U_grid),).",
        },
        "plotting_convention": {
            "map_extent": "[w_over_U_grid.min(), w_over_U_grid.max(), de_over_DeltaU_grid.min(), de_over_DeltaU_grid.max()]",
            "map_origin": "lower",
            "map_quantity": "Tmap_idft",
            "color_scale_min": Tmin,
            "color_scale_max": Tmax,
        },
        "array_shapes": {
            "Tmap_idft": list(Tmap_idft.shape),
            "densities": list(densities.shape),
            "sum_errs": list(sum_errs.shape),
            "linecut_1": list(linecut_1.shape),
            "linecut_2": list(linecut_2.shape),
            "linecut_3": list(linecut_3_arr.shape),
            "linecut_4": list(linecut_4_arr.shape),
            "w_nrg_b": list(w_nrg_b_arr.shape),
            "w_nrg_c": list(w_nrg_c_arr.shape),
        },
        "files": {
            "npz": npz_path.name,
            "metadata": metadata_path.name,
            "csv_directory": "csv/",
        },
        "extra_metadata": extra_metadata or {},
    }

    _write_json(metadata_path, metadata)

    extra = extra_metadata or {}
    extra_lines = "".join(f"- `{k} = {v}`\n" for k, v in {**extra.get("functional", {}), **extra.get("temperatures", {})}.items())

    readme = f"""# Numerical data for Figure 4

This directory contains the numerical data used to generate Figure 4 of the paper
"Spectral and transmission properties of multiple correlated quantum dots made simple".

Figure 4 shows the i-DFT transmission spectral function of a double quantum dot as a function of `omega/U`
and `DeltaE/DeltaU`, together with selected line cuts compared against digitized NRG data.

## Main files

- `figure4_data.npz`: compressed NumPy archive containing all arrays.
- `metadata_figure4.json`: parameter values, units, array descriptions, and orientation conventions.
- `csv/`: CSV exports for the map, line cuts, densities, sum errors, and digitized NRG data.

## Loading the NumPy archive

```python
import numpy as np

data = np.load("figure4_data.npz")
w = data["w_over_U_grid"]
de = data["de_over_DeltaU_grid"]
Tmap = data["Tmap_idft"]
```

The transmission map has shape

```python
(len(de_over_DeltaU_grid), len(w_over_U_grid))
```

and can be plotted using

```python
extent = [w.min(), w.max(), de.min(), de.max()]
ax.imshow(Tmap, origin="lower", extent=extent, aspect="auto")
```

The line cuts can be loaded as

```python
linecut_1 = data["linecut_1"]
linecut_2 = data["linecut_2"]
w_nrg_b = data["w_nrg_b"]
T_nrg_b = data["T_nrg_b"]
```

## Model parameters

- `U = {U}`
- `DeltaU/U = {DeltaU_over_U}`
- `U12/U = {U12_over_U}`
- `Gamma/U = {Gamma_over_U}`
- `T/U = {T_over_U}`
- `gammaL/Gamma = {gammaL_over_Gamma}`
- `Vxc_approach = {Vxc_approach}`
- `Kondo_factor = {Kondo_factor}`
{extra_lines}
The stored `Tmap_idft` corresponds to the same rescaled transmission used in the plotting script,
namely `(1/2)*iDFT.T_iDFT_sweep(w_grid)`.

## Figure

`figure4.pdf` is Figure 4 of the paper, drawn from `figure4_data.npz` by `plot_figure4.py`
(numpy and matplotlib):

```bash
python plot_figure4.py
```

The paper shows the same figure as a PNG.
"""
    readme_path.write_text(readme, encoding="utf-8")

    return npz_path


def load_fig4_dataset(path: str | Path = "zenodo_repository/data/figure_4/figure4_data.npz") -> Dict[str, np.ndarray]:
    """Load the saved .npz archive as a plain dictionary."""
    archive = np.load(path, allow_pickle=False)
    return {key: archive[key] for key in archive.files}
