"""
Auxiliary data-saving utilities for Figure 1 of
"Spectral and transmission properties of multiple correlated quantum dots made simple".

Recommended use:
    1. Import this file in the script that computes Figure 1.
    2. Call save_fig1_dataset(...) immediately after A_map, A_DFT_line,
       A_GCE_line, n_DFT_vs_v, and n_GCE_vs_v are computed.
    3. Use load_fig1_dataset(...) in a plotting-only script to regenerate the figure
       without repeating the expensive i-DFT/GCE calculations.

The saved dataset contains:
    - One compressed NumPy archive (.npz) with all numerical arrays.
    - CSV files for the spectral maps, line cuts, and occupations.
    - A JSON metadata file with parameters, units, grid definitions, and array descriptions.
    - A README file describing how to load the data.

Array conventions follow the plotting script:
    A_map[i, iv, iw] = A_i(omega=w_range[iw], gate=v_scan[iv])
    A_DFT_line[i, iw] = local i-DFT spectral function at v_line
    A_GCE_line[i, iw] = local GCE Lehmann spectral function at v_line
    n_DFT_vs_v[iv, i] = i-DFT occupation of dot i at gate v_scan[iv]
    n_GCE_vs_v[iv, i] = GCE occupation of dot i at gate v_scan[iv]
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


def _save_map_csv(path: Path, omega_grid_red: np.ndarray, v_grid_red: np.ndarray, z_map: np.ndarray) -> None:
    """
    Save a 2D spectral map as a long-form CSV table with columns:
        omega_over_gamma, v_over_gamma, value

    z_map must have shape (len(v_grid_red), len(omega_grid_red)).
    """
    omega_grid_red = np.asarray(omega_grid_red)
    v_grid_red = np.asarray(v_grid_red)
    z_map = np.asarray(z_map)

    expected_shape = (v_grid_red.size, omega_grid_red.size)
    if z_map.shape != expected_shape:
        raise ValueError(f"Map shape {z_map.shape} is inconsistent with expected {expected_shape}.")

    ww, vv = np.meshgrid(omega_grid_red, v_grid_red)
    table = np.column_stack([ww.ravel(), vv.ravel(), z_map.ravel()])
    np.savetxt(path, table, delimiter=",", header="omega_over_gamma,v_over_gamma,value", comments="")


def _save_line_csv(
    path: Path,
    omega_grid_red: np.ndarray,
    dft_line: np.ndarray,
    gce_line: np.ndarray,
) -> None:
    """
    Save one local spectral line cut as a CSV table.
    """
    omega_grid_red = np.asarray(omega_grid_red)
    dft_line = np.asarray(dft_line)
    gce_line = np.asarray(gce_line)

    if dft_line.shape != omega_grid_red.shape:
        raise ValueError(f"dft_line has shape {dft_line.shape}, expected {omega_grid_red.shape}.")
    if gce_line.shape != omega_grid_red.shape:
        raise ValueError(f"gce_line has shape {gce_line.shape}, expected {omega_grid_red.shape}.")

    table = np.column_stack([omega_grid_red, dft_line, gce_line])
    np.savetxt(
        path,
        table,
        delimiter=",",
        header="omega_over_gamma,gamma_A_iDFT_over_4,gamma_A_GCE_over_4",
        comments="",
    )


def _save_occupations_csv(
    path: Path,
    v_grid_red: np.ndarray,
    n_DFT_vs_v: np.ndarray,
    n_GCE_vs_v: np.ndarray,
) -> None:
    """
    Save occupations as a column-based CSV table.
    """
    v_grid_red = np.asarray(v_grid_red)
    n_DFT_vs_v = np.asarray(n_DFT_vs_v)
    n_GCE_vs_v = np.asarray(n_GCE_vs_v)

    if n_DFT_vs_v.shape != n_GCE_vs_v.shape:
        raise ValueError("n_DFT_vs_v and n_GCE_vs_v must have the same shape.")
    if n_DFT_vs_v.shape[0] != v_grid_red.size:
        raise ValueError("Occupation arrays must have shape (len(v_grid_red), N).")

    N = n_DFT_vs_v.shape[1]
    columns = ["v_over_gamma"]
    values = [v_grid_red]

    for i in range(N):
        columns.append(f"n{i + 1}_iDFT")
        values.append(n_DFT_vs_v[:, i])
    for i in range(N):
        columns.append(f"n{i + 1}_GCE")
        values.append(n_GCE_vs_v[:, i])

    table = np.column_stack(values)
    np.savetxt(path, table, delimiter=",", header=",".join(columns), comments="")


def save_fig1_dataset(
    *,
    output_dir: str | Path = "zenodo_repository/data/figure_1",
    N: int,
    T: float,
    t_hop: float,
    gamma_tip: float,
    gamma_rest: float,
    gamma: float,
    Uij: float,
    U_list: Sequence[float],
    V: float,
    mu: float,
    w_range: np.ndarray,
    w_range2: np.ndarray,
    v_scan: np.ndarray,
    delta_v_map: float,
    delta_v_line: float,
    v_line: float,
    A_map: np.ndarray,
    A_DFT_line: np.ndarray,
    A_GCE_line: np.ndarray,
    n_DFT_vs_v: np.ndarray,
    n_GCE_vs_v: np.ndarray,
    extra_metadata: Dict[str, Any] | None = None,
) -> Path:
    """
    Save all numerical data needed to reproduce Figure 1.

    Parameters
    ----------
    output_dir:
        Directory where the files will be written.
    N:
        Number of quantum dots.
    T, t_hop, gamma_tip, gamma_rest, gamma, Uij, U_list, V, mu:
        Model and numerical parameters used in the calculation.
    w_range:
        Frequency grid used for the spectral maps, in physical units.
    w_range2:
        Frequency grid used for the line cuts, in physical units.
    v_scan:
        Common gate-voltage grid used for the maps and occupations, in physical units.
    delta_v_map, delta_v_line, v_line:
        Detuning and line-cut parameters.
    A_map:
        Local i-DFT spectral maps with shape (N, len(v_scan), len(w_range)).
    A_DFT_line:
        Local i-DFT line spectra with shape (N, len(w_range2)).
    A_GCE_line:
        Local GCE/Lehmann line spectra with shape (N, len(w_range2)).
    n_DFT_vs_v:
        i-DFT occupations with shape (len(v_scan), N).
    n_GCE_vs_v:
        GCE occupations with shape (len(v_scan), N).
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

    npz_path = output_dir / "figure1_data.npz"
    metadata_path = output_dir / "metadata_figure1.json"
    readme_path = output_dir / "README_figure1.md"
    csv_dir = output_dir / "csv"
    csv_dir.mkdir(exist_ok=True)

    w_range = np.asarray(w_range, dtype=float)
    w_range2 = np.asarray(w_range2, dtype=float)
    v_scan = np.asarray(v_scan, dtype=float)
    U_array = np.asarray(U_list, dtype=float)

    A_map = np.asarray(A_map, dtype=float)
    A_DFT_line = np.asarray(A_DFT_line, dtype=float)
    A_GCE_line = np.asarray(A_GCE_line, dtype=float)
    n_DFT_vs_v = np.asarray(n_DFT_vs_v, dtype=float)
    n_GCE_vs_v = np.asarray(n_GCE_vs_v, dtype=float)

    expected_A_map = (N, v_scan.size, w_range.size)
    expected_lines = (N, w_range2.size)
    expected_occ = (v_scan.size, N)

    if A_map.shape != expected_A_map:
        raise ValueError(f"A_map has shape {A_map.shape}, expected {expected_A_map}.")
    if A_DFT_line.shape != expected_lines:
        raise ValueError(f"A_DFT_line has shape {A_DFT_line.shape}, expected {expected_lines}.")
    if A_GCE_line.shape != expected_lines:
        raise ValueError(f"A_GCE_line has shape {A_GCE_line.shape}, expected {expected_lines}.")
    if n_DFT_vs_v.shape != expected_occ:
        raise ValueError(f"n_DFT_vs_v has shape {n_DFT_vs_v.shape}, expected {expected_occ}.")
    if n_GCE_vs_v.shape != expected_occ:
        raise ValueError(f"n_GCE_vs_v has shape {n_GCE_vs_v.shape}, expected {expected_occ}.")

    # Reduced grids and plotted quantities used in the publication figure.
    g = gamma_rest
    w_range_red = w_range / g
    w_range2_red = w_range2 / g
    v_scan_red = v_scan / g
    v_line_red = v_line / g

    A_map_plot = A_map * g / 4.0
    A_DFT_line_plot = A_DFT_line * g / 4.0
    A_GCE_line_plot = A_GCE_line * g / 4.0

    beta = 1.0 / T
    vmax_global = float(np.nanmax(A_map_plot[np.isfinite(A_map_plot)])) if np.any(np.isfinite(A_map_plot)) else 1.0

    np.savez_compressed(
        npz_path,
        N=N,
        T=T,
        beta=beta,
        t_hop=t_hop,
        gamma_tip=gamma_tip,
        gamma_rest=gamma_rest,
        gamma=gamma,
        Uij=Uij,
        U_list=U_array,
        V=V,
        mu=mu,
        delta_v_map=delta_v_map,
        delta_v_line=delta_v_line,
        v_line=v_line,
        v_line_red=v_line_red,
        w_range=w_range,
        w_range2=w_range2,
        v_scan=v_scan,
        w_range_red=w_range_red,
        w_range2_red=w_range2_red,
        v_scan_red=v_scan_red,
        A_map=A_map,
        A_DFT_line=A_DFT_line,
        A_GCE_line=A_GCE_line,
        A_map_plot=A_map_plot,
        A_DFT_line_plot=A_DFT_line_plot,
        A_GCE_line_plot=A_GCE_line_plot,
        n_DFT_vs_v=n_DFT_vs_v,
        n_GCE_vs_v=n_GCE_vs_v,
        vmax_global=vmax_global,
    )

    # CSV exports.
    for i in range(N):
        dot = i + 1
        _save_map_csv(csv_dir / f"figure1_dot{dot}_spectral_map_iDFT.csv", w_range_red, v_scan_red, A_map_plot[i])
        _save_line_csv(csv_dir / f"figure1_dot{dot}_linecut_iDFT_GCE.csv", w_range2_red, A_DFT_line_plot[i], A_GCE_line_plot[i])

    _save_occupations_csv(csv_dir / "figure1_occupations_iDFT_GCE.csv", v_scan_red, n_DFT_vs_v, n_GCE_vs_v)

    metadata = {
        "dataset": "Numerical data for Figure 1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "format_version": "1.0",
        "paper": {
            "title": "Spectral and transmission properties of multiple correlated quantum dots made simple",
            "figure_description": "Local spectral functions of a quadruple quantum dot in the Coulomb blockade regime.",
        },
        "parameters": {
            "N": N,
            "temperature": T,
            "beta": beta,
            "t_hop": t_hop,
            "gamma_tip": gamma_tip,
            "gamma_rest": gamma_rest,
            "gamma_total": gamma,
            "gamma_used_for_plot_units": gamma_rest,
            "Uij": Uij,
            "U_list": list(U_array),
            "V": V,
            "mu": mu,
            "delta_v_map": delta_v_map,
            "delta_v_line": delta_v_line,
            "v_line": v_line,
            "v_line_over_gamma": v_line_red,
            "notes": "All plotted axes are expressed in units of gamma = gamma_rest. Plotted spectral functions are gamma*A_i(omega)/4.",
        },
        "units": {
            "w_range": "omega in physical energy units",
            "w_range2": "omega in physical energy units for line cuts",
            "v_scan": "common gate voltage v in physical energy units",
            "w_range_red": "omega/gamma_rest",
            "w_range2_red": "omega/gamma_rest for line cuts",
            "v_scan_red": "v/gamma_rest",
            "A_map": "raw local i-DFT spectral function A_i(omega)",
            "A_DFT_line": "raw local i-DFT spectral function A_i(omega) at v_line",
            "A_GCE_line": "raw local GCE Lehmann spectral function A_i(omega) at v_line, Lorentzian broadened",
            "A_map_plot": "gamma_rest*A_i(omega)/4 for the publication figure",
            "A_DFT_line_plot": "gamma_rest*A_iDFT(omega)/4 for the publication line cuts",
            "A_GCE_line_plot": "gamma_rest*A_GCE(omega)/4 for the publication line cuts",
            "n_DFT_vs_v": "i-DFT occupations n_i(v)",
            "n_GCE_vs_v": "GCE occupations n_i(v)",
        },
        "array_orientation": {
            "A_map": "Shape (N, len(v_scan), len(w_range)); A_map[i, iv, iw] is dot i, gate index iv, frequency index iw.",
            "A_map_plot": "Same orientation as A_map, but rescaled as gamma_rest*A_map/4.",
            "A_DFT_line": "Shape (N, len(w_range2)); A_DFT_line[i, iw] is the line cut for dot i at v_line.",
            "A_GCE_line": "Shape (N, len(w_range2)); A_GCE_line[i, iw] is the GCE line cut for dot i at v_line.",
            "n_DFT_vs_v": "Shape (len(v_scan), N); n_DFT_vs_v[iv, i] is the occupation of dot i at v_scan[iv].",
            "n_GCE_vs_v": "Shape (len(v_scan), N); n_GCE_vs_v[iv, i] is the GCE occupation of dot i at v_scan[iv].",
        },
        "plotting_convention": {
            "spectral_map_extent": "[w_range_red.min(), w_range_red.max(), v_scan_red.min(), v_scan_red.max()]",
            "spectral_map_origin": "lower",
            "spectral_map_quantity": "A_map_plot[i] = gamma_rest*A_i(omega)/4",
            "linecut_x_axis": "w_range2_red = omega/gamma_rest",
            "linecut_quantities": ["A_DFT_line_plot", "A_GCE_line_plot"],
            "vmax_global": vmax_global,
        },
        "array_shapes": {
            "A_map": list(A_map.shape),
            "A_DFT_line": list(A_DFT_line.shape),
            "A_GCE_line": list(A_GCE_line.shape),
            "n_DFT_vs_v": list(n_DFT_vs_v.shape),
            "n_GCE_vs_v": list(n_GCE_vs_v.shape),
        },
        "files": {
            "npz": npz_path.name,
            "metadata": metadata_path.name,
            "csv_directory": "csv/",
        },
        "extra_metadata": extra_metadata or {},
    }

    _write_json(metadata_path, metadata)

    readme = f"""# Numerical data for Figure 1

This directory contains the numerical data used to generate Figure 1 of the paper
"Spectral and transmission properties of multiple correlated quantum dots made simple".

Figure 1 shows local spectral functions of a quadruple quantum dot in the Coulomb blockade regime.
The top row contains color maps of `gamma*A_i(omega)/4` as a function of `omega/gamma` and `v/gamma`.
The bottom row compares i-DFT and GCE/Lehmann line cuts at the gate voltage indicated in the maps.

## Main files

- `figure1_data.npz`: compressed NumPy archive containing all arrays.
- `metadata_figure1.json`: parameter values, units, array descriptions, and orientation conventions.
- `csv/`: long-form CSV exports for maps, line cuts, and occupations.

## Loading the NumPy archive

```python
import numpy as np

data = np.load("figure1_data.npz")
w = data["w_range_red"]
v = data["v_scan_red"]
A_map_plot = data["A_map_plot"]
A_DFT_line_plot = data["A_DFT_line_plot"]
A_GCE_line_plot = data["A_GCE_line_plot"]
```

The spectral maps have shape

```python
(N, len(v_scan), len(w_range))
```

For dot index `i`, they can be plotted using

```python
extent = [w.min(), w.max(), v.min(), v.max()]
ax.imshow(A_map_plot[i], origin="lower", extent=extent, aspect="auto")
```

The line cuts have shape

```python
(N, len(w_range2))
```

and can be plotted using

```python
w2 = data["w_range2_red"]
ax.plot(w2, A_DFT_line_plot[i])
ax.plot(w2, A_GCE_line_plot[i])
```

## Model parameters

- `N = {N}`
- `T/gamma = {T / g}`
- `U_i/gamma = {list(U_array / g)}`
- `Uij/gamma = {Uij / g}`
- `t/gamma = {t_hop / g}`
- `v_line/gamma = {v_line_red}`

Here `gamma = gamma_rest = {g}` is the energy scale used for the plotted axes.

## Figure

`figure1.pdf` is Figure 1 of the paper, drawn from `figure1_data.npz` by `plot_figure1.py`
(numpy and matplotlib; LaTeX is used for the labels if it is installed):

```bash
python plot_figure1.py
```
"""
    readme_path.write_text(readme, encoding="utf-8")

    return npz_path


def load_fig1_dataset(path: str | Path = "zenodo_repository/data/figure_1/figure1_data.npz") -> Dict[str, np.ndarray]:
    """Load the saved .npz archive as a plain dictionary."""
    archive = np.load(path, allow_pickle=False)
    return {key: archive[key] for key in archive.files}
