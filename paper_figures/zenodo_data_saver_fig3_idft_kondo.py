"""
Auxiliary data-saving utilities for Figure 3 of
"Spectral and transmission properties of multiple correlated quantum dots made simple".

Recommended use:
    1. Import this file in the script that computes Figure 3.
    2. Call save_fig3_dataset(...) immediately after n_DFT_2, A_DFT_line_2,
       n_DFT_3, A_DFT_line_3, n_DFT_4, and A_DFT_line_4 are computed.
    3. Use load_fig3_dataset(...) in a plotting-only script to regenerate the figure
       without repeating the i-DFT calculations.

The saved dataset contains:
    - One compressed NumPy archive (.npz) with all numerical arrays.
    - CSV files for the spectral line cuts and occupation curves for M=2,3,4.
    - A JSON metadata file with parameters, units, grid definitions, and array descriptions.
    - A README file describing how to load the data.

Array conventions:
    A_DFT_line_M[i, iw] = local i-DFT spectral function of dot i at w_range2[iw]
    n_DFT_M[iv, i] = i-DFT occupation of dot i at v_scan[iv]
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


def _save_spectra_csv(path: Path, omega_grid_red: np.ndarray, spectra_plot: np.ndarray) -> None:
    """
    Save all local spectral curves for a given M as a column-based CSV table.

    spectra_plot has shape (M, len(omega_grid_red)) and should correspond to
    gamma*A_i(omega)/4.
    """
    omega_grid_red = np.asarray(omega_grid_red)
    spectra_plot = np.asarray(spectra_plot)

    M = spectra_plot.shape[0]
    expected_shape = (M, omega_grid_red.size)
    if spectra_plot.shape != expected_shape:
        raise ValueError(f"spectra_plot has shape {spectra_plot.shape}, expected {expected_shape}.")

    columns = ["omega_over_gamma"] + [f"gamma_A{i + 1}_over_4" for i in range(M)]
    table = np.column_stack([omega_grid_red] + [spectra_plot[i] for i in range(M)])
    np.savetxt(path, table, delimiter=",", header=",".join(columns), comments="")


def _save_occupations_csv(path: Path, v_grid_red: np.ndarray, occupations: np.ndarray, label: str = "iDFT") -> None:
    """
    Save all occupation curves for a given M as a column-based CSV table.
    """
    v_grid_red = np.asarray(v_grid_red)
    occupations = np.asarray(occupations)

    if occupations.shape[0] != v_grid_red.size:
        raise ValueError("occupations must have shape (len(v_grid_red), M).")

    M = occupations.shape[1]
    columns = ["v_over_gamma"] + [f"n{i + 1}_{label}" for i in range(M)]
    table = np.column_stack([v_grid_red] + [occupations[:, i] for i in range(M)])
    np.savetxt(path, table, delimiter=",", header=",".join(columns), comments="")


def save_fig3_dataset(
    *,
    output_dir: str | Path = "zenodo_repository/data/figure_3",
    gamma_rest: float,
    gamma_tip: float,
    gamma: float,
    T: float,
    t_hop: float,
    V: float,
    mu: float,
    delta_v: float,
    w_range2: np.ndarray,
    v_scan: np.ndarray,
    N2: int,
    U_list_2: Sequence[float],
    Uij_2: float,
    v_line_2: float,
    n_DFT_2: np.ndarray,
    A_DFT_line_2: np.ndarray,
    N3: int,
    U_list_3: Sequence[float],
    Uij_3: float,
    v_line_3: float,
    n_DFT_3: np.ndarray,
    A_DFT_line_3: np.ndarray,
    N4: int,
    U_list_4: Sequence[float],
    Uij_4: float,
    v_line_4: float,
    n_DFT_4: np.ndarray,
    A_DFT_line_4: np.ndarray,
    extra_metadata: Dict[str, Any] | None = None,
    nrg: Dict[str, Dict[str, Any]] | None = None,
) -> Path:
    """
    Save all numerical data needed to reproduce Figure 3.

    Parameters
    ----------
    output_dir:
        Directory where the files will be written.
    gamma_rest, gamma_tip, gamma:
        Lead-coupling parameters. The plotted energy unit is gamma_rest.
    T, t_hop, V, mu, delta_v:
        Model and numerical parameters used in the calculation.
    w_range2:
        Frequency grid used for the spectral line cuts, in physical units.
    v_scan:
        Common gate-voltage grid used for the occupation insets, in physical units.
    N2, U_list_2, Uij_2, v_line_2, n_DFT_2, A_DFT_line_2:
        Data and parameters for the M=2 panel.
    N3, U_list_3, Uij_3, v_line_3, n_DFT_3, A_DFT_line_3:
        Data and parameters for the M=3 panel.
    N4, U_list_4, Uij_4, v_line_4, n_DFT_4, A_DFT_line_4:
        Data and parameters for the M=4 panel.
    extra_metadata:
        Optional dictionary where you can add model notes, paper title, commit hash,
        software versions, or journal information.
    nrg:
        Optional NRG reference curves, {"M2": {...}, "M3": {...}}, each with
        w_red (omega/gamma), A_plot (gamma*A_i/4, shape (M, len(w_red))),
        v_red (v/gamma), n (shape (len(v_red), M)), info (settings string) and
        source (files in the repository).

    Returns
    -------
    Path
        Path to the main .npz file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / "figure3_data.npz"
    metadata_path = output_dir / "metadata_figure3.json"
    readme_path = output_dir / "README_figure3.md"
    csv_dir = output_dir / "csv"
    csv_dir.mkdir(exist_ok=True)

    g = gamma_rest
    beta = 1.0 / T

    w_range2 = np.asarray(w_range2, dtype=float)
    v_scan = np.asarray(v_scan, dtype=float)
    w_range2_red = w_range2 / g
    v_scan_red = v_scan / g

    datasets = {
        "M2": {
            "N": int(N2),
            "U_list": np.asarray(U_list_2, dtype=float),
            "Uij": float(Uij_2),
            "v_line": float(v_line_2),
            "n_DFT": np.asarray(n_DFT_2, dtype=float),
            "A_DFT_line": np.asarray(A_DFT_line_2, dtype=float),
        },
        "M3": {
            "N": int(N3),
            "U_list": np.asarray(U_list_3, dtype=float),
            "Uij": float(Uij_3),
            "v_line": float(v_line_3),
            "n_DFT": np.asarray(n_DFT_3, dtype=float),
            "A_DFT_line": np.asarray(A_DFT_line_3, dtype=float),
        },
        "M4": {
            "N": int(N4),
            "U_list": np.asarray(U_list_4, dtype=float),
            "Uij": float(Uij_4),
            "v_line": float(v_line_4),
            "n_DFT": np.asarray(n_DFT_4, dtype=float),
            "A_DFT_line": np.asarray(A_DFT_line_4, dtype=float),
        },
    }

    for label, d in datasets.items():
        N = d["N"]
        n_DFT = d["n_DFT"]
        A_DFT_line = d["A_DFT_line"]
        if n_DFT.shape != (v_scan.size, N):
            raise ValueError(f"{label}: n_DFT has shape {n_DFT.shape}, expected {(v_scan.size, N)}.")
        if A_DFT_line.shape != (N, w_range2.size):
            raise ValueError(f"{label}: A_DFT_line has shape {A_DFT_line.shape}, expected {(N, w_range2.size)}.")
        if d["U_list"].shape != (N,):
            raise ValueError(f"{label}: U_list has shape {d['U_list'].shape}, expected {(N,)}.")

    # Plotted spectra: gamma*A_i(omega)/4.
    A_DFT_line_2_plot = datasets["M2"]["A_DFT_line"] * g / 4.0
    A_DFT_line_3_plot = datasets["M3"]["A_DFT_line"] * g / 4.0
    A_DFT_line_4_plot = datasets["M4"]["A_DFT_line"] * g / 4.0

    np.savez_compressed(
        npz_path,
        gamma_rest=gamma_rest,
        gamma_tip=gamma_tip,
        gamma=gamma,
        T=T,
        beta=beta,
        t_hop=t_hop,
        V=V,
        mu=mu,
        delta_v=delta_v,
        w_range2=w_range2,
        v_scan=v_scan,
        w_range2_red=w_range2_red,
        v_scan_red=v_scan_red,
        N2=N2,
        U_list_2=datasets["M2"]["U_list"],
        Uij_2=Uij_2,
        v_line_2=v_line_2,
        v_line_2_red=v_line_2 / g,
        n_DFT_2=datasets["M2"]["n_DFT"],
        A_DFT_line_2=datasets["M2"]["A_DFT_line"],
        A_DFT_line_2_plot=A_DFT_line_2_plot,
        N3=N3,
        U_list_3=datasets["M3"]["U_list"],
        Uij_3=Uij_3,
        v_line_3=v_line_3,
        v_line_3_red=v_line_3 / g,
        n_DFT_3=datasets["M3"]["n_DFT"],
        A_DFT_line_3=datasets["M3"]["A_DFT_line"],
        A_DFT_line_3_plot=A_DFT_line_3_plot,
        N4=N4,
        U_list_4=datasets["M4"]["U_list"],
        Uij_4=Uij_4,
        v_line_4=v_line_4,
        v_line_4_red=v_line_4 / g,
        n_DFT_4=datasets["M4"]["n_DFT"],
        A_DFT_line_4=datasets["M4"]["A_DFT_line"],
        A_DFT_line_4_plot=A_DFT_line_4_plot,
        **{f"{key}_{label[1:]}": np.asarray(val, dtype=float)
           for label, d in (nrg or {}).items()
           for key, val in (("w_NRG_red", d["w_red"]), ("A_NRG_line_plot", d["A_plot"]),
                            ("v_NRG_red", d["v_red"]), ("n_NRG", d["n"]))},
    )

    # CSV exports.
    _save_spectra_csv(csv_dir / "figure3_M2_spectral_functions_iDFT.csv", w_range2_red, A_DFT_line_2_plot)
    _save_spectra_csv(csv_dir / "figure3_M3_spectral_functions_iDFT.csv", w_range2_red, A_DFT_line_3_plot)
    _save_spectra_csv(csv_dir / "figure3_M4_spectral_functions_iDFT.csv", w_range2_red, A_DFT_line_4_plot)

    _save_occupations_csv(csv_dir / "figure3_M2_occupations_iDFT.csv", v_scan_red, datasets["M2"]["n_DFT"])
    _save_occupations_csv(csv_dir / "figure3_M3_occupations_iDFT.csv", v_scan_red, datasets["M3"]["n_DFT"])
    _save_occupations_csv(csv_dir / "figure3_M4_occupations_iDFT.csv", v_scan_red, datasets["M4"]["n_DFT"])

    for label, d in (nrg or {}).items():
        _save_spectra_csv(csv_dir / f"figure3_{label}_spectral_functions_NRG.csv", d["w_red"], d["A_plot"])
        _save_occupations_csv(csv_dir / f"figure3_{label}_occupations_NRG.csv", d["v_red"], d["n"], label="NRG")

    metadata = {
        "dataset": "Numerical data for Figure 3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "format_version": "1.0",
        "paper": {
            "title": "Spectral and transmission properties of multiple correlated quantum dots made simple",
            "figure_description": "Local spectral functions of multiple quantum dots without interdot hopping in the Kondo regime.",
        },
        "parameters_global": {
            "gamma_rest": gamma_rest,
            "gamma_tip": gamma_tip,
            "gamma_total": gamma,
            "gamma_used_for_plot_units": gamma_rest,
            "temperature": T,
            "beta": beta,
            "t_hop": t_hop,
            "V": V,
            "mu": mu,
            "delta_v": delta_v,
            "Kondo_factor": True,
            "gamma_off": 0.0,  # figure3_kondo_2,3,4.py uses gamma_off=0 (each dot its own reservoir)
            "notes": "All plotted axes are expressed in units of gamma = gamma_rest. Plotted spectral functions are gamma*A_i(omega)/4. "
                     + ("The figure shows i-DFT (all panels) and NRG (M=2 and M=3) results." if nrg else
                        "This figure contains only i-DFT data, without GCE benchmark curves."),
        },
        "systems": {
            "M2": {
                "N": N2,
                "U_list": list(datasets["M2"]["U_list"]),
                "U_list_over_gamma": list(datasets["M2"]["U_list"] / g),
                "Uij": Uij_2,
                "Uij_over_gamma": Uij_2 / g,
                "v_line": v_line_2,
                "v_line_over_gamma": v_line_2 / g,
            },
            "M3": {
                "N": N3,
                "U_list": list(datasets["M3"]["U_list"]),
                "U_list_over_gamma": list(datasets["M3"]["U_list"] / g),
                "Uij": Uij_3,
                "Uij_over_gamma": Uij_3 / g,
                "v_line": v_line_3,
                "v_line_over_gamma": v_line_3 / g,
            },
            "M4": {
                "N": N4,
                "U_list": list(datasets["M4"]["U_list"]),
                "U_list_over_gamma": list(datasets["M4"]["U_list"] / g),
                "Uij": Uij_4,
                "Uij_over_gamma": Uij_4 / g,
                "v_line": v_line_4,
                "v_line_over_gamma": v_line_4 / g,
            },
        },
        "units": {
            "w_range2": "omega in physical energy units",
            "v_scan": "common gate voltage v in physical energy units",
            "w_range2_red": "omega/gamma_rest",
            "v_scan_red": "v/gamma_rest",
            "A_DFT_line_M": "raw local i-DFT spectral function A_i(omega)",
            "A_DFT_line_M_plot": "gamma_rest*A_i(omega)/4 for the publication figure",
            "n_DFT_M": "i-DFT occupations n_i(v)",
        },
        "array_orientation": {
            "A_DFT_line_2": "Shape (N2, len(w_range2)); A_DFT_line_2[i, iw] is the spectrum for dot i in the M=2 system.",
            "A_DFT_line_3": "Shape (N3, len(w_range2)); A_DFT_line_3[i, iw] is the spectrum for dot i in the M=3 system.",
            "A_DFT_line_4": "Shape (N4, len(w_range2)); A_DFT_line_4[i, iw] is the spectrum for dot i in the M=4 system.",
            "n_DFT_2": "Shape (len(v_scan), N2); n_DFT_2[iv, i] is the occupation of dot i in the M=2 system.",
            "n_DFT_3": "Shape (len(v_scan), N3); n_DFT_3[iv, i] is the occupation of dot i in the M=3 system.",
            "n_DFT_4": "Shape (len(v_scan), N4); n_DFT_4[iv, i] is the occupation of dot i in the M=4 system.",
        },
        "plotting_convention": {
            "spectral_x_axis": "w_range2_red = omega/gamma_rest",
            "spectral_y_quantity": "A_DFT_line_M_plot = gamma_rest*A_i(omega)/4",
            "occupation_x_axis": "v_scan_red = v/gamma_rest",
            "occupation_y_quantity": "n_DFT_M",
        },
        "array_shapes": {
            "A_DFT_line_2": list(datasets["M2"]["A_DFT_line"].shape),
            "A_DFT_line_3": list(datasets["M3"]["A_DFT_line"].shape),
            "A_DFT_line_4": list(datasets["M4"]["A_DFT_line"].shape),
            "n_DFT_2": list(datasets["M2"]["n_DFT"].shape),
            "n_DFT_3": list(datasets["M3"]["n_DFT"].shape),
            "n_DFT_4": list(datasets["M4"]["n_DFT"].shape),
        },
        "files": {
            "npz": npz_path.name,
            "metadata": metadata_path.name,
            "csv_directory": "csv/",
        },
        "extra_metadata": extra_metadata or {},
    }
    if nrg:
        metadata["nrg"] = {label: {"settings": d["info"], "source": d["source"],
                                   "arrays": {f"w_NRG_red_{label[1:]}": "omega/gamma_rest",
                                              f"A_NRG_line_plot_{label[1:]}": "gamma_rest*A_i(omega)/4, shape (M, len(w))",
                                              f"v_NRG_red_{label[1:]}": "v/gamma_rest",
                                              f"n_NRG_{label[1:]}": "NRG occupations, shape (len(v), M)"}}
                           for label, d in nrg.items()}

    _write_json(metadata_path, metadata)

    nrg_section = ""
    if nrg:
        nrg_section = "\n## NRG reference curves (M=2 and M=3)\n\n" + (
            "The dashed lines and circles of Figure 3 are numerical renormalization group (NRG) results, stored as\n"
            "`w_NRG_red_M`, `A_NRG_line_plot_M` (gamma*A_i(omega)/4, shape (M, len(w))), `v_NRG_red_M` and `n_NRG_M`\n"
            "(shape (len(v), M)) for M = 2, 3, and exported to `csv/figure3_M*_NRG.csv`. The NRG code, input parameters\n"
            "and full outputs are in `../../nrg_benchmark/`.\n\n") + "".join(
            f"- M={label[1:]}: {d['info']}; source `{d['source']}`\n" for label, d in nrg.items())

    readme = f"""# Numerical data for Figure 3

This directory contains the numerical data used to generate Figure 3 of the paper
"Spectral and transmission properties of multiple correlated quantum dots made simple".

Figure 3 shows local i-DFT spectral functions of multiple quantum dots without interdot hopping in the Kondo regime.
The three panels correspond to `M=2`, `M=3`, and `M=4`. The insets show the corresponding i-DFT occupations as
functions of the common gate voltage.

## Main files

- `figure3_data.npz`: compressed NumPy archive containing all arrays.
- `metadata_figure3.json`: parameter values, units, array descriptions, and orientation conventions.
- `csv/`: CSV exports for the spectral functions and occupation curves.

## Loading the NumPy archive

```python
import numpy as np

data = np.load("figure3_data.npz")
w = data["w_range2_red"]
v = data["v_scan_red"]
A2 = data["A_DFT_line_2_plot"]
n2 = data["n_DFT_2"]
```

For the `M=2` system, the spectral array has shape

```python
(N2, len(w_range2))
```

and can be plotted using

```python
for i in range(data["N2"]):
    ax.plot(w, A2[i])
```

The occupation array has shape

```python
(len(v_scan), N2)
```

and can be plotted using

```python
for i in range(data["N2"]):
    ax.plot(v, n2[:, i])
```

Analogously, use `A_DFT_line_3_plot`, `n_DFT_3`, `A_DFT_line_4_plot`, and `n_DFT_4`
for the `M=3` and `M=4` panels.

## Model parameters

- `gamma = gamma_rest = {g}`
- `T/gamma = {T / g:g}`
- `t/gamma = {t_hop / g}`
- `Uij/gamma = {Uij_2 / g}, {Uij_3 / g}, {Uij_4 / g}` for `M=2,3,4`
- `v_line/gamma = {v_line_2 / g}, {v_line_3 / g}, {v_line_4 / g}` for `M=2,3,4`

The plotted spectral functions are stored as `gamma*A_i(omega)/4`.
{nrg_section}
## Figure

`figure3.pdf` is Figure 3 of the paper, drawn from `figure3_data.npz` by `plot_figure3.py`
(numpy and matplotlib; LaTeX is used for the labels if it is installed):

```bash
python plot_figure3.py
```
"""
    readme_path.write_text(readme, encoding="utf-8")

    return npz_path


def load_fig3_dataset(path: str | Path = "zenodo_repository/data/figure_3/figure3_data.npz") -> Dict[str, np.ndarray]:
    """Load the saved .npz archive as a plain dictionary."""
    archive = np.load(path, allow_pickle=False)
    return {key: archive[key] for key in archive.files}
