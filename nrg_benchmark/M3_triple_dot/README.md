# M = 3: triple dot of Fig. 3(b)

    U_1 = 4, U_2 = 6, U_3 = 8,  U_ij = 2 (all pairs),  v = -10,  T = 1e-5

Units are gamma. Each dot is coupled to its own reservoir, with no interdot
hopping.

**Status: complete. The spectra are in `output/tqd_nrg.npz` and the occupation
sweep `n_i(v)` is in `output/tqd_nrg_sweep.npz`, both with the production
settings.**

## Layout

| path | content |
|---|---|
| `input/tqd_parameters.json` | model and NRG settings of the production run |
| `input/cluster_run.md` | how the run was launched on the cluster: commands per task, resources, environment, and how the settings were chosen |
| `scripts/nrg_multidot.py` | the NRG engine, identical to the copy the cluster run uses |
| `scripts/test_nrg_ed.py` | engine vs exact diagonalization (~20 s) |
| `scripts/01_run_nrg_tqd.py` | the calculation. One task per z shift (`--z-index`), `--assemble`, the gate sweep (`--gate-index`, `--assemble-sweep`), `--probe` and `--selftest`. Its defaults are the production settings |
| `output/` | results, listed below |

## NRG settings

`Lambda = 16`, `Nz = 8`, `Nkeep = 4000` with 48000 states in the first 6
shells, `Ecut = 10`, 19 shells, FDM at `T = 1e-5 gamma`, log-Gaussian
broadening `b = 0.3` with a linear Gaussian of width `T` below `T`, and the
self-energy trick. See `input/cluster_run.md` for how these were chosen. With
three channels, the occupation split between dots 1 and 2 converges slowly in
the early-shell truncation, which is why `Lambda` and `Nkeep_early` are
large.

## Reproduce

```bash
cd scripts
python test_nrg_ed.py                        # ~20 s: engine vs exact diagonalization
python 01_run_nrg_tqd.py --selftest          # ~1 min: environment + pipeline
python 01_run_nrg_tqd.py --z-index K         # K = 1..8, one core, ~7 h and ~77 GB each (see input/cluster_run.md)
python 01_run_nrg_tqd.py --assemble          # -> ../output/tqd_nrg.npz
python 01_run_nrg_tqd.py --gate-index K --checkpoints ../output/checkpoints_sweep     # K = 1..31
python 01_run_nrg_tqd.py --assemble-sweep --checkpoints ../output/checkpoints_sweep  # -> ../output/tqd_nrg_sweep.npz
```

## Results

From `output/tqd_nrg.npz`. `A_i(0)` is taken at the smallest resolved
frequency, `|omega| ~ T`.

| quantity | dot | NRG |
|---|---|---|
| occupation `n_i` | 1 | 1.832 |
| | 2 | 1.108 |
| | 3 | 1.004 |
| spread of `n_i` over the 8 shifts | 1 | 0.024 (0.002 without `z = 1/8`) |
| | 2 | 0.004 |
| | 3 | 0.0007 |
| `gamma A_i(0)/4` (Friedel value of `n_i`) | 1 | 0.021 (0.022) |
| | 2 | 0.305 (0.309) |
| | 3 | 0.318 (0.318) |
| Kondo peak HWHM / gamma | 2 | 0.060 |
| | 3 | 0.0057 |

Double occupancies: `<n_i,up n_i,dn> = 0.836, 0.150, 0.052`. Dot 1 is
nearly doubly occupied and has no Kondo peak. The Lehmann sum rule holds to
1e-13 for every shift.

## Outputs

| file | content |
|---|---|
| `output/tqd_nrg.npz` | written by `--assemble`. `w_over_gamma`, `gA4_nrg[i]` (gamma*A_i/4, self-energy trick), `gA4_raw[i]` (plain FDM), `Sigma_over_gamma[i]`, `n_nrg`, `n_nrg_z` (per shift), `docc_nrg`, `nn_ij`, `sumrule`, the raw FDM histograms `hist_pos`/`hist_neg` `[z, dot, (d, O), bin]` on `hist_bins_D`, and all settings |
| `output/tqd_nrg_sweep.npz` | written by `--assemble-sweep`. `v_sweep` (v/gamma = -25, -24, ..., 5), `n_sweep[v, i]`, and the model parameters. `z = 1`, same `Lambda`, `Nkeep` and early shells as the spectra. At `v = -10` it gives `n = (1.834, 1.107, 1.004)`, within 0.003 of the z-averaged `n_nrg` in `tqd_nrg.npz` |

The per-shift checkpoints (`output/checkpoints/z_kk_of_08.npz`) are not
included. Their raw FDM histograms are stored in `tqd_nrg.npz` as
`hist_pos`/`hist_neg`.
