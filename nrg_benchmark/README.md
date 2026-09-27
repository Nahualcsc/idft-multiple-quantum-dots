# NRG benchmark for the multidot spectra of Fig. 3

This folder supplements the dataset *Numerical data for "Spectral and
transmission properties of multiple correlated quantum dots made simple"*
(N. Sobrino and S. Kurth, [doi:10.5281/zenodo.20154932](https://doi.org/10.5281/zenodo.20154932)).
It contains numerical renormalization group (NRG) reference calculations for
the Kondo-regime panels of Fig. 3 with `M = 2` and `M = 3` dots: the NRG
code, the input parameters and the outputs.

## Layout

```
nrg_benchmark/
├── README.md                     this file
├── requirements.txt
├── M2_double_dot/                Fig. 3(a): complete
│   ├── README.md                 settings, convergence, results, how to reproduce
│   ├── input/                    dqd_parameters.json (read by the scripts)
│   ├── scripts/                  nrg_multidot.py (NRG engine), test_nrg_ed.py, 01-03 run scripts
│   └── output/                   NRG spectra and occupations, raw FDM checkpoints, convergence table
└── M3_triple_dot/                Fig. 3(b): complete
    ├── README.md
    ├── input/                    tqd_parameters.json, cluster_run.md (how the cluster run was set up)
    ├── scripts/                  nrg_multidot.py (NRG engine), test_nrg_ed.py, run script
    └── output/                   NRG spectra (tqd_nrg.npz) and occupations (tqd_nrg_sweep.npz)
```

## Model and conventions

```
H = sum_i v n_i + sum_i U_i n_i,up n_i,dn + sum_{i<j} U_ij n_i n_j
    + sum_{i,s} V (d^+_{i s} c_{i s} + h.c.) + sum_i H_lead,i
```

Each dot `i` has its own reservoir, so `Gamma = gamma * identity`, and there
is no interdot hopping. This is the model of Fig. 3. All energies are in units
of `gamma = 2 pi V^2 rho_0`, the full level width. The conduction band is flat
with half-width `D = 100 gamma`, which is the wide-band limit used in the manuscript.
The temperature is `T = 1e-5 gamma`.

Spectra are stored as `gamma A_i(omega)/4`, the quantity plotted in the
manuscript. It equals `(gamma/2) A_i^NRG(omega)` with `A_i^NRG` the spectral
function per spin, and the unitary limit is `1/pi`.

## NRG method

- Campo–Oliveira logarithmic discretization with z averaging. The Wilson
  chain is tridiagonalized by Lanczos in multiprecision.
- Abelian symmetry `U(1)^(2M)`. The particle number of each flavour
  `(dot, spin)` is conserved, because the dots interact only through
  densities and each has its own channel.
- Truncation: `Nkeep` states per shell, and more (`Nkeep_early`) in the first
  `n_early` shells, where the multichannel Wilson chain is not yet scale
  separated.
- Full-density-matrix NRG at finite `T` (Weichselbaum and von Delft), so the
  spectral sum rule holds to about 1e-15.
- Self-energy trick (Bulla, Hewson and Pruschke), generalized to the interdot
  interaction:
  `F_i = << (U_i n_i,dn + sum_j U_ij n_j) d_i,up ; d^+_i,up >>` and
  `Sigma_i = F_i / G_i`.
- Log-Gaussian broadening of the discrete FDM weights. Transitions with
  `|omega|` of order `T` get a linear Gaussian of width `T` instead.

Each system folder contains the engine `nrg_multidot.py` that produced its
results, unchanged. The two copies have the same Hamiltonian, Wilson chain,
NRG iteration and FDM. They differ only in the thermal-window broadening:

- `M = 2`: transitions with `|omega| < 4T` are binned on a linear grid
  centred at zero.
- `M = 3`: log bins with centres below `T` are broadened linearly.

The `M = 3` copy also records the truncation energy of each shell. Both
treatments change the spectra only at `|omega| ~ T`.

`test_nrg_ed.py`, in both script folders, checks the engine against exact
diagonalization of a short Wilson chain without truncation. Spectrum,
occupations and Lehmann weights of `d` and `F` agree to 1e-14. It takes
about 20 s.

## Requirements and quick start

Python 3 with the packages in `requirements.txt`. It was tested with Python
3.12.9, numpy 2.3.2, scipy 1.17.1, mpmath 1.3.0 and joblib 1.5.3.

```bash
pip install -r requirements.txt
cd M2_double_dot/scripts
python test_nrg_ed.py            # ~20 s: engine vs exact diagonalization
python 02_generate_nrg.py        # ~2 s: rebroaden the archived FDM checkpoints -> spectra
```

Recomputing the NRG from scratch is much more expensive. The `M = 2` spectra
take about 15 min and up to 18 GB per z shift. The `M = 3` run needs a
cluster; see `M3_triple_dot/input/cluster_run.md`.

## Main results

`T = 1e-5` and `U_ij = 2` (units of gamma). Details in the README of each
system folder.

| | dot | `M = 2`: `U = (4, 6)`, `v = -5` | `M = 3`: `U = (4, 6, 8)`, `v = -10` |
|---|---|---|---|
| occupation `n_i` | 1 | 1.190 | 1.832 |
| | 2 | 0.974 | 1.108 |
| | 3 | | 1.004 |
| `gamma A_i(0)/4` | 1 | 0.291 | 0.021 |
| | 2 | 0.317 | 0.305 |
| | 3 | | 0.318 |

In both systems `A_i(0)` is close to the Friedel value `sin^2(pi n_i/2)/pi`
of the NRG occupation.

## References

- V. L. Campo and L. N. Oliveira, Phys. Rev. B **72**, 104432 (2005): discretization.
- A. Weichselbaum and J. von Delft, Phys. Rev. Lett. **99**, 076402 (2007): full-density-matrix NRG.
- R. Bulla, A. C. Hewson and Th. Pruschke, J. Phys.: Condens. Matter **10**, 8365 (1998): self-energy trick.
- R. Bulla, T. A. Costi and Th. Pruschke, Rev. Mod. Phys. **80**, 395 (2008): NRG review and log-Gaussian broadening.
