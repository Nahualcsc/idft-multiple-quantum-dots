# M = 2: double dot of Fig. 3(a)

    U_1 = 4, U_2 = 6, U_12 = 2,  v_1 = v_2 = v = -5,  T = 1e-5

Units are gamma, and `gamma/D = 0.01`. Each dot is coupled to its own
reservoir, with no interdot hopping.

## Layout

| path | content |
|---|---|
| `input/dqd_parameters.json` | model, NRG settings for the spectra and for the occupation sweep. Read by every script |
| `scripts/nrg_multidot.py` | the NRG engine that produced these results |
| `scripts/test_nrg_ed.py` | engine vs exact diagonalization (~20 s) |
| `scripts/common.py` | parameters, paths, model, diagnostics |
| `scripts/01_converge_occupations.py` | occupations at fixed z vs kept states (convergence ladder) |
| `scripts/02_generate_nrg.py` | production spectra: one checkpoint per z shift, then broadening and the self-energy trick |
| `scripts/03_occupation_sweep.py` | NRG occupations `n_i(v)` along the gate |
| `output/` | results, listed below |

## NRG settings

| | spectra (`02`) | occupation sweep (`03`) |
|---|---|---|
| discretization | `Lambda = 3`, `Nz = 8` (`z = k/8`) | `Lambda = 3`, `z = 1` |
| kept states | `Nkeep = 6000`, 24000 in the first 7 shells | `Nkeep = 2500`, 8000 in the first 7 shells |
| cutoff, shells | `Ecut = 10`, 43 shells | `Ecut = 10`, 43 shells |
| broadening | log-Gaussian `b = 0.35` (also 0.25, 0.50); linear Gaussian of width `T` for `|omega| < 4T` | — |

## Convergence

`output/convergence/dqd_occupation_convergence.csv` lists the occupations at
`v = -5` for each truncation and shift. With two channels, each Wilson site
multiplies the Hilbert space by 16. The first chain hoppings are not yet
scale separated, so extra states in the first 7 shells are essential. With a
uniform `Nkeep = 4000`, `n_1` varies with `z` between 1.170 and 1.188.

- z averages: `n_1` = 1.1899 (2500/8000) and 1.1900 (6000/24000); `n_2` =
  0.9733 and 0.9738.
- The spread over the 8 shifts at 6000/24000 is 5e-4 (`n_1`) and 1.5e-4 (`n_2`).
- At fixed `z = 1`, `n_1` changes by less than 1e-4 from 4000/16000 on.
- At the smallest shift, `z = 0.125`, `n_1` still changes by up to 0.003 between
  consecutive truncations, and not monotonically: 1.1899, 1.1925, 1.1900 and 1.1873 for 4000, 5000, 6000 and
  7000 kept states.
- The occupations do not depend on `Lambda`. At `z = 1` and `Nkeep = 8000`,
  `Lambda = 2.5 / 3 / 4` give `n_1 = 1.1902 / 1.1897 / 1.1899` and
  `n_2 = 0.9737 / 0.9738 / 0.9738`.

The occupations are therefore reliable to about 0.003 or better. The Kondo
widths are the most truncation-sensitive quantity: between 2500 and 6000 kept
states, the dot-2 HWHM went from 0.032 to 0.029 gamma. The spectra do not
depend on the broadening: `b = 0.25, 0.35, 0.50` give the same `A(0)` to 5e-5
(`output/dqd_nrg_quality.json`). The Lehmann sum rule holds to 1e-10 for
every shift. The single-channel SIAM with the same `eps` and `U` has
occupations independent of `z` and `Lambda` (spread 3e-5).

## Results

NRG at 6000/24000 kept states, 8 shifts and `b = 0.35` (`output/dqd_nrg_spectra.npz`):

| quantity | dot | NRG |
|---|---|---|
| occupation `n_i` | 1 | 1.190 |
| | 2 | 0.974 |
| `gamma A_i(0)/4` (Friedel value of `n_i`) | 1 | 0.291 (0.291) |
| | 2 | 0.317 (0.318) |
| Kondo peak HWHM / gamma | 1 | 0.199 |
| | 2 | 0.029 |
| Kondo peak maximum at omega / gamma | 1 | -0.033 |
| | 2 | 0.000 |
| side peaks, omega / gamma : height | 1 | -2.80 : 0.040 |
| | 2 | -2.04 : 0.054, +3.04 : 0.053 |

Double occupancies: `<n_1,up n_1,dn> = 0.242`, `<n_2,up n_2,dn> = 0.059`.
`A_i(0)` is the Friedel value of the NRG occupation `n_i`.

## Outputs

| file | content |
|---|---|
| `dqd_nrg_spectra.npz` | NRG at `v = -5`: `w_over_gamma`; `gA4_nrg[i]` = gamma*A_i/4 (self-energy trick, `b = 0.35`); `gA4_b025`, `gA4_b035`, `gA4_b050`; `gA4_raw` (plain FDM); `gA4_unclipped` (without the causality clip of Im Sigma); `Sigma_over_gamma[i]`; `n_nrg`, `n_nrg_z[i, z]`, `n1n2_nrg`, `docc_nrg`; `sumrule[i, z]`; all settings |
| `dqd_nrg_quality.json` | per broadening and dot: FDM occupation, occupation from the spectrum, spectral integrals, `A(0)` vs Friedel, fraction of clipped Im Sigma |
| `dqd_nrg_occupation_sweep.npz` | `v_sweep` (-25..5 in steps of 1), `n_sweep[v, i]`, `n1n2_sweep`, settings |
| `checkpoints/z_kk_of_08.npz` | raw FDM Lehmann histograms of `d` and `F` for each shift (log bins `bins` in units of D, linear bins `lin_bins` for `|omega| < 4T`), occupations, kept states per shell. `02_generate_nrg.py` rebroadens them in seconds |
| `convergence/dqd_occupation_convergence.csv` | occupation ladder in kept states and z |

## Reproduce

```bash
cd scripts
python test_nrg_ed.py              # ~20 s
python 02_generate_nrg.py          # ~2 s with the archived checkpoints; without them ~15 min and <=18 GB per shift
python 03_occupation_sweep.py      # ~6 min on 15 cores
```

Without the checkpoints, `02_generate_nrg.py --n-jobs 2` runs two shifts at a
time on a 48 GB machine. Each shift spills its shell eigenvectors to a
temporary directory. The convergence ladder is rerun with, for example,
`python 01_converge_occupations.py --nkeep 7000 --nkeep-early 28000 --z 0.125 --n-jobs 1`.
