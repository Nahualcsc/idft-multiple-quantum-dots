# How the M = 3 production run was set up

The triple-dot NRG was run on a SLURM cluster as independent single-core
tasks. The batch files themselves are not included. This page records what
they ran and the resources they requested. The script is
`../scripts/01_run_nrg_tqd.py`, and its defaults equal the flags below, so
`--z-index K` with no other options reproduces a task.

## Environment

- Intel Xeon Gold 6230 (2.1 GHz) nodes, one core per task.
- Python 3.12.9, numpy 2.3.2, scipy 1.17.1, mpmath 1.3.0. joblib is only
  needed for `--all` / `--all-gates`.
- One thread per task:
  `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=VECLIB_MAXIMUM_THREADS=1`.
  Extra BLAS threads do not speed up these block sizes. The parallelism comes
  from running many tasks.
- Check a new machine with
  `python 01_run_nrg_tqd.py --selftest` (about 1 min) and
  `python test_nrg_ed.py` (about 20 s), both in `../scripts/`.

## Settings

| | value |
|---|---|
| model | `U = (4, 6, 8)`, `U_ij = 2` (all pairs), `v = -10`, `T = 1e-5` (units of gamma), `gamma/D = 0.01` |
| discretisation | Campo–Oliveira, `Lambda = 16`, `Nz = 8` shifts `z = k/8` |
| truncation | `Nkeep = 4000`, `Ecut = 10`; `Nkeep_early = 48000` in the first `n_early = 6` shells |
| shells | 19, from `ceil(2 ln(0.5/(1e-3 T/D)) / ln Lambda) + 2` |
| broadening | log-Gaussian `b = 0.3`, linear Gaussian of width `T` below `omega0 = T` |

## Spectra: one task per z shift

8 tasks (`K = 1..8`), each with 1 node, 1 task, 1 CPU, 120 GB of memory,
48 h of wall time, and about 0.4 TB of scratch on the shared filesystem for
the shell eigenvectors. The scratch is deleted when the task ends.

```bash
python 01_run_nrg_tqd.py --nz 8 --lam 16 --nkeep 4000 --n-early 6 --nkeep-early 48000 \
       --checkpoints checkpoints_L16 --scratch "$SCRATCH" --z-index K
```

On the cluster, `K` came from `SLURM_ARRAY_TASK_ID`, which the script reads
when `--z-index` is not given. The eight tasks took 5.3–8.5 h each (about
56 core hours in total) with a peak memory of 76–77 GB per task, measured by
SLURM (`MaxRSS`). The requests above were set generously from a
thermodynamics-only probe at the same settings (7.6 h, 46 GB, no spectra).

After all 8 tasks finish:

```bash
python 01_run_nrg_tqd.py --assemble --nz 8 --lam 16 --nkeep 4000 --n-early 6 --nkeep-early 48000 \
       --checkpoints checkpoints_L16          # -> tqd_nrg.npz
```

`--assemble` refuses to mix checkpoints written with different settings. It
also refuses to write the file if the Lehmann sum rule fails.

## Occupation sweep n_i(v): one task per gate value

31 tasks (`K = 1..31`, gate `v = -25, -24, ..., 5`). Each task uses 1 CPU,
80 GB of memory and 24 h of wall time. It runs `z = 1` only, computes no
spectra and needs no scratch. The 31 tasks took 3.0–6.6 h each, with a peak memory of 41–42 GB per task.

```bash
python 01_run_nrg_tqd.py --gate-index K --lam 16 --nkeep 4000 --n-early 6 --nkeep-early 48000 \
       --checkpoints checkpoints_L16_sweep
python 01_run_nrg_tqd.py --assemble-sweep --checkpoints checkpoints_L16_sweep   # -> tqd_nrg_sweep.npz
```

## How the settings were chosen

Three channels are much harder than two. Each Wilson site multiplies the
Hilbert space by 64. At this gate, `n_1 + n_2` is stable, but the split
between dots 1 and 2 is soft. It is set by the first shells, where the chain
is not yet scale separated.

A first ladder at `Lambda = 8`, `z = 1` (see the header of the script)
converged `z = 1` from about 3000 kept states on, but different shifts
disagreed. The settings were then chosen from thermodynamics-only probe runs
(`--probe Z`, no spectra, no scratch) at `Nkeep = 4000`, comparing
`z = 1/16` with `z = 1/2`:

| Lambda | n_early | Nkeep_early | memory / time requested |
|---|---|---|---|
| 8 | 8 | 24000 | 32 GB / 3 h |
| 8 | 8 | 48000 | 80 GB / 8 h |
| 8 | 8 | 96000 | 180 GB / 24 h |
| 16 | 6 | 24000 | 32 GB / 3 h |
| **16** | **6** | **48000** | 80 GB / 8 h |

`Lambda = 16` with 48000 states in the first 6 shells is the setting where the
two shifts nearly agree. It is the production setting.

```bash
python 01_run_nrg_tqd.py --probe 0.0625 --lam 16 --nkeep 4000 --n-early 6 --nkeep-early 48000 \
       --checkpoints checkpoints_probes
```
