"""NRG occupations n_i(v) of the double dot along the common gate v.

Settings: block "nrg_occupation_sweep" of ../input/dqd_parameters.json
(thermodynamics only, no spectra; ~1-2 min and ~1.5 GB per gate value).
Output: ../output/dqd_nrg_occupation_sweep.npz

    python 03_occupation_sweep.py              # all gate values, n_jobs = all cores but one
    python 03_occupation_sweep.py --n-jobs 4
"""

import argparse
import os
import time

# few BLAS threads per worker; parallelism comes from joblib
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np
from joblib import Parallel, delayed

from common import GAMMA, OUTPUT, P, U_LIST, U12, V_LINE, model, settings_from, nm

S = P["nrg_occupation_sweep"]


def occ_job(v, z, nrg):
    r = nm.run_single_z(model(v), z, spectral=False, **nrg)
    ex = r["expect"]
    return v, z, ex[("n", 0)], ex[("n", 1)], ex[("nn", 0, 1)], r["time"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-jobs", type=int, default=-2)
    ap.add_argument("--out", default=str(OUTPUT / "dqd_nrg_occupation_sweep.npz"))
    a = ap.parse_args()

    nrg = settings_from(S)
    g = S["v_over_gamma"]
    v_sweep = np.round(np.arange(g["start"], g["stop"] + 1e-9, g["step"]), 6)
    zs = list(S["z"])
    jobs = [(v, z) for v in v_sweep for z in zs]
    nj = nm.resolve_n_jobs(a.n_jobs, len(jobs), mem_per_job_gb=1.5)
    print(f"NRG settings: {nrg}; gate sweep: {len(jobs)} NRG runs on {nj} workers", flush=True)
    t1 = time.time()
    out = Parallel(n_jobs=nj)(delayed(occ_job)(v, z, nrg) for v, z in jobs)
    print(f"sweep done in {time.time() - t1:.0f}s", flush=True)
    arr = np.array([o[:5] for o in out])
    n_sweep = np.zeros((len(v_sweep), 2)); nn_sweep = np.zeros(len(v_sweep))
    for k, v in enumerate(v_sweep):
        sel = arr[:, 0] == v
        n_sweep[k] = arr[sel][:, 2:4].mean(axis=0)
        nn_sweep[k] = arr[sel][:, 4].mean()
    np.savez_compressed(a.out, U_list=np.array(U_LIST), U12=U12, v_line=V_LINE,
                        T_over_gamma=nrg["T"] / GAMMA, gamma_over_D=GAMMA, Lam=nrg["Lam"],
                        Nkeep=nrg["Nkeep"], Nkeep_early=nrg["Nkeep_early"], n_early=nrg["n_early"],
                        Ecut=nrg["Ecut"], nshells=nrg["nshells"], sweep_z=np.array(zs),
                        v_sweep=v_sweep, n_sweep=n_sweep, n1n2_sweep=nn_sweep)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
