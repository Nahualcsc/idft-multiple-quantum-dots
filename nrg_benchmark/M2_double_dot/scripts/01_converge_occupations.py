"""Occupation convergence in the number of kept states, at fixed z shifts.

Runs the thermodynamic NRG (no spectra) at v = v_line for a few z values and
one truncation, and writes a JSON report.  With --previous (a report of this
script at a smaller truncation) it also gives the change at fixed z.  The
ladder used for the archived spectra is summarised in
../output/convergence/dqd_occupation_convergence.csv.

    python 01_converge_occupations.py --nkeep 6000 --nkeep-early 24000 --n-jobs 1
"""

import argparse
import json
import os
import time
from pathlib import Path

early = argparse.ArgumentParser(add_help=False)
early.add_argument("--blas-threads", type=int, default=1)
early_args, _ = early.parse_known_args()
if early_args.blas_threads < 1:
    early.error("--blas-threads must be positive")
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = str(early_args.blas_threads)

import numpy as np
from joblib import Parallel, delayed

from common import OUTPUT, P, model, settings, nm

Z_VALUES = (0.125, 0.25, 1.0)


def previous_occupations(path):
    data = json.loads(path.read_text())
    return {float(r["z"]): np.asarray(r["n"], float) for r in data.get("results", [])}


def job(z, nrg):
    r = nm.run_single_z(model(), z, spectral=False, **nrg)
    return {"z": z, "n": [float(r["expect"][("n", i)]) for i in range(2)],
            "n1n2": float(r["expect"][("nn", 0, 1)]),
            "seconds": r["time"]}


def main():
    s = P["nrg_spectra"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nkeep", type=int, default=s["Nkeep"])
    ap.add_argument("--nkeep-early", type=int, default=s["Nkeep_early"])
    ap.add_argument("--n-early", type=int, default=s["n_early"])
    ap.add_argument("--lam", type=float, default=s["Lambda"])
    ap.add_argument("--ecut", type=float, default=s["Ecut"])
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--blas-threads", type=int, default=1,
                    help="BLAS/LAPACK threads per job; useful when --n-jobs 1")
    ap.add_argument("--z", type=float, nargs="+", default=Z_VALUES,
                    help="discretization shifts (default: 0.125 0.25 1)")
    ap.add_argument("--previous", type=Path, help="report of a smaller truncation, for the fixed-z change")
    ap.add_argument("--out", type=Path, help="default: ../output/convergence/occupations_nk<N>_early<N>.json")
    ap.add_argument("--tolerance", type=float, default=1e-3,
                    help="occupation change at fixed z regarded as converged")
    a = ap.parse_args()
    if a.n_jobs == 0 or a.nkeep <= 0 or a.nkeep_early < a.nkeep:
        ap.error("invalid worker or kept-state count")
    if not a.z or any(z <= 0 or z > 1 for z in a.z):
        ap.error("every z must lie in (0, 1]")
    zs = tuple(a.z)
    nrg = settings(a.nkeep, a.nkeep_early, a.lam, a.ecut, a.n_early)
    prior = previous_occupations(a.previous) if a.previous is not None and a.previous.exists() else {}
    out = a.out or OUTPUT / "convergence" / f"occupations_nk{a.nkeep}_early{a.nkeep_early}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"settings={nrg}; z={zs}; previous={a.previous.name if prior else 'none'}", flush=True)
    t0 = time.time()
    workers = nm.resolve_n_jobs(a.n_jobs, len(zs), mem_per_job_gb=6.0)
    print(f"Running {workers} occupation jobs with {a.blas_threads} BLAS threads each", flush=True)
    rows = Parallel(n_jobs=workers, verbose=10, pre_dispatch=workers)(
        delayed(job)(z, nrg) for z in zs)
    for r in rows:
        p = prior.get(r["z"])
        r["previous_n"] = p.tolist() if p is not None else None
        r["delta"] = (np.asarray(r["n"]) - p).tolist() if p is not None else None
        print(f"z={r['z']}: n={r['n']}, delta={r['delta']}", flush=True)
    deltas = [abs(x) for r in rows if r["delta"] is not None for x in r["delta"]]
    max_change = max(deltas) if deltas else None
    report = dict(settings=nrg, results=rows, previous=a.previous.name if prior else None,
                  blas_threads=a.blas_threads,
                  max_fixed_z_change=max_change, tolerance=a.tolerance,
                  passed=(max_change is not None and max_change <= a.tolerance),
                  z_spread=np.ptp([r["n"] for r in rows], axis=0).tolist(),
                  elapsed_seconds=time.time() - t0)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {out}; max fixed-z change={max_change} (tolerance {a.tolerance})", flush=True)


if __name__ == "__main__":
    main()
