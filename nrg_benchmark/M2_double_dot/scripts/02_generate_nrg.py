"""Production NRG spectra of the double dot at v = v_line, with z averaging.

Settings: block "nrg_spectra" of ../input/dqd_parameters.json.  Every z shift
writes its raw FDM Lehmann histograms to ../output/checkpoints/z_kk_of_NN.npz;
existing checkpoints are reused, so rerunning the script with the archived
checkpoints only redoes the broadening (about a minute).  Output:

    ../output/dqd_nrg_spectra.npz     spectra (b = 0.25, 0.35, 0.50), occupations, self-energy
    ../output/dqd_nrg_quality.json    sum rules, Friedel values, n from the spectrum, clipping

A full recomputation takes ~15 min and up to ~18 GB of RAM per z shift
(Nkeep = 6000, 24000 in the first 7 shells), plus temporary disk space for
the shell eigenvectors.

    python 02_generate_nrg.py                     # reuse / complete the checkpoints
    python 02_generate_nrg.py --n-jobs 2          # two z shifts at a time (48 GB machine)
"""

import argparse
import json
import os
import time

early = argparse.ArgumentParser(add_help=False)
early.add_argument("--blas-threads", type=int, default=1)
early_args, _ = early.parse_known_args()
if early_args.blas_threads < 1:
    early.error("--blas-threads must be positive")
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = str(early_args.blas_threads)

import numpy as np
from joblib import Parallel, delayed

from common import GAMMA, OUTPUT, P, U_LIST, U12, V_LINE, model, settings_from, spectrum_metrics, nm

S = P["nrg_spectra"]
LINEAR = dict(linear_factor=S["linear_factor"], linear_eta_factor=S["linear_eta_factor"],
              linear_nbins=S["linear_nbins"], bins_per_decade=S["bins_per_decade"])


def save_checkpoint(path, result, z, nrg):
    data = dict(z=z, settings_json=json.dumps(nrg, sort_keys=True),
                linear_json=json.dumps(LINEAR, sort_keys=True),
                bins=result["bins"], lin_bins=result["lin_bins"],
                lin_eta=result["lin_eta"],
                n=np.array([result["expect"][("n", i)] for i in range(2)]),
                nn=result["expect"][("nn", 0, 1)],
                docc=np.array([result["expect"][("docc", i)] for i in range(2)]),
                nkept=result["nkept"], scale=result["scale"],
                run_seconds=result["time"])
    for i in range(2):
        for op in ("d", "O"):
            h = result["hist"][(op, i)]
            for suffix, val in zip(("pos", "neg", "total", "linear"), h):
                data[f"{op}{i}_{suffix}"] = val
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **data)
    os.replace(tmp, path)


def load_checkpoint(path, z, nrg):
    with np.load(path, allow_pickle=False) as data:
        if not np.isclose(float(data["z"]), z):
            raise ValueError(f"z mismatch in {path}")
        if str(data["settings_json"]) != json.dumps(nrg, sort_keys=True):
            raise ValueError(f"settings changed; use a different --run-name or remove {path}")
        if str(data["linear_json"]) != json.dumps(LINEAR, sort_keys=True):
            raise ValueError(f"linear-broadening settings changed in {path}")
        expect = {("n", i): float(data["n"][i]) for i in range(2)}
        expect[("nn", 0, 1)] = float(data["nn"])
        expect.update({("docc", i): float(data["docc"][i]) for i in range(2)})
        hist = {}
        for i in range(2):
            for op in ("d", "O"):
                hist[(op, i)] = tuple(data[f"{op}{i}_{suffix}"].copy()
                                      for suffix in ("pos", "neg", "total", "linear"))
        return dict(z=z, bins=data["bins"].copy(), lin_bins=data["lin_bins"].copy(),
                    lin_eta=float(data["lin_eta"]), hist=hist, expect=expect,
                    nkept=data["nkept"].copy(), scale=data["scale"].copy(),
                    time=float(data["run_seconds"]))


def run_z_job(k, total, z, nrg, checkpoint_dir):
    path = checkpoint_dir / f"z_{k:02d}_of_{total:02d}.npz"
    t0 = time.time()
    r = nm.run_single_z(model(), z, spectral=True, disk_backed=True,
                        **nrg, **LINEAR)
    save_checkpoint(path, r, z, nrg)
    return k, z, time.time() - t0, [r["expect"][("n", i)] for i in range(2)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nz", type=int, default=S["Nz"])
    ap.add_argument("--zs", type=float, nargs="+",
                    help="explicit z values; use 1.0 for a single-shift preview")
    ap.add_argument("--run-name", default="production",
                    help="anything else writes checkpoints_<name>/ and dqd_nrg_spectra_<name>.npz")
    ap.add_argument("--temp-over-gamma", type=float,
                    help="override T/gamma; lower T needs more Wilson shells")
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="parallel z runs; 2 is the suggested maximum on 48 GB")
    ap.add_argument("--mem-per-job-gb", type=float, default=18.0)
    ap.add_argument("--blas-threads", type=int, default=1,
                    help="BLAS/LAPACK threads per z worker")
    a = ap.parse_args()
    if a.nz < 1 or not a.run_name.replace("_", "").isalnum():
        ap.error("--nz must be >=1 and --run-name must be alphanumeric/underscore")
    if a.n_jobs == 0 or a.mem_per_job_gb <= 0:
        ap.error("--n-jobs must be nonzero and --mem-per-job-gb positive")
    if a.temp_over_gamma is not None and a.temp_over_gamma <= 0:
        ap.error("--temp-over-gamma must be positive")
    nrg = settings_from(S, None if a.temp_over_gamma is None else a.temp_over_gamma * GAMMA)
    zs = list(a.zs) if a.zs is not None else [(k + 1) / a.nz for k in range(a.nz)]
    if any(z <= 0 or z > 1 for z in zs) or len(set(zs)) != len(zs):
        ap.error("z values must be distinct and lie in (0, 1]")
    nz = len(zs)
    production = a.run_name == "production"
    checkpoint_dir = OUTPUT / ("checkpoints" if production else f"checkpoints_{a.run_name}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"NRG settings={nrg}; z={zs}; linear crossover={LINEAR}; "
          f"BLAS threads={a.blas_threads}", flush=True)
    print(f"Checkpoints: {checkpoint_dir}", flush=True)
    missing = []
    for k, z in enumerate(zs, 1):
        path = checkpoint_dir / f"z_{k:02d}_of_{nz:02d}.npz"
        if path.exists():
            load_checkpoint(path, z, nrg)  # validate before dispatch
            print(f"[{k}/{nz}] resume z={z:g}", flush=True)
        else:
            missing.append((k, z))
    if missing:
        workers = nm.resolve_n_jobs(a.n_jobs, len(missing),
                                    mem_per_job_gb=a.mem_per_job_gb)
        print(f"Running {len(missing)} missing z shifts on {workers} workers", flush=True)
        done = Parallel(n_jobs=workers, verbose=10, pre_dispatch=workers)(
            delayed(run_z_job)(k, nz, z, nrg, checkpoint_dir) for k, z in missing)
        for k, z, seconds, occupations in done:
            print(f"[{k}/{nz}] z={z:g} done in {seconds:.1f}s; n={occupations}", flush=True)
    results = [load_checkpoint(checkpoint_dir / f"z_{k:02d}_of_{nz:02d}.npz", z, nrg)
               for k, z in enumerate(zs, 1)]

    w_d = nm.default_grid(wmin=min(1e-9, nrg["T"] / 100), wmax=0.999,
                          per_decade=80)
    w = w_d / GAMMA
    m = model()
    b_ref = S["reference_b"]
    out = dict(w_over_gamma=w, U_list=np.asarray(U_LIST), U12=U12, v_line=V_LINE,
               gamma_over_D=GAMMA, T_over_gamma=nrg["T"] / GAMMA, Lam=nrg["Lam"],
               Ecut=nrg["Ecut"], Nkeep=nrg["Nkeep"],
               Nkeep_early=nrg["Nkeep_early"], n_early=nrg["n_early"],
               nshells=nrg["nshells"], Nz=nz, b=b_ref, zs=np.asarray(zs),
               n_nrg_z=np.array([[r["expect"][("n", i)] for r in results] for i in range(2)]),
               run_times=np.array([r["time"] for r in results]),
               linear_factor=LINEAR["linear_factor"],
               linear_eta_factor=LINEAR["linear_eta_factor"])
    out["n_nrg"] = np.mean(out["n_nrg_z"], axis=1)
    out["n1n2_nrg"] = np.mean([r["expect"][("nn", 0, 1)] for r in results])
    out["docc_nrg"] = np.mean([[r["expect"][("docc", i)] for i in range(2)] for r in results], axis=0)
    quality = dict(settings=nrg, Nz=nz, zs=zs, run_name=a.run_name,
                   n_nrg=out["n_nrg"].tolist(),
                   n_z_spread=np.ptp(out["n_nrg_z"], axis=1).tolist(),
                   dots={})
    for b in S["broadening_b"]:
        print(f"Broadening all z runs with b={b:g}...", flush=True)
        res = nm.broaden_and_sigma(m, results, w_d, b)
        tag = f"b{int(round(100*b)):03d}"
        out[f"gA4_{tag}"] = np.array([0.5 * GAMMA * res[("A_sigma", i)] for i in range(2)])
        quality["dots"][tag] = [spectrum_metrics(w, out[f"gA4_{tag}"][i],
                                                 0.5 * GAMMA * res[("A_raw", i)],
                                                 res[("Sigma", i)] / GAMMA,
                                                 out["n_nrg"][i]) for i in range(2)]
        if b == b_ref:
            out["gA4_nrg"] = out[f"gA4_{tag}"]
            out["gA4_raw"] = np.array([0.5 * GAMMA * res[("A_raw", i)] for i in range(2)])
            out["gA4_unclipped"] = np.array([0.5 * GAMMA * res[("A_unclipped", i)] for i in range(2)])
            out["Sigma_over_gamma"] = np.array([res[("Sigma", i)] / GAMMA for i in range(2)])
            out["sumrule"] = np.array([res[("sumrule", i)] for i in range(2)])
    if not np.allclose(out["sumrule"], 1.0, atol=1e-6):
        raise RuntimeError("Lehmann sum rule failed; reference file not written")
    stem = "dqd_nrg_spectra" if production else f"dqd_nrg_spectra_{a.run_name}"
    qstem = "dqd_nrg_quality" if production else f"dqd_nrg_quality_{a.run_name}"
    np.savez_compressed(OUTPUT / f"{stem}.npz", **out)
    (OUTPUT / f"{qstem}.json").write_text(json.dumps(quality, indent=2) + "\n")
    print(f"Saved {stem}.npz and {qstem}.json", flush=True)
    for tag, dots in quality["dots"].items():
        print(tag, "n_from_A=", [round(d["n_from_A"], 6) for d in dots],
              "A_near_zero=", [round(d["A_near_zero"], 6) for d in dots], flush=True)


if __name__ == "__main__":
    main()
